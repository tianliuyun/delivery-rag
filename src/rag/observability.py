"""生产级 AI 可观测性（Phoenix / OpenTelemetry 接入）。

职责：
1. 初始化 OpenTelemetry TracerProvider，通过 OTLP exporter 发送到 Phoenix
   （Phoenix 默认 http://localhost:6006/v1/traces，支持 LangSmith 风格 trace 可视化）
2. 提供 `rag_span` 上下文管理器，包装 RAG pipeline 的 检索/重排/生成 三段，
   记录每段的耗时与 token 用量，形成可追溯的 trace
3. 提供 evaluation 埋点：RAGAS 评估时逐题出 span，trace 全链路可复现

用法：
    from rag.observability import init_phoenix, rag_span
    init_phoenix(service_name="delivery-rag")
    with rag_span("hybrid_search", attributes={"top_k": 50}) as span:
        results = hybrid.search(...)
        span.set_attribute("num_results", len(results))

依赖：opentelemetry-sdk / opentelemetry-exporter-otlp-proto-http
（随 arize-phoenix 一并安装）
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

_TP = None  # TracerProvider 单例


def init_phoenix(service_name: str = "delivery-rag",
                 endpoint: str = "http://localhost:6006/v1/traces",
                 enabled: bool = True) -> bool:
    """初始化 OpenTelemetry → Phoenix 链路。

    返回 False 表示不可用（未装依赖或连接失败），调用方应降级为无观测运行。
    """
    global _TP
    if not enabled:
        _TP = None
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource = Resource(attributes={SERVICE_NAME: service_name})
        _TP = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint)
        _TP.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_TP)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Phoenix 初始化失败，降级为无观测: {type(e).__name__}: {e}")
        _TP = None
        return False


def get_tracer(name: str = "delivery-rag"):
    if _TP is None:
        return None
    from opentelemetry import trace
    return trace.get_tracer(name)


@contextmanager
def rag_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[Any]:
    """RAG 流程 span 上下文管理器。

    用法：with rag_span("hybrid_search", {"top_k": 50}) as span: ...
    若未初始化 Phoenix（_TP=None），则退化为计时器，不影响业务逻辑。
    """
    tracer = get_tracer()
    t0 = time.time()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span
        span.set_attribute("duration_ms", round((time.time() - t0) * 1000, 2))


def record_span_event(span, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """在 span 上记录一个事件（如检索命中数、生成 token 数）。"""
    if span is None:
        return
    try:
        span.add_event(name, attributes or {})
    except Exception:  # noqa: BLE001
        pass
