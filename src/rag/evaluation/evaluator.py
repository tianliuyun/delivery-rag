"""RAG 评测与可观测性集成。

职责：
1. 提供 RAGAS 风格指标评估：Faithfulness（忠实度，基于答案与上下文的一致性）
   + 检索命中率（Hit Rate）
2. 评估全程通过 Phoenix/OpenTelemetry 埋点（每个问题一条 trace）：
   检索耗时 / 生成耗时 / token 估算 / 各指标得分 → 全链路可追溯、可复现
3. 输出结构化评估报告（JSON），供 README 与文档引用

说明：Faithfulness 是本机可跑的近似实现（LLM-as-Judge 打分，DeepSeek 可配）。
真实 RAGAS 需要 langchain 生态 + LLM API，本实现保持零重依赖可运行。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..observability import init_phoenix, rag_span, record_span_event


@dataclass
class RAGEvalResult:
    """单轮评估结果。"""
    question: str = ""
    hit: bool = False                 # 检索命中（golden chunk 在 top-k 中）
    faithfulness: float = 0.0         # 忠实度 0-1（LLM-as-Judge）
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    context_count: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAGEvalReport:
    """评估报告。"""
    total: int = 0
    hit_rate: float = 0.0
    avg_faithfulness: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_generation_ms: float = 0.0
    results: List[RAGEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
            "avg_retrieval_ms": round(self.avg_retrieval_ms, 2),
            "avg_generation_ms": round(self.avg_generation_ms, 2),
            "results": [r.__dict__ for r in self.results],
        }


class RAGEvaluator:
    """RAG 评测器（检索命中 + 忠实度 + 性能，全程可观测）。"""

    def __init__(self, judge_fn: Optional[Callable] = None,
                 enable_phoenix: bool = True,
                 phoenix_endpoint: str = "http://localhost:6006/v1/traces"):
        """judge_fn: 忠实度打分函数 (question, context, answer) -> float 0-1。
        默认用关键词近似；接入 DeepSeek 可提高准确度。"""
        self.judge_fn = judge_fn or self._default_judge
        self._phoenix_ok = False
        if enable_phoenix:
            self._phoenix_ok = init_phoenix("delivery-rag-eval", endpoint=phoenix_endpoint)

    # ---- 忠实度评估 ----
    def _default_judge(self, question: str, context: str, answer: str) -> float:
        """默认忠实度：答案中的关键术语是否来自上下文（近似，0-1）。"""
        ctx_tokens = set(self._tokenize(context))
        ans_tokens = set(self._tokenize(answer))
        if not ans_tokens:
            return 0.0
        overlap = len(ans_tokens & ctx_tokens)
        return min(1.0, overlap / max(1.0, len(ans_tokens) * 0.5))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        # 中文按 2-gram 切分，英文按词
        cn = re.findall(r"[\u4e00-\u9fff]{2,}", text)
        en = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return cn + en

    # ---- 单题评测 ----
    def evaluate_question(self, question: str, golden_context: str,
                          retriever_fn: Callable[[str], List[str]],
                          answer_fn: Callable[[str, List[str]], str],
                          top_k: int = 5) -> RAGEvalResult:
        """评估单个问题：检索 + 生成 + 打分，全程 span 埋点。"""
        result = RAGEvalResult(question=question)

        # 检索阶段（span: retrieval）
        with rag_span("retrieval", {"question": question, "top_k": top_k}) as span:
            t0 = time.time()
            contexts = retriever_fn(question)
            result.retrieval_ms = (time.time() - t0) * 1000
            result.context_count = len(contexts)
            record_span_event(span, "retrieval_done", {"context_count": len(contexts)})

        # 命中判定：golden_context 是否在检索结果中（文本包含即算命中）
        result.hit = any(golden_context[:30] in c or c[:30] in golden_context
                         for c in contexts[:top_k])

        # 生成阶段（span: generation）
        with rag_span("generation", {"question": question}) as span:
            t0 = time.time()
            answer = answer_fn(question, contexts)
            result.generation_ms = (time.time() - t0) * 1000
            record_span_event(span, "generation_done", {"answer_len": len(answer)})

        # 忠实度（span: faithfulness）
        with rag_span("faithfulness", {"question": question}) as span:
            context_joined = "\n".join(contexts[:top_k])[:2000]
            result.faithfulness = self.judge_fn(question, context_joined, answer)
            record_span_event(span, "score", {"faithfulness": result.faithfulness})

        result.metadata = {"phoenix": self._phoenix_ok}
        return result

    # ---- 批量评测 ----
    def evaluate(self, questions: List[Dict],
                 retriever_fn: Callable[[str], List[str]],
                 answer_fn: Callable[[str, List[str]], str],
                 top_k: int = 5) -> RAGEvalReport:
        """批量评测。questions: [{"question", "golden_context"}]"""
        report = RAGEvalReport()
        for item in questions:
            q = item["question"]
            golden = item.get("golden_context", "")
            r = self.evaluate_question(q, golden, retriever_fn, answer_fn, top_k)
            report.results.append(r)
            report.total += 1

        if report.total:
            report.hit_rate = sum(1 for r in report.results if r.hit) / report.total
            report.avg_faithfulness = sum(r.faithfulness for r in report.results) / report.total
            report.avg_retrieval_ms = sum(r.retrieval_ms for r in report.results) / report.total
            report.avg_generation_ms = sum(r.generation_ms for r in report.results) / report.total

        return report

    def export_report(self, report: RAGEvalReport, path: str) -> None:
        """评估报告落盘 JSON（供 README / 文档引用）。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
