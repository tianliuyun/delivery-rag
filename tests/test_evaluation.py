"""RAG 评测（RAGAS 风格 + Phoenix 可观测）测试。

验证：
- RAGEvaluator 能对单题产出 hit/faithfulness/耗时
- 批量评测能算出汇总指标
- Phoenix 未运行时自动降级（不抛异常）
- 评测报告可 JSON 导出
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from rag.evaluation import RAGEvaluator, RAGEvalReport

EVAL_QS = [
    {"question": "桌面云登录慢第一步查什么？",
     "golden_context": "检查客户端到服务器的网络延迟"},
    {"question": "EDS 磁盘故障看什么？",
     "golden_context": "查看磁盘 SMART 信息"},
]


def _retriever(q):
    # 模拟检索：golden 命中与否由问题关键字决定
    if "桌面云" in q:
        return ["检查客户端到服务器的网络延迟（ping 测试）"]
    return ["查看磁盘 SMART 信息"]


def _answer(q, contexts):
    return contexts[0][:60] if contexts else "无"


class TestRAGEvaluator:

    def test_single_question_metrics(self):
        ev = RAGEvaluator(enable_phoenix=False)
        r = ev.evaluate_question(
            EVAL_QS[0]["question"], EVAL_QS[0]["golden_context"],
            _retriever, _answer, top_k=5)
        assert r.hit is True
        assert 0.0 <= r.faithfulness <= 1.0
        assert r.retrieval_ms >= 0
        assert r.generation_ms >= 0

    def test_batch_report(self):
        ev = RAGEvaluator(enable_phoenix=False)
        report = ev.evaluate(EVAL_QS, _retriever, _answer, top_k=5)
        assert report.total == 2
        assert report.hit_rate == 1.0
        assert 0.0 <= report.avg_faithfulness <= 1.0

    def test_phoenix_unavailable_no_crash(self):
        # 不启动 Phoenix 服务，直接初始化应降级而非抛异常
        ev = RAGEvaluator(enable_phoenix=True,
                          phoenix_endpoint="http://localhost:6999/v1/traces")
        r = ev.evaluate_question(
            EVAL_QS[0]["question"], EVAL_QS[0]["golden_context"],
            _retriever, _answer, top_k=5)
        assert r.hit is True

    def test_report_export(self, tmp_path):
        ev = RAGEvaluator(enable_phoenix=False)
        report = ev.evaluate(EVAL_QS, _retriever, _answer, top_k=5)
        out = tmp_path / "report.json"
        ev.export_report(report, str(out))
        data = json.loads(out.read_text())
        assert data["total"] == 2
        assert "hit_rate" in data and "avg_faithfulness" in data
