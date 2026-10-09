"""运行 RAG 评测（含 Phoenix 观测），输出评估报告。

用法：
    python scripts/run_evaluation.py --docs data/sample_docs --out /tmp/rag_eval_report.json

说明：
- 使用 sample_docs 的 3 篇文档建索引
- 评测问题从文档内容构造（golden_context 取原文片段），共 8 题
- 检索用 pipeline 的 hybrid 检索，生成用 mock LLM（本机可跑）
- 忠实度用内置近似 judge；如需更准可接 DeepSeek（--judge deepseek）
- 全程 OpenTelemetry → Phoenix（默认 localhost:6006），无 Phoenix 时降级无观测
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.evaluation import RAGEvaluator

# 从 sample_docs 构造的评测问题（golden_context = 文档原文片段，用于命中判定）
EVAL_QUESTIONS = [
    {"question": "桌面云登录慢，第一步应该检查什么？",
     "golden_context": "检查客户端到服务器的网络延迟（ping 测试）"},
    {"question": "桌面云每用户建议带宽至少多少？",
     "golden_context": "确认带宽是否充足（建议每用户至少 2Mbps）"},
    {"question": "EDS 磁盘故障排查第一步看什么？",
     "golden_context": "查看磁盘 SMART 信息"},
    {"question": "EDS 网络故障排查第二步确认什么网卡？",
     "golden_context": "确认万兆网卡状态"},
    {"question": "GPU 虚拟化部署中，NVIDIA 授权服务器默认端口是多少？",
     "golden_context": "配置 License 服务端口（默认 7070）"},
    {"question": "GPU 虚拟化推荐哪些显卡型号？",
     "golden_context": "GPU：NVIDIA A10 / A16 / L20 等 vGPU 支持卡"},
    {"question": "GPU 虚拟化存储网络建议什么规格？",
     "golden_context": "存储网络：万兆以上"},
    {"question": "EDS 重要数据建议开启几副本？",
     "golden_context": "重要数据开启多副本（至少 3 副本）"},
]


def build_answer_fn(rag: RAGPipeline):
    def answer_fn(question: str, contexts: list):
        # 直接用第一个命中上下文生成简单回答（mock 生成，本机零依赖）
        if contexts:
            return f"根据资料：{contexts[0][:120]}"
        return "未检索到相关内容。"
    return answer_fn


def build_retriever_fn(rag: RAGPipeline):
    def retriever_fn(question: str):
        results = rag.hybrid.search(question, top_k=5)
        texts = []
        for doc_id, score, source in results:
            doc_id_int = int(doc_id)
            if doc_id_int < len(rag._chunks):
                texts.append(rag._chunks[doc_id_int].text)
        return texts
    return retriever_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--out", default="/tmp/rag_eval_report.json")
    parser.add_argument("--phoenix", action="store_true", default=True,
                        help="启用 Phoenix 观测（默认开，无 Phoenix 自动降级）")
    args = parser.parse_args()

    print("📦 构建 RAG pipeline（faiss 后端）...")
    config = RAGConfig(vector_store="faiss")
    rag = RAGPipeline(config)
    rag.build_index(args.docs)

    evaluator = RAGEvaluator(enable_phoenix=args.phoenix)
    print(f"🔍 评测 {len(EVAL_QUESTIONS)} 题（Phoenix: {evaluator._phoenix_ok}）...")
    report = evaluator.evaluate(
        EVAL_QUESTIONS,
        retriever_fn=build_retriever_fn(rag),
        answer_fn=build_answer_fn(rag),
        top_k=5,
    )
    evaluator.export_report(report, args.out)

    print("=" * 50)
    print("RAG 评估报告（RAGAS 风格 + Phoenix 观测）")
    print("=" * 50)
    print(f"总题数: {report.total}")
    print(f"检索命中率 (Hit Rate@5): {report.hit_rate:.1%}")
    print(f"平均忠实度 (Faithfulness): {report.avg_faithfulness:.2f}")
    print(f"平均检索耗时: {report.avg_retrieval_ms:.1f}ms")
    print(f"平均生成耗时: {report.avg_generation_ms:.1f}ms")
    print(f"报告已保存: {args.out}")


if __name__ == "__main__":
    main()
