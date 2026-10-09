"""完整 RAG 评测（真实链路）：真实 BGE 向量 + BM25 + RRF + CrossEncoder 重排 + DeepSeek 生成 + DeepSeek judge。

与 A/B 检索评测互补：本脚本评估端到端答案质量（Faithfulness / 拒答率 / 引用率）。
输出 results/rag_eval_deepseek_report.json。

用法：
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python scripts/run_rag_eval_deepseek.py --docs data/sample_docs --testset data/qa_eval_set.jsonl --out results/rag_eval_deepseek_report.json
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import RAGConfig
from rag.document.loader import DocumentLoader
from rag.document.chunker import TextChunker
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.evaluation.evaluator import RAGEvaluator

ENV_PATH = Path.home() / ".hermes/workspace/tasks/llm-course-interview/01-基础学习/week15graph和llm/.env"


def load_key():
    key = base = None
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("DEEPSEEK_BASE_URL"):
                base = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not found")
    return key, base or "https://api.deepseek.com"


API_KEY, BASE_URL = load_key()


def call_deepseek(messages, max_tokens=400, temperature=0.2, retries=3):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == retries - 1:
                print(f"  ⚠️ LLM 调用失败: {e}")
                return ""
            time.sleep(3 * (attempt + 1))


def build_pipeline(docs_path, embedding_path, reranker_path):
    documents = DocumentLoader.load(str(docs_path))
    chunker = TextChunker(strategy="semantic_parent_child", child_size=256, parent_size=1024, overlap=50)
    chunks, parents = [], []
    parent_offset = 0
    for doc in documents:
        cs = chunker.chunk(doc.content)
        doc_parents = getattr(chunker, "_parent_chunks", [])
        for c in cs:
            c.metadata["source"] = doc.source
            if c.parent_index >= 0:
                c.parent_index += parent_offset
        chunks.extend(cs)
        parents.extend([p.text for p in doc_parents])
        parent_offset += len(doc_parents)
    texts = [c.text for c in chunks]
    vs = VectorStore(dimension=512)
    vs.load_embedding_model(embedding_path)
    vs.add_documents(texts)
    bm25 = BM25Retriever()
    bm25.index(texts)
    hybrid = HybridRetriever(vs, bm25, rrf_k=60)
    reranker = Reranker(model_name=reranker_path)
    reranker.load()
    return chunks, parents, hybrid, reranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--testset", default="data/qa_eval_set.jsonl")
    parser.add_argument("--out", default="results/rag_eval_deepseek_report.json")
    parser.add_argument("--embedding", default="models/bge-small-zh-v1.5")
    parser.add_argument("--reranker", default="models/bge-reranker-base")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sample", type=int, default=0, help="抽样 N 题（0=全量）")
    args = parser.parse_args()

    questions = [json.loads(l) for l in open(args.testset, encoding="utf-8") if l.strip()]
    if args.sample:
        questions = questions[: args.sample]
    print(f"📋 评测集: {len(questions)} 题（top_k={args.top_k}）")

    print("🔨 构建 RAG pipeline（真实向量 + 混合检索 + 重排）...")
    chunks, parents, hybrid, reranker = build_pipeline(args.docs, args.embedding, args.reranker)
    print(f"  child_chunks={len(chunks)}, parent_chunks={len(parents)}")

    def retriever_fn(q):
        results = hybrid.search(q, top_k=30)
        texts = []
        for doc_id, score, src in results:
            idx = int(doc_id)
            if idx < len(chunks):
                c = chunks[idx]
                if c.parent_index >= 0 and c.parent_index < len(parents):
                    texts.append(parents[c.parent_index])
                else:
                    texts.append(c.text)
        if len(texts) > 1:
            reranked = reranker.rerank(q, texts[:20], top_n=args.top_k)
            return [texts[i] for i, _ in reranked]
        return texts[: args.top_k]

    def answer_fn(q, contexts):
        if not contexts:
            return "根据现有资料无法回答。"
        ctx = "\n\n".join(contexts[: args.top_k])[:2500]
        prompt = (
            "仅根据以下知识库上下文回答问题。若上下文不含答案，回复「根据现有资料无法回答」。\n"
            f"上下文：\n{ctx}\n\n问题：{q}\n答案："
        )
        ans = call_deepseek([{"role": "user", "content": prompt}], max_tokens=300)
        return ans or "根据现有资料无法回答。"

    def judge_fn(q, context, answer):
        if not context or not answer or "无法回答" in answer:
            return 0.0 if context else 1.0
        ctx = context[:2000]
        prompt = (
            "你是忠实度评测器。判断答案中的每个陈述是否都能由给定上下文推断支持。\n"
            "只输出一个 0 到 1 之间的数字（1=完全支持，0=完全虚构）。\n\n"
            f"上下文：\n{ctx}\n\n答案：{answer}\n\n忠实度："
        )
        raw = call_deepseek([{"role": "user", "content": prompt}], max_tokens=20)
        # 优先匹配完整小数或 1.0 / 0.0，避免把 "10" / 解释文字里的 0 误当分数
        m = re.search(r"(?:1\.0|0\.\d+|1|0)(?:\s*分)?", raw)
        try:
            v = float(m.group(0)) if m else 0.0
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    evaluator = RAGEvaluator(judge_fn=judge_fn, enable_phoenix=True)
    print("🔍 运行端到端评测（DeepSeek 生成 + DeepSeek judge）...")
    report = evaluator.evaluate(questions, retriever_fn, answer_fn, top_k=args.top_k)

    # 增强报告：样例
    samples = []
    for r in report.results[:5]:
        samples.append({"question": r.question, "hit": r.hit, "faithfulness": r.faithfulness})

    out_dict = report.to_dict()
    out_dict["config"] = {
        "embedding": args.embedding,
        "reranker": args.reranker,
        "hybrid": "vector+bm25+rrf",
        "top_k": args.top_k,
        "llm": "deepseek-chat (answer+judge)",
    }
    out_dict["samples"] = samples
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 50)
    print(f"总题数: {report.total}")
    print(f"Hit Rate@{args.top_k}: {report.hit_rate:.2%}")
    print(f"Faithfulness: {report.avg_faithfulness:.3f}")
    print(f"平均检索: {report.avg_retrieval_ms:.1f}ms / 生成: {report.avg_generation_ms:.1f}ms")
    print(f"报告已保存: {out}")


if __name__ == "__main__":
    main()
