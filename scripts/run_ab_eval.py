"""A/B 实测：固定分块+纯向量 vs 语义分块+混合检索+重排（真实 BGE 向量 + 真实 CrossEncoder）。

输出 results/ab_eval_report.json：
- 每个方案：Hit Rate@1/3/5、MRR、平均检索耗时
- 逐题明细：question / hit / golden_context / retrieved[0:3]

用法：
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="" python scripts/run_ab_eval.py --docs data/sample_docs --testset data/qa_eval_set.jsonl --out results/ab_eval_report.json
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import RAGConfig
from rag.document.chunker import TextChunker
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import Reranker
from rag.document.loader import DocumentLoader


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def hit_in_chunks(golden: str, chunks: list) -> bool:
    """golden_context（原文子串）是否出现在返回的 chunks 中（归一化后匹配）。"""
    if not golden:
        return False
    ng = norm(golden)
    if len(ng) < 4:
        return False
    for c in chunks:
        if ng in norm(c):
            return True
    return False


def build_baseline(docs_path: Path, embedding_path: str):
    """基线 A：固定分块 + 纯向量检索（无 BM25、无重排）。"""
    documents = DocumentLoader.load(str(docs_path))
    chunker = TextChunker(strategy="fixed", child_size=512)
    chunks = []
    for doc in documents:
        for c in chunker.chunk(doc.content):
            c.metadata["source"] = doc.source
            chunks.append(c)
    texts = [c.text for c in chunks]

    vs = VectorStore(dimension=512)
    ok = vs.load_embedding_model(embedding_path)
    if not ok:
        raise RuntimeError("向量模型加载失败")
    vs.add_documents(texts)
    return chunks, vs


def build_production(docs_path: Path, embedding_path: str, reranker_path: str, load_reranker: bool = True):
    """方案 B：语义父子块 + 混合检索（向量+BM25+RRF）+ CrossEncoder 重排。"""
    documents = DocumentLoader.load(str(docs_path))
    chunker = TextChunker(strategy="semantic_parent_child", child_size=256, parent_size=1024, overlap=50)
    chunks = []
    parents = []
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
    reranker = None
    if load_reranker:
        reranker = Reranker(model_name=reranker_path)
        reranker.load()
    return chunks, parents, hybrid, reranker


def eval_retrieval(questions, retriever_fn, top_ks=(1, 3, 5)):
    """通用检索评测：返回 hit 明细 + 指标。"""
    detail = []
    hits = {k: 0 for k in top_ks}
    rr_sum = 0.0
    t0 = time.time()
    for idx, q in enumerate(questions):
        if idx % 10 == 0:
            print(f"  ⏳ 检索评测进度 {idx}/{len(questions)}")
        q_text = q["question"]
        golden = q.get("golden_context", "")
        results = retriever_fn(q_text)
        retrieved = [t for t, _ in results[: max(top_ks)]]
        # 命中判定：golden 出现在第 i 位（i 从 0）
        ng = norm(golden)
        hit_pos = -1
        for i, t in enumerate(retrieved):
            if ng and len(ng) >= 4 and ng in norm(t):
                hit_pos = i
                break
        if hit_pos >= 0:
            rr_sum += 1.0 / (hit_pos + 1)
        for k in top_ks:
            if hit_pos >= 0 and hit_pos < k:
                hits[k] += 1
        detail.append({
            "question": q_text,
            "hit_position": hit_pos,
            "golden_context": golden[:100],
            "retrieved_top": [t[:80] for t in retrieved[:3]],
        })
    elapsed = time.time() - t0
    n = len(questions)
    metrics = {"total": n, "avg_ms": elapsed / n * 1000 if n else 0}
    for k in top_ks:
        metrics[f"hit_rate@{k}"] = round(hits[k] / n, 4) if n else 0
    metrics["mrr"] = round(rr_sum / n, 4) if n else 0
    return metrics, detail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/sample_docs")
    parser.add_argument("--testset", default="data/qa_eval_set.jsonl")
    parser.add_argument("--out", default="results/ab_eval_report.json")
    parser.add_argument("--embedding", default="models/bge-small-zh-v1.5")
    parser.add_argument("--reranker", default="models/bge-reranker-base")
    parser.add_argument("--no-rerank", action="store_true", help="方案B不启用重排（快速跑检索对比）")
    args = parser.parse_args()

    docs_path = Path(args.docs)
    questions = [json.loads(l) for l in open(args.testset, encoding="utf-8") if l.strip()]
    print(f"📋 评测集: {len(questions)} 条")

    # 基线 A
    print("\n🔨 构建基线 A（固定分块 + 纯向量）...")
    chunks_a, vs_a = build_baseline(docs_path, args.embedding)
    print(f"  chunks={len(chunks_a)}")
    metrics_a, detail_a = eval_retrieval(
        questions,
        lambda q: [(chunks_a[i].text, s) for i, s in vs_a.search(q, top_k=10)],
    )
    print(f"  基线A: {metrics_a}")

    # 方案 B
    print("\n🔨 构建方案 B（语义父子块 + 混合检索 + 重排）...")
    chunks_b, parents_b, hybrid_b, reranker_b = build_production(
        docs_path, args.embedding, args.reranker, load_reranker=not args.no_rerank)
    print(f"  child_chunks={len(chunks_b)}, parent_chunks={len(parents_b)}")

    def prod_retriever(q):
        results = hybrid_b.search(q, top_k=30)
        texts = []
        for doc_id, score, src in results:
            idx = int(doc_id)
            if idx < len(chunks_b):
                c = chunks_b[idx]
                if c.parent_index >= 0 and c.parent_index < len(parents_b):
                    texts.append(parents_b[c.parent_index])
                else:
                    texts.append(c.text)
        if args.no_rerank:
            return [(t, 0.0) for t in texts[:5]]
        if len(texts) > 1:
            reranked = reranker_b.rerank(q, texts[:10], top_n=5)
            return [(texts[i], s) for i, s in reranked]
        return [(t, 0.0) for t in texts[:5]]

    metrics_b, detail_b = eval_retrieval(questions, prod_retriever)
    print(f"  方案B: {metrics_b}")

    report = {
        "testset": str(args.testset),
        "docs": str(args.docs),
        "embedding": args.embedding,
        "reranker": args.reranker,
        "baseline": {"name": "fixed_chunk + pure_vector", "metrics": metrics_a, "detail": detail_a},
        "production": {"name": "semantic_parent_child + hybrid + rerank", "metrics": metrics_b, "detail": detail_b},
        "delta": {
            "hit_rate@1": round(metrics_b["hit_rate@1"] - metrics_a["hit_rate@1"], 4),
            "hit_rate@5": round(metrics_b["hit_rate@5"] - metrics_a["hit_rate@5"], 4),
            "mrr": round(metrics_b["mrr"] - metrics_a["mrr"], 4),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ A/B 评测完成 -> {out}")
    print(f"Delta: {report['delta']}")


if __name__ == "__main__":
    main()
