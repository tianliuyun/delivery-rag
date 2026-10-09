"""
检索模块测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.vector_store import VectorStore
from rag.retrieval.hybrid import HybridRetriever


def test_bm25_index_and_search():
    """测试 BM25 检索"""
    docs = [
        "桌面云接入速度优化指南",
        "分布式存储故障排查手册",
        "GPU虚拟化部署指南",
        "网络安全配置教程",
        "云桌面性能调优最佳实践",
    ]
    bm25 = BM25Retriever()
    bm25.index(docs)
    assert bm25._initialized

    results = bm25.search("桌面云", top_k=3)
    assert len(results) > 0
    assert len(results) <= 3
    # 第一个结果应该最相关
    assert results[0][1] > 0  # 分数大于 0


def test_bm25_empty_query():
    """空查询测试"""
    bm25 = BM25Retriever()
    bm25.index(["test doc"])
    results = bm25.search("", top_k=3)
    assert results == []


def test_vector_store_mock():
    """测试向量库（mock 模式）"""
    vs = VectorStore(dimension=64)
    # 不加载真实模型，用 mock 模式
    docs = ["文档一的内容", "文档二的内容", "文档三的内容"]
    vs.add_documents(docs)
    assert len(vs.vectors) == 3

    results = vs.search("查询内容", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r[0], int) for r in results)
    assert all(isinstance(r[1], float) for r in results)


def test_hybrid_retrieval():
    """测试混合检索 + RRF 融合"""
    vs = VectorStore(dimension=64)
    bm25 = BM25Retriever()

    docs = [
        "桌面云接入速度优化",
        "分布式存储磁盘故障排查",
        "GPU虚拟化云桌面部署",
        "网络安全配置指南",
        "云桌面性能调优实践",
    ]

    vs.add_documents(docs)
    bm25.index(docs)

    hybrid = HybridRetriever(vs, bm25, rrf_k=60)
    results = hybrid.search("桌面云", top_k=5)

    assert len(results) > 0
    assert len(results) <= 5
    # 结果应该有 doc_id, score, source
    for doc_id, score, source in results:
        assert isinstance(doc_id, int)
        assert isinstance(score, float)
        assert source in ("vector", "bm25", "vector+bm25")


def test_hybrid_vector_only():
    """只用向量检索"""
    vs = VectorStore(dimension=64)
    bm25 = BM25Retriever()
    vs.add_documents(["doc1", "doc2"])
    bm25.index(["doc1", "doc2"])

    hybrid = HybridRetriever(vs, bm25)
    results = hybrid.search("test", top_k=3, use_vector=True, use_bm25=False)
    assert len(results) > 0
    assert all(s == "vector" for _, _, s in results)


def test_hybrid_bm25_only():
    """只用 BM25 检索"""
    vs = VectorStore(dimension=64)
    bm25 = BM25Retriever()
    vs.add_documents(["doc1", "doc2"])
    bm25.index(["doc1", "doc2"])

    hybrid = HybridRetriever(vs, bm25)
    results = hybrid.search("test", top_k=3, use_vector=False, use_bm25=True)
    assert len(results) > 0
    assert all(s == "bm25" for _, _, s in results)
