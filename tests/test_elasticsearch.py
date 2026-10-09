"""Elasticsearch 生产级向量检索后端集成测试。

依赖：本地 ES 单节点（docker compose -f deploy/elasticsearch/docker-compose.yml up -d）。
ES 不可用时自动 skip（不阻塞普通单测）。

覆盖：
- ESVectorStore add/search 基本链路（与 FAISS/Milvus 接口一致）
- 完整 pipeline build → query（vector_store=elasticsearch 切换）
"""
import pytest

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.retrieval.vector_store import ElasticsearchVectorStore

ES_URI = "http://localhost:9200"
TEST_INDEX = "delivery_rag_pytest_es"


def _es_available() -> bool:
    try:
        from elasticsearch import Elasticsearch
        c = Elasticsearch(ES_URI, request_timeout=5)
        return c.ping()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _es_available(),
    reason="Elasticsearch 未运行（deploy/elasticsearch/docker-compose.yml）",
)


@pytest.fixture()
def clean_index():
    from elasticsearch import Elasticsearch
    c = Elasticsearch(ES_URI, request_timeout=10)
    if c.indices.exists(index=TEST_INDEX):
        c.indices.delete(index=TEST_INDEX)
    yield
    if c.indices.exists(index=TEST_INDEX):
        c.indices.delete(index=TEST_INDEX)


class TestElasticsearchVectorStore:

    def test_add_and_search(self, clean_index):
        store = ElasticsearchVectorStore(
            dimension=512, uri=ES_URI, index_name=TEST_INDEX
        )
        store.add_documents(["桌面云登录超时排查", "存储RAID降级处理", "VPN证书失效修复"])
        results = store.search("桌面云登录问题", top_k=3)
        assert len(results) == 3
        assert all(isinstance(doc_id, int) and isinstance(score, float) for doc_id, score in results)

    def test_pipeline_build_and_query_with_es(self, clean_index):
        """ES 后端跑完整 pipeline：build → query。"""
        config = RAGConfig(
            vector_store="elasticsearch",
            es_uri=ES_URI,
            es_index=TEST_INDEX,
        )
        rag = RAGPipeline(config)
        rag.build_index("data/sample_docs")
        assert rag.vector_store.stats()["num_entities"] >= 30  # sample_docs 分块数

        ans = rag.query("EDS分布式存储故障怎么排查？")
        assert ans.has_answer is True
        assert len(ans.contexts) > 0
