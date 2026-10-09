"""Milvus 生产级向量库后端集成测试。

依赖：本地 Milvus standalone 服务（docker compose -f deploy/milvus/docker-compose.yml up -d）。
Milvus 不可用时自动 skip（不阻塞普通单测）。
"""
import pytest

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline
from rag.retrieval.vector_store import MilvusVectorStore

MILVUS_URI = "http://localhost:19530"
TEST_COLLECTION = "delivery_rag_pytest"


def _milvus_available() -> bool:
    try:
        from pymilvus import MilvusClient
        c = MilvusClient(uri=MILVUS_URI)
        c.list_collections()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _milvus_available(),
    reason="Milvus standalone 未运行（deploy/milvus/docker-compose.yml）",
)


@pytest.fixture()
def clean_collection():
    from pymilvus import MilvusClient
    c = MilvusClient(uri=MILVUS_URI)
    if c.has_collection(TEST_COLLECTION):
        c.drop_collection(TEST_COLLECTION)
    yield
    if c.has_collection(TEST_COLLECTION):
        c.drop_collection(TEST_COLLECTION)


class TestMilvusVectorStore:

    def test_add_and_search(self, clean_collection):
        store = MilvusVectorStore(
            dimension=512, uri=MILVUS_URI, collection_name=TEST_COLLECTION
        )
        store.add_documents(["桌面云登录超时排查", "存储RAID降级处理", "VPN证书失效修复"])
        results = store.search("桌面云登录问题", top_k=3)
        assert len(results) == 3
        assert all(isinstance(doc_id, int) and isinstance(score, float) for doc_id, score in results)

    def test_pipeline_build_and_query_with_milvus(self, clean_collection):
        """Milvus 后端跑完整 pipeline：build → query。"""
        config = RAGConfig(
            vector_store="milvus",
            milvus_uri=MILVUS_URI,
            milvus_collection=TEST_COLLECTION,
        )
        rag = RAGPipeline(config)
        rag.build_index("data/sample_docs")
        assert rag.vector_store.stats()["num_entities"] >= 30  # sample_docs 分块数

        ans = rag.query("EDS分布式存储故障怎么排查？")
        assert ans.has_answer is True
        assert len(ans.contexts) > 0
