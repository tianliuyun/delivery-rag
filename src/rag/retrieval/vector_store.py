"""
向量检索（FAISS）
支持 BGE 等向量模型
"""
from typing import List, Tuple, Optional
import numpy as np


class VectorStore:
    """向量存储与检索

    生产级推荐：
    - 小规模（<100万）: FAISS 足够
    - 中大规模: Milvus / Qdrant / Weaviate
    """

    def __init__(self, dimension: int = 512, metric: str = "cosine"):
        self.dimension = dimension
        self.metric = metric
        self.vectors: List[np.ndarray] = []
        self.doc_ids: List[int] = []
        self.index = None
        self._model = None
        self._model_name = None

    def load_embedding_model(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        """加载向量模型

        优先使用 sentence-transformers，不可用时用 mock 模式
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
            self.dimension = self._model.get_sentence_embedding_dimension()
            print(f"✅ 向量模型加载成功: {model_name} (dim={self.dimension})")
            return True
        except ImportError:
            print(f"⚠️  sentence-transformers 未安装，使用 mock 向量模式")
            self._model = None
            return False

    def encode(self, texts: List[str]) -> np.ndarray:
        """文本向量化"""
        if self._model is not None:
            # 加 "为这个句子生成表示以用于检索相关文章：" 前缀是 BGE 的最佳实践
            return self._model.encode(texts, normalize_embeddings=True)
        else:
            # Mock 模式：用简单的哈希生成伪向量（仅用于演示/测试）
            import hashlib
            vectors = []
            for text in texts:
                # 用 md5 哈希生成伪随机但稳定的向量
                h = hashlib.md5(text.encode()).digest()
                vec = np.zeros(self.dimension, dtype=np.float32)
                for i in range(min(self.dimension, len(h) * 8)):
                    byte_idx = i // 8
                    bit_idx = i % 8
                    if byte_idx < len(h) and (h[byte_idx] >> bit_idx) & 1:
                        vec[i] = 1.0
                # 归一化
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
            return np.array(vectors, dtype=np.float32)

    def add_documents(self, texts: List[str], doc_ids: Optional[List[int]] = None):
        """添加文档到向量库"""
        if doc_ids is None:
            doc_ids = list(range(len(self.vectors), len(self.vectors) + len(texts)))

        embeddings = self.encode(texts)
        for i, emb in enumerate(embeddings):
            self.vectors.append(emb)
            self.doc_ids.append(doc_ids[i])

        # 重建 FAISS 索引
        self._build_index()

    def _build_index(self):
        """构建 FAISS 索引"""
        if not self.vectors:
            return

        try:
            import faiss
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product = 余弦相似度（已归一化）
            else:
                self.index = faiss.IndexFlatL2(self.dimension)

            vectors_np = np.array(self.vectors).astype('float32')
            self.index.add(vectors_np)
        except ImportError:
            # 没有 FAISS，用 numpy 暴力搜索
            self.index = None
            self._vectors_np = np.array(self.vectors).astype('float32')

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """搜索，返回 (doc_id, score) 列表"""
        if not self.vectors:
            return []

        query_vec = self.encode([query])[0]

        if self.index is not None:
            # FAISS 搜索
            import faiss
            scores, indices = self.index.search(
                np.array([query_vec]).astype('float32'), top_k
            )
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx < len(self.doc_ids):
                    results.append((self.doc_ids[idx], float(scores[0][i])))
            return results
        else:
            # NumPy 暴力搜索
            vectors_np = self._vectors_np
            if self.metric == "cosine":
                scores = np.dot(vectors_np, query_vec)
            else:
                scores = -np.linalg.norm(vectors_np - query_vec, axis=1)

            # 取 top_k
            top_indices = np.argsort(scores)[::-1][:top_k]
            return [(self.doc_ids[i], float(scores[i])) for i in top_indices]


class MilvusVectorStore(VectorStore):
    """生产级向量检索（Milvus）—— 面向中大规模企业知识库

    与 FAISS 后端接口完全一致（load_embedding_model / add_documents / search），
    通过 ``config.vector_store`` 切换：
      - faiss（默认）：轻量、进程内，适合 <100 万向量
      - milvus：企业级分布式向量库，适合中大规模 + 多实例共享

    实现要点：
      - 使用 pymilvus MilvusClient，连接即建集合（COSINE 度量）
      - insert 后显式 flush，保证写入可见
      - search 返回 (doc_id, score)，与 FAISS 后端签名一致
    """

    def __init__(self, dimension: int = 512, metric: str = "cosine",
                 uri: str = "http://localhost:19530",
                 collection_name: str = "delivery_rag"):
        super().__init__(dimension=dimension, metric=metric)
        self.uri = uri
        self.collection_name = collection_name
        self._client = None

    def _connect(self):
        """连接 Milvus 并确保集合存在。"""
        from pymilvus import MilvusClient

        self._client = MilvusClient(uri=self.uri)
        if not self._client.has_collection(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                metric_type="COSINE",
            )
        self._client.load_collection(self.collection_name)

    def add_documents(self, texts: List[str], doc_ids: Optional[List[int]] = None):
        """写入向量到 Milvus。"""
        if doc_ids is None:
            doc_ids = list(range(len(texts)))
        if not texts:
            return

        if self._client is None:
            self._connect()

        embeddings = self.encode(texts)
        rows = [
            {"id": int(doc_ids[i]), "vector": emb.tolist(), "text": texts[i]}
            for i, emb in enumerate(embeddings)
        ]
        self._client.insert(self.collection_name, rows)
        self._client.flush(self.collection_name)
        print(f"✅ Milvus 写入 {len(rows)} 条向量 → {self.collection_name}")

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Milvus 向量检索，返回 (doc_id, score)。"""
        if self._client is None:
            self._connect()

        query_vec = self.encode([query])[0]
        res = self._client.search(
            collection_name=self.collection_name,
            data=[query_vec.tolist()],
            limit=top_k,
            output_fields=["id"],
        )
        results = []
        for hit in res[0]:
            results.append((int(hit["id"]), float(hit["distance"])))
        return results

    def stats(self) -> dict:
        """集合统计（向量条数）。"""
        if self._client is None:
            self._connect()
        return {"backend": "milvus", "collection": self.collection_name,
                "num_entities": self._client.get_collection_stats(self.collection_name).get("row_count", 0)}


class ElasticsearchVectorStore(VectorStore):
    """生产级向量检索（Elasticsearch）—— 面向企业级私有化部署

    与 FAISS / Milvus 后端接口完全一致（load_embedding_model / add_documents / search），
    通过 ``config.vector_store`` 切换：
      - faiss（默认）：轻量、进程内，适合 <100 万向量
      - milvus：企业级分布式向量库，适合中大规模 + 多实例共享
      - elasticsearch：搜索引擎 + 向量检索一体，适合已有 ES 基础设施的企业（交付场景常见）

    实现要点：
      - 使用 elasticsearch-py 原生客户端（HTTP API，docker compose 即可拉起单节点）
      - dense_vector 字段存向量（cosine 相似度），text 字段存原文
      - knn 检索返回 (doc_id, score)，与 FAISS / Milvus 后端签名一致
    """

    def __init__(self, dimension: int = 512, metric: str = "cosine",
                 uri: str = "http://localhost:9200",
                 index_name: str = "delivery_rag"):
        super().__init__(dimension=dimension, metric=metric)
        self.uri = uri
        self.index_name = index_name
        self._client = None

    def _connect(self):
        """连接 ES 并确保索引存在（含 dense_vector 映射）。"""
        from elasticsearch import Elasticsearch

        self._client = Elasticsearch(self.uri, request_timeout=10)
        if not self._client.indices.exists(index=self.index_name):
            self._client.indices.create(
                index=self.index_name,
                mappings={
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": self.dimension,
                            "index": True,
                            "similarity": "cosine",
                        },
                        "text": {"type": "text"},
                        "doc_id": {"type": "long"},
                    }
                },
            )
        print(f"✅ Elasticsearch 连接成功: {self.uri} (index={self.index_name})")

    def add_documents(self, texts: List[str], doc_ids: Optional[List[int]] = None):
        """写入向量到 ES。"""
        if doc_ids is None:
            doc_ids = list(range(len(texts)))
        if not texts:
            return

        if self._client is None:
            self._connect()

        embeddings = self.encode(texts)
        for i, emb in enumerate(embeddings):
            self._client.index(
                index=self.index_name,
                document={
                    "doc_id": int(doc_ids[i]),
                    "vector": emb.tolist(),
                    "text": texts[i],
                },
            )
        self._client.indices.refresh(index=self.index_name)
        print(f"✅ Elasticsearch 写入 {len(texts)} 条向量 → {self.index_name}")

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """ES 向量检索（knn），返回 (doc_id, score)。"""
        if self._client is None:
            self._connect()

        query_vec = self.encode([query])[0]
        res = self._client.search(
            index=self.index_name,
            knn={
                "field": "vector",
                "query_vector": query_vec.tolist(),
                "k": top_k,
                "num_candidates": max(top_k * 4, 100),
            },
            source=["doc_id", "text"],
        )
        results = []
        for hit in res["hits"]["hits"]:
            doc_id = hit["_source"].get("doc_id", -1)
            score = hit["_score"] if hit["_score"] is not None else 0.0
            results.append((int(doc_id), float(score)))
        return results

    def stats(self) -> dict:
        """索引统计（文档条数）。"""
        if self._client is None:
            self._connect()
        count = self._client.count(index=self.index_name)["count"]
        return {"backend": "elasticsearch", "index": self.index_name,
                "num_entities": count}

