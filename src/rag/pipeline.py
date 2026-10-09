"""
完整的 RAG Pipeline
把文档处理、检索、重排、生成串起来

架构：
文档 → 分块 → 向量化 + BM25索引 → 混合检索 → 重排 → LLM生成 → 答案（带引用）
"""
from typing import List, Optional
from pathlib import Path
import json

from .config import RAGConfig
from .document.loader import DocumentLoader, Document
from .document.chunker import TextChunker, Chunk
from .retrieval.vector_store import VectorStore, MilvusVectorStore, ElasticsearchVectorStore
from .retrieval.bm25 import BM25Retriever
from .retrieval.hybrid import HybridRetriever
from .retrieval.reranker import Reranker
from .generation.llm import LLMClient
from .generation.answer import AnswerGenerator, RAGAnswer


class RAGPipeline:
    """完整的 RAG 流水线

    使用示例：
    ```python
    rag = RAGPipeline()
    rag.build_index("data/docs/")
    answer = rag.query("桌面云接入慢怎么排查？")
    print(answer.answer)
    ```
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig.default()

        # 文档处理
        self.chunker = TextChunker(
            strategy=self.config.chunk_strategy,
            child_size=self.config.child_chunk_size,
            parent_size=self.config.parent_chunk_size,
            overlap=self.config.chunk_overlap,
        )

        # 检索层
        if self.config.vector_store == "milvus":
            self.vector_store = MilvusVectorStore(
                dimension=self.config.embedding_dim,
                uri=self.config.milvus_uri,
                collection_name=self.config.milvus_collection,
            )
        elif self.config.vector_store == "elasticsearch":
            self.vector_store = ElasticsearchVectorStore(
                dimension=self.config.embedding_dim,
                uri=self.config.es_uri,
                index_name=self.config.es_index,
            )
        else:
            self.vector_store = VectorStore(dimension=self.config.embedding_dim)
        self.bm25 = BM25Retriever()
        self.hybrid = HybridRetriever(
            self.vector_store,
            self.bm25,
            rrf_k=self.config.rrf_k,
        )
        self.reranker = Reranker(model_name=self.config.reranker_model)

        # 生成层
        self.llm = LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key,
            temperature=self.config.llm_temperature,
        )
        self.answer_gen = AnswerGenerator(
            self.llm,
            enable_citations=self.config.enable_citations,
            enable_refusal=self.config.enable_refusal,
            confidence_threshold=self.config.confidence_threshold,
        )

        # 状态
        self._built = False
        self._chunks: List[Chunk] = []
        self._parent_texts: List[str] = []  # 父块文本（用于生成）
        self._chunk_texts: List[str] = []  # 子块文本（用于检索）

    def build_index(self, docs_path: str | Path, model_name: Optional[str] = None):
        """构建知识库索引

        Args:
            docs_path: 文档目录或文件
            model_name: 向量模型名称
        """
        print("🔨 正在构建知识库...")

        # 1. 加载文档
        print("  📄 加载文档...")
        documents = DocumentLoader.load(docs_path)
        print(f"    加载了 {len(documents)} 个文档")

        # 2. 加载向量模型
        print("  🧠 加载向量模型...")
        if model_name:
            self.config.embedding_model = model_name
        self.vector_store.load_embedding_model(self.config.embedding_model)

        # 3. 分块
        print("  ✂️  语义分块 (策略: {})...".format(self.config.chunk_strategy))
        all_chunks = []
        all_parent_texts = []
        parent_offset = 0  # 父块偏移量，跨文档累计
        for doc in documents:
            chunks = self.chunker.chunk(doc.content)
            # 给每个 chunk 加上源信息，并修正父块索引（跨文档）
            doc_parents = getattr(self.chunker, '_parent_chunks', [])
            for chunk in chunks:
                chunk.metadata['source'] = doc.source
                chunk.metadata['filename'] = doc.metadata.get('filename', '')
                if chunk.parent_index >= 0:
                    chunk.parent_index += parent_offset
            all_chunks.extend(chunks)
            all_parent_texts.extend([p.text for p in doc_parents])
            parent_offset += len(doc_parents)

        self._chunks = all_chunks
        self._chunk_texts = [c.text for c in all_chunks]
        self._parent_texts = all_parent_texts
        print(f"    生成了 {len(all_chunks)} 个文本块, {len(all_parent_texts)} 个父块")

        # 4. 构建向量索引
        print("  📊 构建向量索引...")
        self.vector_store.add_documents(self._chunk_texts)

        # 5. 构建 BM25 索引
        print("  🔍 构建 BM25 索引...")
        self.bm25.index(self._chunk_texts)

        # 6. 加载重排模型（可选）
        if self.config.use_reranker:
            print("  🎯 加载重排模型...")
            self.reranker.load()

        # 7. 完成
        self._built = True
        print(f"✅ 知识库构建完成！")
        return self

    def query(self, question: str,
              use_hybrid: bool = True,
              use_rerank: bool = True) -> RAGAnswer:
        """查询

        Args:
            question: 用户问题
            use_hybrid: 是否使用混合检索
            use_rerank: 是否使用重排

        Returns:
            RAGAnswer 回答对象
        """
        if not self._built:
            raise RuntimeError("请先调用 build_index() 构建知识库")

        # 1. 混合检索
        if use_hybrid:
            results = self.hybrid.search(question, top_k=self.config.hybrid_top_k)
        else:
            results = self.vector_store.search(question, top_k=self.config.hybrid_top_k)
            results = [(doc_id, score, "vector") for doc_id, score in results]

        # 2. 取出对应的文本（父块优先，如果有的话）
        context_texts = []
        for doc_id, score, source in results:
            doc_id_int = int(doc_id)
            if self._chunks and doc_id_int < len(self._chunks):
                chunk = self._chunks[doc_id_int]
                # 如果有父块，用父块（上下文更完整）
                if chunk.parent_index >= 0 and chunk.parent_index < len(self._parent_texts):
                    parent_text = self._parent_texts[chunk.parent_index]
                    context_texts.append((parent_text, score))
                else:
                    context_texts.append((chunk.text, score))

        # 3. 重排
        if use_rerank and self.config.use_reranker and len(context_texts) > 1:
            texts = [t for t, _ in context_texts[:20]]  # 重排前 20 个
            reranked = self.reranker.rerank(question, texts, top_n=self.config.reranker_top_n)
            context_texts = [(texts[idx], score) for idx, score in reranked]

        # 4. 生成答案
        answer = self.answer_gen.generate(question, context_texts)

        return answer

    def save(self, path: str | Path):
        """保存知识库到本地"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # 简单实现：保存 chunk 和索引
        # 生产环境推荐用 FAISS 原生保存 + JSON 元数据
        import pickle
        data = {
            'chunks': [(c.text, c.index, c.parent_index, c.metadata) for c in self._chunks],
            'config': {
                'embedding_model': self.config.embedding_model,
                'chunk_strategy': self.config.chunk_strategy,
            }
        }
        with open(path / 'index.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 知识库已保存到 {path}")

    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        return {
            'built': self._built,
            'chunk_count': len(self._chunks),
            'embedding_model': self.config.embedding_model,
            'chunk_strategy': self.config.chunk_strategy,
            'reranker': self.config.reranker_model if self.config.use_reranker else None,
        }
