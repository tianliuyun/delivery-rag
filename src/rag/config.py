"""
配置管理
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RAGConfig:
    """RAG 系统配置"""

    # 向量模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512

    # 重排模型配置
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 5
    use_reranker: bool = True

    # 分块配置
    chunk_strategy: str = "semantic_parent_child"  # fixed / semantic / parent_child
    child_chunk_size: int = 256
    parent_chunk_size: int = 1024
    chunk_overlap: int = 50

    # 检索配置
    vector_store: str = "faiss"  # faiss（轻量，默认） / milvus（企业级向量库） / elasticsearch（搜索引擎+向量一体）
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "delivery_rag"
    es_uri: str = "http://localhost:9200"
    es_index: str = "delivery_rag"
    hybrid_top_k: int = 50
    vector_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60  # RRF 融合参数

    # LLM 配置
    llm_provider: str = "mock"  # mock / openai / ollama / qwen
    llm_model: str = "mock"
    llm_base_url: Optional[str] = None
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.1

    # 幻觉控制
    enable_citations: bool = True
    enable_refusal: bool = True
    confidence_threshold: float = 0.3

    # 数据路径
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    @classmethod
    def default(cls) -> "RAGConfig":
        return cls()
