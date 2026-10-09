"""
FastAPI 微服务入口 —— Delivery RAG 私有化部署

用法：
    # 启动（默认使用 mock 向量/LLM，开箱即用）
    uvicorn rag.api:app --host 0.0.0.0 --port 8000

    # 构建知识库 + 查询
    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question": "桌面云接入慢怎么排查？", "docs_path": "data/sample_docs"}'

    # 健康检查
    curl http://localhost:8000/health
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import RAGConfig
from .pipeline import RAGPipeline

app = FastAPI(
    title="Delivery RAG API",
    description="企业交付知识库 RAG 系统（容器化私有化部署）",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局单例 Pipeline（懒加载，首次查询时构建知识库）
_pipeline: Optional[RAGPipeline] = None
_BUILD_LOCK = False


class QueryRequest(BaseModel):
    question: str = Field(..., description="用户问题", min_length=1)
    docs_path: Optional[str] = Field(
        None, description="知识库文档目录（首次构建时使用；之后可省略）"
    )
    llm_provider: Optional[str] = Field(None, description="LLM provider: mock/openai/ollama")
    llm_model: Optional[str] = Field(None, description="模型名称")


class QueryResponse(BaseModel):
    question: str
    answer: str
    has_answer: bool
    confidence: float
    contexts: List[str] = Field(default_factory=list, description="答案引用的上下文文本")
    context_ids: List[int] = Field(default_factory=list, description="上下文编号（对应答案中的 [n] 引用）")


class HealthResponse(BaseModel):
    status: str
    version: str
    stats: dict


def _get_pipeline(docs_path: Optional[str] = None,
                  llm_provider: Optional[str] = None,
                  llm_model: Optional[str] = None) -> RAGPipeline:
    """懒加载/重建 Pipeline。"""

    global _pipeline, _BUILD_LOCK
    if _BUILD_LOCK:
        raise HTTPException(status_code=503, detail="知识库构建中，请稍后重试")

    if _pipeline is not None:
        return _pipeline

    config = RAGConfig.default()
    if llm_provider:
        config.llm_provider = llm_provider
    if llm_model:
        config.llm_model = llm_model

    _BUILD_LOCK = True
    try:
        pipe = RAGPipeline(config)
        docs = docs_path or os.environ.get("RAG_DOCS_PATH", "data/sample_docs")
        if not Path(docs).exists():
            raise HTTPException(status_code=400, detail=f"文档目录不存在: {docs}")
        pipe.build_index(docs)
        _pipeline = pipe
    finally:
        _BUILD_LOCK = False
    return _pipeline


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """健康检查 + 知识库状态。"""
    stats = _pipeline.get_stats() if _pipeline else {"built": False}
    return HealthResponse(status="ok", version="1.0.0", stats=stats)


@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest) -> QueryResponse:
    """检索增强问答。"""
    pipe = _get_pipeline(
        docs_path=req.docs_path,
        llm_provider=req.llm_provider,
        llm_model=req.llm_model,
    )
    answer = pipe.query(req.question)
    return QueryResponse(
        question=req.question,
        answer=answer.answer,
        has_answer=answer.has_answer,
        confidence=float(answer.confidence),
        contexts=list(answer.contexts),
        context_ids=list(answer.context_ids),
    )
