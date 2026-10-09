# Delivery RAG — 容器化私有化部署镜像
# 基于 Python 3.11 slim，纯 CPU 运行，开箱即用（mock 模式）
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（轻量）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 依赖层（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

# 应用代码
COPY src/ /app/src/
COPY data/ /app/data/
ENV PYTHONPATH=/app/src
ENV RAG_DOCS_PATH=/app/data/sample_docs

# 非 root 运行（安全基线）
RUN useradd -m rag && chown -R rag:rag /app
USER rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
