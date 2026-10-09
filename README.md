# Delivery RAG — 企业交付知识库 RAG 系统

> 轻量级生产级 RAG（检索增强生成）系统，专为企业技术交付场景设计。
> 基于真实工作场景：云计算交付团队知识库，让新人快速上手、老工程师少被打扰。

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![RAG](https://img.shields.io/badge/RAG-生产级-orange)
![tests](https://img.shields.io/badge/tests-19%20passed-brightgreen)

---

## ✨ 特性

- **混合检索**：向量检索（BGE）+ BM25 关键词检索 + RRF 融合，准确率远超纯向量
- **三向量库后端**：FAISS（轻量，默认）/ Milvus（企业级向量库）/ Elasticsearch（搜索引擎+向量一体），`config.vector_store` 一键切换，三者均实测通过（19 测）
- **容器化部署**：Docker + FastAPI 微服务，全链路私有化独立部署
- **可观测性**：OpenTelemetry 标准埋点 → Phoenix trace 全链路（检索/生成/评测逐 span），`scripts/run_evaluation.py` 可复现
- **语义分块**：智能语义分块 + 父子块（Parent-Child）策略，检索更准
- **精排重排**：CrossEncoder 重排模型，Top-K 准确率提升 20%+
- **幻觉控制三件套**：Prompt 约束 + 答案引用溯源 + 低置信度主动拒绝
- **多模型支持**：支持 OpenAI / Qwen / Claude / 本地 Ollama / vLLM
- **评估体系**：RAGAS 四指标 + Hit Rate + MRR，全面评估效果
- **增量更新**：文档修改自动重新向量化，知识库持续保持最新
- **Web UI**：基于 Gradio 的简洁界面，开箱即用
- **CLI 工具**：命令行一键建库、检索、问答

---

## 🏗️ 架构

```
用户提问
   │
   ▼
┌─────────────┐
│  查询改写    │  Query rewriting
└──────┬──────┘
       │
       ▼
┌─────────────────────────────┐
│     混合检索（Hybrid）       │
│  ┌───────┐   ┌──────────┐   │
│  │ 向量库 │   │ BM25索引  │   │
│  └───┬───┘   └────┬─────┘   │
│      └─────┬──────┘         │
│            ▼                │
│      RRF 融合（融合排序）    │
└────────────┬────────────────┘
             │ Top-K (50)
             ▼
┌─────────────────┐
│ CrossEncoder 重排 │  Reranker
└────────┬────────┘
         │ Top-N (5)
         ▼
┌─────────────────┐
│  上下文组装      │  Context building
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM 生成       │  Answer generation
│   带引用溯源      │
│   幻觉控制       │
└────────┬────────┘
         │
         ▼
      答案 [1][2][3]
```

---

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/tianliuyun/delivery-rag.git
cd delivery-rag

# 安装依赖（推荐用虚拟环境）
pip install -e .

# 或开发模式
pip install -r requirements.txt
```

**依赖说明**：
- `sentence-transformers` — 向量模型（BGE）
- `rank-bm25` — BM25 关键词检索
- `FlagEmbedding` — CrossEncoder 重排
- `faiss-cpu` — 向量库（也可用 faiss-gpu）
- `langchain` — LLM 调用封装
- `gradio` — Web UI
- `ragas` — 评估（可选）

---

## 🚀 快速开始

### 1. 准备文档

把你的文档（Markdown / TXT / PDF）放到 `data/sample_docs/` 目录。

```bash
# 示例：自带了云计算交付文档样例
ls data/sample_docs/
```

### 2. 构建知识库

```bash
# 一键建库
python -m rag.cli build --docs data/sample_docs/ --output data/knowledge_base
```

建库过程：
1. 读取文档 → 清洗 → 语义分块（父子块策略）
2. 小块向量化 → 存入 FAISS
3. 构建 BM25 索引
4. 父块存入 SQLite

### 3. 命令行问答

```bash
# 交互式问答
python -m rag.cli chat --db data/knowledge_base

# 单条查询
python -m rag.cli query "桌面云接入慢怎么排查？" --db data/knowledge_base
```

### 4. 启动 Web UI

```bash
python -m rag.web.app
# 打开 http://localhost:7860
```

---

## 📁 项目结构

```
delivery-rag/
├── src/rag/
│   ├── __init__.py
│   ├── cli.py              # 命令行工具
│   ├── config.py           # 配置
│   ├── document/           # 文档处理
│   │   ├── loader.py       # 文档加载
│   │   ├── cleaner.py      # 文档清洗
│   │   └── chunker.py      # 语义分块 + 父子块
│   ├── retrieval/          # 检索层
│   │   ├── vector_store.py # 向量检索（FAISS）
│   │   ├── bm25.py         # BM25 关键词检索
│   │   ├── hybrid.py       # 混合检索 + RRF 融合
│   │   └── reranker.py     # CrossEncoder 重排
│   ├── generation/         # 生成层
│   │   ├── llm.py          # LLM 封装
│   │   ├── prompt.py       # Prompt 模板
│   │   └── answer.py       # 答案生成 + 引用 + 幻觉控制
│   ├── evaluation/         # 评估
│   │   └── metrics.py      # Hit Rate / MRR / RAGAS
│   └── pipeline.py         # 完整 RAG pipeline
├── src/web/
│   └── app.py              # Gradio Web UI
├── data/
│   └── sample_docs/        # 示例文档
├── tests/                  # 单元测试
├── examples/               # 使用示例
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🎯 效果评估

### 自动化评测（本机实测，Phoenix 观测）

`scripts/run_ab_eval.py` 基于 **100 条领域 QA 评测集**（25 篇交付文档 × 4 题，含 golden_context 命中判据）跑出三级方案对比；`scripts/run_rag_eval_deepseek.py` 跑端到端质量（DeepSeek 生成 + DeepSeek judge）：

| 方案 | Hit Rate@1 | Hit Rate@5 | MRR | 单题检索耗时 |
|------|-----------|-----------|-----|-------------|
| A 固定分块 + 纯向量 | 55% | 82% | 0.65 | 11ms |
| B 语义父子块 + 混合检索（BGE+BM25+RRF） | 81% | 99% | 0.89 | 13ms |
| C B + CrossEncoder 重排 | **91%** | 99% | **0.95** | 3.3s（CPU；GPU 快 10 倍+） |

| 端到端指标（30 题抽样，top_k=3） | 实测值 | 说明 |
|------|--------|------|
| Hit Rate@3 | **100%** | 检索命中率（golden_context 在 top-3 内） |
| Faithfulness | **0.967** | DeepSeek-as-Judge，答案忠实度（阈值 0.85 已达成并超出） |
| 平均检索耗时 | 6.3s | CPU + bge-reranker-base（GPU 可显著下降） |
| 平均生成耗时 | 0.76s | DeepSeek-chat |

> 数字口径：全部为本机实测（`scripts/run_ab_eval.py` / `scripts/run_rag_eval_deepseek.py` 一键复现，报告落盘 `results/`）。评测集由领域文档 LLM 生成 + golden_context 原文校验，100 条可复现。

---

## 🔧 配置说明

### 模型选择

在 `config.yaml` 中配置：

```yaml
# 向量模型（中文推荐 BGE）
embedding:
  model: "BAAI/bge-large-zh-v1.5"
  dimension: 1024

# 重排模型
reranker:
  model: "BAAI/bge-reranker-large"
  top_n: 5

# LLM 模型
llm:
  provider: "ollama"  # openai / qwen / ollama / vllm
  model: "qwen2:7b"
  base_url: "http://localhost:11434"

# 分块策略
chunking:
  strategy: "semantic_parent_child"  # fixed / semantic / parent_child
  child_size: 256
  parent_size: 1024
  overlap: 50
```

---

## 🧪 测试

```bash
# 运行单元测试
pytest tests/ -v

# 评估检索效果
python -m rag.evaluation.metrics --db data/knowledge_base --testset tests/testset.json
```

---

## 📊 适用场景

- 企业内部知识库问答
- 技术文档智能检索
- 产品手册查询
- 故障排查辅助
- 新人培训助手
- 客服问答系统

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

### 开发环境

```bash
pip install -e ".[dev]"
pre-commit install
```

---

## 📄 License

MIT License

---

## 🙏 致谢

- [BGE 系列模型](https://github.com/FlagOpen/FlagEmbedding) — 中文向量和重排模型
- [FAISS](https://github.com/facebookresearch/faiss) — 向量检索引擎
- [LangChain](https://github.com/langchain-ai/langchain) — LLM 应用框架
- [Gradio](https://github.com/gradio-app/gradio) — Web UI
