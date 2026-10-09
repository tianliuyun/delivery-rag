## RAG知识库建设与效果评估指南

### 1. 文档清洗与分块策略

#### 1.1 清洗流程
1. 去重：使用 MinHash LSH，阈值设为 0.85。
2. 去噪：正则移除页眉页脚、`<!-- -->` 注释、连续空行（`\n{3,}` 替换为 `\n\n`）。
3. 格式归一：统一 PDF 提取后的换行符，合并断行（若行尾为中文且下一行首非标点，则拼接）。
4. 元数据注入：为每个 chunk 添加 `source`、`page`、`section`、`timestamp`。

#### 1.2 分块策略
- **语义分块**：使用 `text-embedding-3-small` 计算相邻句子余弦相似度，阈值低于 0.7 时切分。最大块长 512 token，重叠 64 token。
- **父子块**：
  - 父块：1024 token，用于生成回答上下文。
  - 子块：256 token，用于检索。子块命中后返回其父块。
  - 实现：LangChain `ParentDocumentRetriever`，`child_splitter=RecursiveCharacterTextSplitter(chunk_size=256)`，`parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1024)`。

### 2. 混合检索与融合

#### 2.1 向量检索
- 模型：`bge-m3`，维度 1024。
- 向量库：Milvus 2.4，端口 `19530`。
- 索引：HNSW，`M=16`，`efConstruction=200`，检索 `ef=128`。
- 返回 Top 20。

#### 2.2 BM25 检索
- 使用 Elasticsearch 8.x，端口 `9200`。
- 索引 mapping：`content` 字段 `type: text`，`analyzer: ik_max_word`。
- 返回 Top 20。

#### 2.3 RRF 融合
- 公式：`score = Σ 1/(k + rank_i)`，`k=60`。
- 融合后取 Top 10 进入重排。

### 3. 重排与幻觉控制

#### 3.1 CrossEncoder 重排
- 模型：`bge-reranker-large`。
- 输入：query + 候选文档，输出相关性分数。
- 阈值：分数 < 0.3 的文档丢弃。保留 Top 3 作为最终上下文。

#### 3.2 幻觉控制
- **Prompt 约束**：
  ```
  仅根据以下上下文回答。若上下文不包含答案，回复“根据现有资料无法回答”。
  上下文：
  {context}
  问题：{question}
  ```
- **引用溯源**：每个句子末尾添加 `[source:page]`，如 `[doc1:5]`。
- **低置信拒答**：若重排后最高分 < 0.4 或上下文总 token < 50，直接返回拒答模板。

### 4. RAGAS 评估指标

#### 4.1 环境
- 安装：`pip install ragas==0.1.7`
- 评估数据集：`question`、`answer`、`contexts`、`ground_truth` 四列。

#### 4.2 核心指标
- **Faithfulness**：答案中每个陈述是否可由上下文推断。阈值 ≥ 0.85。
- **Answer Relevancy**：答案与问题的语义相似度。阈值 ≥ 0.80。
- **Context Precision**：相关文档在检索结果中的排名质量。阈值 ≥ 0.75。
- **Hit Rate**：Top K 中是否包含正确文档。K=3，阈值 ≥ 0.90。

#### 4.3 执行命令
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, hit_rate

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, hit_rate]
)
print(result)
```

#### 4.4 调优闭环
1. 若 Faithfulness < 0.85：收紧 Prompt，增加拒答阈值。
2. 若 Hit Rate < 0.90：调整分块大小或提高 RRF 的 `k` 至 80。
3. 若 Context Precision < 0.75：提升重排阈值至 0.35，或换用 `bge-reranker-v2-m3`。