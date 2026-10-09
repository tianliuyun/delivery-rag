"""
CrossEncoder 重排（Reranker）
粗排后的 Top-K 用 CrossEncoder 精排，提升 10-20% 准确率
"""
from typing import List, Tuple


class Reranker:
    """CrossEncoder 重排器

    为什么需要重排：
    - 向量检索（双塔）：快，但两个塔独立编码，交互少，排序不够准
    - CrossEncoder（单塔）：问题和文档一起输入，充分交互，更准但慢
    - 所以先用向量检索粗排召回（50-100条），再用 CrossEncoder 精排（5-10条），又快又准
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", use_gpu: bool = False):
        self.model_name = model_name
        self._model = None
        self._use_gpu = use_gpu
        self._mock_mode = False

    def load(self):
        """加载模型"""
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            print(f"✅ 重排模型加载成功: {self.model_name}")
            return True
        except ImportError:
            print(f"⚠️  sentence-transformers 未安装，使用 mock 重排模式")
            self._mock_mode = True
            return False

    def rerank(self, query: str, documents: List[str], top_n: int = 5) -> List[Tuple[int, float]]:
        """重排

        Args:
            query: 查询
            documents: 待重排的文档列表
            top_n: 返回前 N 个

        Returns:
            [(原索引, 重排分数)]，按分数降序
        """
        if not documents:
            return []

        if self._mock_mode or self._model is None:
            # Mock 模式：按文档长度和 query 关键词匹配粗略打分
            return self._mock_rerank(query, documents, top_n)

        # 真实 CrossEncoder 重排
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)

        # 排序并返回原索引
        scored = [(i, float(scores[i])) for i in range(len(documents))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _mock_rerank(self, query: str, documents: List[str], top_n: int) -> List[Tuple[int, float]]:
        """Mock 重排（仅用于演示）
        用简单的关键词匹配度打分
        """
        query_words = set(query.lower().split())
        query_chars = set(query)

        scored = []
        for i, doc in enumerate(documents):
            doc_lower = doc.lower()
            # 关键词匹配数
            word_match = sum(1 for w in query_words if w in doc_lower)
            char_match = sum(1 for c in query if c in doc_lower)
            # 长度归一化
            score = word_match * 2 + char_match * 0.1
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]
