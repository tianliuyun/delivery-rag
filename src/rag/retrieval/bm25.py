"""
BM25 关键词检索
基于 rank-bm25，简单高效
"""
from typing import List, Tuple
import re


class BM25Retriever:
    """BM25 关键词检索器

    作为向量检索的补充，擅长：
    - 专有名词、产品型号、精确术语
    - 数字、编号、代码
    - 向量检索语义相近但不是要找的内容的场景
    """

    def __init__(self):
        self.documents = []
        self.tokenized_docs = []
        self.bm25 = None
        self._initialized = False

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（中文按字符 + 英文单词）

        注：生产环境推荐用 jieba 或专门的 tokenizer
        这里用简单实现保证零依赖可用
        """
        text = text.lower()
        # 提取英文单词和数字
        words = re.findall(r'[a-zA-Z0-9_\-\.]+', text)
        # 中文字符也作为 token
        chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 2-gram 中文（更适合 BM25）
        bigrams = []
        for i in range(len(chars) - 1):
            bigrams.append(chars[i] + chars[i+1])
        return words + bigrams

    def index(self, documents: List[str]):
        """构建 BM25 索引"""
        self.documents = documents
        self.tokenized_docs = [self._tokenize(doc) for doc in documents]

        # 简单 BM25 实现（基于 Okapi BM25）
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.tokenized_docs)
            self._use_library = True
        except ImportError:
            # 没有 rank_bm25 库，用简化版 TF-IDF
            self.bm25 = None
            self._use_library = False
            # 计算 IDF
            self._build_simple_bm25()

        self._initialized = True

    def _build_simple_bm25(self):
        """简化版 BM25（TF-IDF + 简单打分）"""
        import math
        N = len(self.tokenized_docs)
        # 计算 DF
        df = {}
        for tokens in self.tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1

        # 计算 IDF
        self.idf = {}
        for token, freq in df.items():
            self.idf[token] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))

        # 计算文档长度和平均长度
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avgdl = sum(self.doc_lengths) / max(N, 1)
        self._use_library = False

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """搜索，返回 (文档索引, 分数) 列表"""
        if not self._initialized:
            raise RuntimeError("请先调用 index() 构建索引")

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        if self._use_library and self.bm25 is not None:
            scores = self.bm25.get_scores(query_tokens)
            # 排序
            scored = [(i, s) for i, s in enumerate(scores)]
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
        else:
            return self._simple_search(query_tokens, top_k)

    def _simple_search(self, query_tokens: List[str], top_k: int) -> List[Tuple[int, float]]:
        """简化版搜索"""
        import math
        k1 = 1.5
        b = 0.75
        N = len(self.tokenized_docs)

        scores = []
        for doc_idx, doc_tokens in enumerate(self.tokenized_docs):
            score = 0.0
            dl = len(doc_tokens)
            tf_dict = {}
            for t in doc_tokens:
                tf_dict[t] = tf_dict.get(t, 0) + 1

            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = tf_dict.get(token, 0)
                if tf == 0:
                    continue
                # BM25 打分公式
                idf = self.idf[token]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * dl / self.avgdl)
                score += idf * numerator / denominator

            scores.append((doc_idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
