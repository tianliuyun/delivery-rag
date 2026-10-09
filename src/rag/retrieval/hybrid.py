"""
混合检索 + RRF 融合
向量检索 + BM25 关键词检索，RRF 融合排序
"""
from typing import List, Tuple, Dict


class HybridRetriever:
    """混合检索器

    核心思想：向量检索管语义，BM25 管精确匹配，两者互补。
    用 RRF（Reciprocal Rank Fusion）融合，不用调权重，鲁棒性好。
    """

    def __init__(self, vector_store, bm25_retriever, rrf_k: int = 60):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k  # RRF 公式中的 k 参数，一般 60 效果较好

    def search(self, query: str, top_k: int = 50,
               use_vector: bool = True, use_bm25: bool = True) -> List[Tuple[int, float, str]]:
        """混合搜索

        返回: [(doc_id, 融合分数, 来源)]
        来源: vector / bm25 / hybrid
        """
        vector_results = {}
        bm25_results = {}

        if use_vector:
            vec_res = self.vector_store.search(query, top_k)
            for rank, (doc_id, score) in enumerate(vec_res):
                vector_results[doc_id] = rank + 1  # rank 从 1 开始

        if use_bm25:
            bm25_res = self.bm25_retriever.search(query, top_k)
            for rank, (doc_id, score) in enumerate(bm25_res):
                bm25_results[doc_id] = rank + 1

        # RRF 融合
        if use_vector and use_bm25:
            return self._rrf_fusion(vector_results, bm25_results, top_k)
        elif use_vector:
            result = []
            for doc_id, rank in sorted(vector_results.items(), key=lambda x: x[1])[:top_k]:
                result.append((doc_id, 1.0 / (self.rrf_k + rank), "vector"))
            return result
        else:
            result = []
            for doc_id, rank in sorted(bm25_results.items(), key=lambda x: x[1])[:top_k]:
                result.append((doc_id, 1.0 / (self.rrf_k + rank), "bm25"))
            return result

    def _rrf_fusion(self, vector_ranks: Dict[int, int],
                    bm25_ranks: Dict[int, int],
                    top_k: int) -> List[Tuple[int, float, str]]:
        """RRF (Reciprocal Rank Fusion) 融合

        公式：score(d) = Σ 1 / (k + rank)
        对每个检索系统，根据排名越靠前的文档，把它们的倒数排名加起来。

        为什么用 RRF 不用加权？因为不同检索的分数分布差异很大，直接加权很难调。
        RRF 基于排名的融合，鲁棒性更好，不用调参。
        """
        all_docs = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        scores = {}

        for doc_id in all_docs:
            score = 0.0
            source_parts = []

            if doc_id in vector_ranks:
                score += 1.0 / (self.rrf_k + vector_ranks[doc_id])
                source_parts.append("vector")

            if doc_id in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[doc_id])
                source_parts.append("bm25")

            scores[doc_id] = (score, "+".join(source_parts))

        # 按分数排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        return [(doc_id, score, source) for doc_id, (score, source) in sorted_docs[:top_k]]
