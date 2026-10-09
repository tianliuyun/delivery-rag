"""
语义分块器
支持：固定分块 / 语义分块 / 父子块（Parent-Child）策略

技术文档推荐用父子块策略：
- 子块（256 token）：用于检索，精确
- 父块（1024 token）：用于生成，上下文完整
"""
from typing import List
from dataclasses import dataclass
import re


@dataclass
class Chunk:
    """文本块"""
    text: str
    index: int
    parent_index: int = -1  # 父块索引，-1 表示没有
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """文本分块器"""

    def __init__(self, strategy: str = "semantic_parent_child",
                 child_size: int = 256,
                 parent_size: int = 1024,
                 overlap: int = 50):
        self.strategy = strategy
        self.child_size = child_size
        self.parent_size = parent_size
        self.overlap = overlap

    def chunk(self, text: str) -> List[Chunk]:
        """根据策略分块"""
        if self.strategy == "fixed":
            return self._fixed_chunk(text)
        elif self.strategy == "semantic":
            return self._semantic_chunk(text)
        elif self.strategy == "semantic_parent_child":
            return self._parent_child_chunk(text)
        else:
            raise ValueError(f"未知分块策略: {self.strategy}")

    def _fixed_chunk(self, text: str) -> List[Chunk]:
        """固定大小分块（基线方案）"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.child_size
            chunk_text = text[start:end]
            chunks.append(Chunk(text=chunk_text, index=idx))
            idx += 1
            start = end - self.overlap if end < len(text) else end
        return chunks

    def _semantic_chunk(self, text: str) -> List[Chunk]:
        """语义分块（按段落/标题/句子边界切）

        技术文档有清晰的结构，按语义边界切比固定大小准确率高。
        切分优先级：标题 > 段落 > 句子 > 固定大小兜底
        """
        # 先按 markdown 标题切大段
        sections = re.split(r'\n(?=#{1,6}\s)', text)
        chunks = []
        idx = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 如果 section 太大，再按段落切
            if len(section) > self.parent_size:
                paragraphs = re.split(r'\n\s*\n', section)
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) < self.parent_size:
                        current += "\n\n" + para if current else para
                    else:
                        if current:
                            chunks.append(Chunk(text=current.strip(), index=idx))
                            idx += 1
                        current = para
                if current:
                    chunks.append(Chunk(text=current.strip(), index=idx))
                    idx += 1
            else:
                chunks.append(Chunk(text=section.strip(), index=idx))
                idx += 1

        return chunks

    def _parent_child_chunk(self, text: str) -> List[Chunk]:
        """父子块（Parent-Child）策略

        子块用于检索（小、精确），父块用于生成（大、上下文完整）。
        子块命中后，返回对应的父块给 LLM。

        这是生产级 RAG 的推荐方案，兼顾检索精度和生成质量。
        """
        # 先做语义分块得到父块
        parent_chunks = self._semantic_chunk(text)

        # 每个父块再切成子块
        child_chunks = []
        child_idx = 0

        for parent_idx, parent in enumerate(parent_chunks):
            parent_text = parent.text

            if len(parent_text) <= self.child_size:
                # 父块本身就不大，子块就是它自己
                child_chunks.append(Chunk(
                    text=parent_text,
                    index=child_idx,
                    parent_index=parent_idx,
                    metadata={'parent_size': len(parent_text)}
                ))
                child_idx += 1
            else:
                # 把父块切成更小的子块，子块间有 overlap
                start = 0
                while start < len(parent_text):
                    end = start + self.child_size
                    child_text = parent_text[start:end]
                    child_chunks.append(Chunk(
                        text=child_text,
                        index=child_idx,
                        parent_index=parent_idx,
                        metadata={'parent_size': len(parent_text)}
                    ))
                    child_idx += 1
                    start = end - self.overlap if end < len(parent_text) else end

        # 把父块存到 metadata 里（实际项目中存 SQLite）
        self._parent_chunks = parent_chunks
        return child_chunks

    def get_parent(self, child_chunk: Chunk) -> Chunk:
        """根据子块获取父块"""
        if hasattr(self, '_parent_chunks') and child_chunk.parent_index >= 0:
            return self._parent_chunks[child_chunk.parent_index]
        return child_chunk
