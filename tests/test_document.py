"""
文档处理模块测试
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from rag.document.loader import DocumentLoader
from rag.document.chunker import TextChunker


def test_document_loader():
    """测试文档加载"""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("# Test\n\nThis is a test document.\n\n## Section 1\n\nHello world.")
        tmp_path = f.name

    try:
        docs = DocumentLoader.load(tmp_path)
        assert len(docs) == 1
        assert "Test" in docs[0].content
        assert docs[0].metadata['filename'].endswith('.md')
    finally:
        os.unlink(tmp_path)


def test_fixed_chunking():
    """测试固定分块"""
    text = "a" * 1000
    chunker = TextChunker(strategy="fixed", child_size=256, overlap=50)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert all(len(c.text) <= 256 + 10 for c in chunks)  # 允许少量误差


def test_semantic_chunking():
    """测试语义分块"""
    text = """# Title

First section content.
More content here.

## Section 2

Second section.
Different topic.

### Subsection

Deep dive into details.
"""
    chunker = TextChunker(strategy="semantic")
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2  # 至少应该分成几段
    assert all(len(c.text.strip()) > 0 for c in chunks)


def test_parent_child_chunking():
    """测试父子块策略"""
    text = "# Main\n\n" + "\n\n".join([f"## Section {i}\n\n" + "Content " * 100 for i in range(5)])
    chunker = TextChunker(strategy="semantic_parent_child", child_size=100, parent_size=500)
    chunks = chunker.chunk(text)

    assert len(chunks) > 5  # 应该有多个子块
    assert hasattr(chunker, '_parent_chunks')
    assert len(chunker._parent_chunks) > 0

    # 每个子块都应该有父块索引
    for chunk in chunks:
        if chunk.parent_index >= 0:
            parent = chunker.get_parent(chunk)
            assert parent is not None
            assert len(parent.text) >= len(chunk.text)


def test_chunk_metadata():
    """测试 chunk metadata"""
    chunker = TextChunker(strategy="fixed")
    chunks = chunker.chunk("hello world")
    assert hasattr(chunks[0], 'metadata')
    assert chunks[0].index == 0
