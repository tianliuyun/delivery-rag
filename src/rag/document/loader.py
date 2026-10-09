"""
文档加载器
支持 .md / .txt / .pdf 等格式
"""
from pathlib import Path
from typing import List, Dict
import re


class Document:
    """文档对象"""
    def __init__(self, content: str, metadata: Dict = None, source: str = ""):
        self.content = content
        self.metadata = metadata or {}
        self.source = source

    def __repr__(self):
        return f"Document(source={self.source[:30]}..., len={len(self.content)})"


class DocumentLoader:
    """文档加载器"""

    SUPPORTED_EXTENSIONS = {'.md', '.txt', '.mdx', '.rst'}

    @classmethod
    def load_file(cls, file_path: str | Path) -> Document:
        """加载单个文件"""
        path = Path(file_path)
        if path.suffix.lower() not in cls.SUPPORTED_EXTENSIONS:
            # 未知格式，按文本读
            pass

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        metadata = {
            'filename': path.name,
            'suffix': path.suffix,
            'size': len(content),
        }

        return Document(content=content, metadata=metadata, source=str(path))

    @classmethod
    def load_directory(cls, dir_path: str | Path) -> List[Document]:
        """加载目录下所有支持的文档"""
        path = Path(dir_path)
        docs = []
        for file_path in sorted(path.rglob('*')):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS:
                try:
                    doc = cls.load_file(file_path)
                    docs.append(doc)
                except Exception as e:
                    print(f"⚠️  加载失败 {file_path}: {e}")
        return docs

    @classmethod
    def load(cls, path: str | Path) -> List[Document]:
        """自动判断是文件还是目录"""
        p = Path(path)
        if p.is_file():
            return [cls.load_file(p)]
        elif p.is_dir():
            return cls.load_directory(p)
        else:
            raise FileNotFoundError(f"路径不存在: {path}")
