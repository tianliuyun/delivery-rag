from setuptools import setup, find_packages

setup(
    name="delivery-rag",
    version="0.1.0",
    description="企业交付知识库 RAG 系统 - 轻量级生产级检索增强生成",
    author="tianliuyun",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "full": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "rank-bm25>=0.2.2",
            "gradio>=4.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "delivery-rag=rag.cli:main",
        ],
    },
)
