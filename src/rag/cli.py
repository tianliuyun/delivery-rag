"""
命令行工具
用法：
  python -m rag.cli build --docs data/sample_docs --output data/kb
  python -m rag.cli query "问题" --db data/kb
  python -m rag.cli chat --db data/kb
"""
import argparse
import sys
import os

# 确保能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .pipeline import RAGPipeline
from .config import RAGConfig


def cmd_build(args):
    """构建知识库"""
    print(f"🔨 构建知识库 from {args.docs}")
    config = RAGConfig(
        llm_provider=args.llm or "mock",
        llm_model=args.model or "mock",
        use_reranker=not args.no_reranker,
        chunk_strategy=args.chunk_strategy,
    )
    rag = RAGPipeline(config)
    rag.build_index(args.docs)
    if args.output:
        rag.save(args.output)
    print("✅ 知识库构建完成！")


def cmd_query(args):
    """单条查询"""
    config = RAGConfig(
        llm_provider=args.llm or "mock",
        llm_model=args.model or "mock",
    )
    rag = RAGPipeline(config)
    # 简化版：直接用 docs 建库再查
    rag.build_index(args.docs)
    answer = rag.query(args.query)
    print(f"\n{'='*60}")
    print(f"❓ 问题: {args.query}")
    print(f"📊 置信度: {answer.confidence:.2f}")
    print(f"{'='*60}\n")
    print(answer.answer)
    print(f"\n📚 参考资料: {len(answer.contexts)} 段")
    for i, ctx in enumerate(answer.contexts[:3]):
        print(f"  [{i+1}] {ctx[:80]}...")


def cmd_chat(args):
    """交互式对话"""
    config = RAGConfig(
        llm_provider=args.llm or "mock",
        llm_model=args.model or "mock",
    )
    rag = RAGPipeline(config)
    rag.build_index(args.docs)
    print("🤖 交付知识库助手已就绪，输入 'quit' 退出")
    print("-" * 60)
    while True:
        try:
            question = input("\n❓ 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break
        if not question or question.lower() in ('quit', 'exit', 'q'):
            print("👋 再见！")
            break
        answer = rag.query(question)
        print(f"\n🤖 助手: {answer.answer}")
        if not answer.has_answer:
            print(f"   (置信度: {answer.confidence:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Delivery RAG - 企业交付知识库 RAG 系统")
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # build
    p_build = subparsers.add_parser('build', help='构建知识库')
    p_build.add_argument('--docs', required=True, help='文档目录')
    p_build.add_argument('--output', help='输出目录')
    p_build.add_argument('--chunk-strategy', default='semantic_parent_child',
                        choices=['fixed', 'semantic', 'semantic_parent_child'],
                        help='分块策略')
    p_build.add_argument('--no-reranker', action='store_true', help='不使用重排')
    p_build.add_argument('--llm', help='LLM provider (mock/openai/ollama)')
    p_build.add_argument('--model', help='模型名称')

    # query
    p_query = subparsers.add_parser('query', help='单条查询')
    p_query.add_argument('query', help='问题')
    p_query.add_argument('--docs', required=True, help='文档目录')
    p_query.add_argument('--llm', help='LLM provider')
    p_query.add_argument('--model', help='模型名称')

    # chat
    p_chat = subparsers.add_parser('chat', help='交互式对话')
    p_chat.add_argument('--docs', required=True, help='文档目录')
    p_chat.add_argument('--llm', help='LLM provider')
    p_chat.add_argument('--model', help='模型名称')

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'build':
        cmd_build(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'chat':
        cmd_chat(args)


if __name__ == '__main__':
    main()
