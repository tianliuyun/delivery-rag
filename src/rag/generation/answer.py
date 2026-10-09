"""
答案生成器
幻觉控制 + 引用溯源 + 低置信度主动拒绝
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .llm import LLMClient, ChatMessage
from .prompt import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT


@dataclass
class RAGAnswer:
    """RAG 回答结果"""
    answer: str
    contexts: List[str]           # 用到的上下文
    context_ids: List[int]        # 上下文编号（对应引用）
    confidence: float             # 置信度（0-1）
    has_answer: bool              # 是否找到了答案（没找到就是拒绝回答）
    source: str = "rag"           # 来源


class AnswerGenerator:
    """答案生成器 + 幻觉控制

    幻觉控制三件套：
    1. Prompt 约束 — 只从资料回答
    2. 引用溯源 — 答案带 [1][2] 引用
    3. 主动拒绝 — 低置信度直接说不知道
    """

    def __init__(self, llm: LLMClient,
                 enable_citations: bool = True,
                 enable_refusal: bool = True,
                 confidence_threshold: float = 0.3):
        self.llm = llm
        self.enable_citations = enable_citations
        self.enable_refusal = enable_refusal
        self.confidence_threshold = confidence_threshold

    def generate(self, question: str, contexts: List[Tuple[str, float]]) -> RAGAnswer:
        """生成回答

        Args:
            question: 用户问题
            contexts: [(上下文文本, 相关度分数)] 列表，已排序
        """
        if not contexts:
            return RAGAnswer(
                answer="抱歉，知识库中没有找到相关内容，请尝试换个问题或联系技术支持。",
                contexts=[],
                context_ids=[],
                confidence=0.0,
                has_answer=False,
            )

        # 置信度估算：取 Top3 的相关度分数
        top_scores = [s for _, s in contexts[:3]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
        confidence = min(1.0, avg_score)

        # 低置信度主动拒绝
        if self.enable_refusal and confidence < self.confidence_threshold:
            return RAGAnswer(
                answer="抱歉，这个问题我不太确定。建议您查阅完整的技术文档或联系技术支持获取准确答案。",
                contexts=[c for c, _ in contexts[:3]],
                context_ids=list(range(min(3, len(contexts)))),
                confidence=confidence,
                has_answer=False,
            )

        # 组装上下文（带编号）
        context_text = ""
        context_list = []
        for i, (text, score) in enumerate(contexts[:5]):
            context_text += f"\n[{i+1}] {text[:500]}\n"
            context_list.append(text)

        # 组装 Prompt
        user_prompt = RAG_USER_PROMPT.format(
            question=question,
            context=context_text
        )

        messages = [
            ChatMessage(role="system", content=RAG_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]

        # 调用 LLM
        answer = self.llm.chat(messages)

        return RAGAnswer(
            answer=answer,
            contexts=context_list,
            context_ids=list(range(len(context_list))),
            confidence=confidence,
            has_answer=True,
        )
