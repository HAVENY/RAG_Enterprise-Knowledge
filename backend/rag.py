from sqlalchemy.orm import Session

from .llm import ask_qwen
from .models import Document


from sqlalchemy.orm import Session

from .config import get_settings
from .llm import ask_qwen
from .vector_store import load_faiss


settings = get_settings()


def rag_answer(question: str, db: Session) -> dict:
    """
    基于 FAISS 检索的 RAG 问答。
    """

    vector_store = load_faiss()

    if vector_store is None:
        prompt = f"""
你是一个通用 AI 助手。
当前知识库中还没有可用的向量索引，因此请基于你的通用知识回答用户问题。

【用户问题】
{question}

【回答】
"""
        answer = ask_qwen(prompt)

        return {
            "answer": answer,
            "sources": [],
        }

    docs = vector_store.similarity_search(
        query=question,
        k=settings.top_k,
    )

    if not docs:
        return {
            "answer": "没有检索到与问题相关的知识库片段。",
            "sources": [],
        }

    context = "\n\n".join(
        [
            f"【片段 {index + 1}】\n{doc.page_content}"
            for index, doc in enumerate(docs)
        ]
    )

    prompt = f"""
你是一个企业内部知识库问答助手。

请优先根据【检索到的知识库片段】回答用户问题。
如果片段中有明确答案，请直接基于片段回答。
如果片段信息不足，请明确说明“根据当前知识库片段无法完整确定”，不要编造企业内部信息。

回答要求：
1. 回答必须围绕用户问题；
2. 优先使用知识库片段中的信息；
3. 不要编造片段中不存在的企业制度、流程、数据或结论；
4. 回答要清晰、准确、结构化。

【检索到的知识库片段】
{context}

【用户问题】
{question}

【回答】
"""

    answer = ask_qwen(prompt)

    sources = []

    for doc in docs:
        metadata = doc.metadata or {}

        sources.append(
            {
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "source": metadata.get("source"),
                "content_preview": doc.page_content[:120],
            }
        )

    return {
        "answer": answer,
        "sources": sources,
    }