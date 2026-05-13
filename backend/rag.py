from sqlalchemy.orm import Session

from .llm import ask_llm, get_current_model_name
from .config import get_settings
from .vector_store import load_faiss


settings = get_settings()


def rag_answer(
    question: str,
    db: Session,
    provider: str = "qwen",
    model_level: str = "default",
    allow_general_answer: bool = False,
) -> dict:
    """
    基于 FAISS 检索的 RAG 问答。

    参数：
    question: 用户问题
    db: 数据库会话
    provider: 模型供应商，支持 qwen / mimo
    model_level: 模型档位，支持 fast / default / strong
    allow_general_answer: 当知识库无索引时，是否允许大模型基于通用知识回答
    """

    if not question or not question.strip():
        return build_response(
            answer="问题不能为空。",
            sources=[],
            provider=provider,
            model_level=model_level,
            mode="empty_question",
        )

    provider = normalize_provider(provider)
    model_level = normalize_model_level(model_level)

    vector_store = load_faiss()

    if vector_store is None:
        if allow_general_answer:
            prompt = f"""
你是一个通用 AI 助手。

当前知识库中还没有可用的向量索引，因此你可以基于通用知识回答用户问题。
但如果用户询问企业内部制度、流程、人员、项目、合同、财务、权限等内部信息，
请明确说明：当前知识库中没有可用依据，无法确认企业内部信息。

【用户问题】
{question}

【回答】
"""

            answer = ask_llm(
                prompt=prompt,
                provider=provider,
                model_level=model_level,
            )

            return build_response(
                answer=answer,
                sources=[],
                provider=provider,
                model_level=model_level,
                mode="general",
            )

        return build_response(
            answer="当前知识库中还没有可用的向量索引，请先上传文档并完成入库。",
            sources=[],
            provider=provider,
            model_level=model_level,
            mode="no_index",
        )

    docs = vector_store.similarity_search(
        query=question,
        k=settings.top_k,
    )

    if not docs:
        return build_response(
            answer="没有检索到与问题相关的知识库片段。",
            sources=[],
            provider=provider,
            model_level=model_level,
            mode="no_retrieval",
        )

    context = build_context(docs)

    prompt = build_rag_prompt(
        question=question,
        context=context,
    )

    answer = ask_llm(
        prompt=prompt,
        provider=provider,
        model_level=model_level,
    )

    sources = build_sources(docs)

    return build_response(
        answer=answer,
        sources=sources,
        provider=provider,
        model_level=model_level,
        mode="rag",
    )


def normalize_provider(provider: str) -> str:
    """
    规范化模型供应商。
    """

    allowed_providers = {"qwen", "mimo"}

    if not provider:
        return "qwen"

    provider = provider.lower().strip()

    if provider not in allowed_providers:
        return "qwen"

    return provider


def normalize_model_level(model_level: str) -> str:
    """
    规范化模型档位。
    避免用户乱传模型名或非法参数。
    """

    allowed_levels = {"fast", "default", "strong"}

    if not model_level:
        return "default"

    model_level = model_level.lower().strip()

    if model_level not in allowed_levels:
        return "default"

    return model_level


def build_context(docs) -> str:
    """
    将 FAISS 检索结果拼接成上下文。
    """

    context = "\n\n".join(
        [
            f"【片段 {index + 1}】\n{doc.page_content}"
            for index, doc in enumerate(docs)
        ]
    )

    return context


def build_rag_prompt(question: str, context: str) -> str:
    """
    构造 RAG Prompt。
    """

    prompt = f"""
你是一个企业内部知识库问答助手。

请优先根据【检索到的知识库片段】回答用户问题。

如果片段中有明确答案，请直接基于片段回答。
如果片段信息不足，请明确说明“根据当前知识库片段无法完整确定”，不要编造企业内部信息。

回答要求：
1. 回答必须围绕用户问题；
2. 优先使用知识库片段中的信息；
3. 不要编造片段中不存在的企业制度、流程、数据或结论；
4. 回答要清晰、准确、结构化；
5. 如果涉及流程、步骤、制度，请使用分点说明。

【检索到的知识库片段】
{context}

【用户问题】
{question}

【回答】
"""

    return prompt


def build_sources(docs) -> list[dict]:
    """
    构造来源信息。
    """

    sources = []

    for doc in docs:
        metadata = doc.metadata or {}

        sources.append(
            {
                "document_id": metadata.get("document_id"),
                "chunk_index": metadata.get("chunk_index"),
                "source": metadata.get("source"),
                "filename": metadata.get("filename"),
                "page": metadata.get("page"),
                "content_preview": doc.page_content[:120],
            }
        )

    return sources


def build_response(
    answer: str,
    sources: list,
    provider: str,
    model_level: str,
    mode: str,
) -> dict:
    """
    统一构造返回结果。
    """

    response = {
        "answer": answer,
        "sources": sources,
        "provider": provider,
        "model_level": model_level,
        "mode": mode,
    }

    if settings.app_env == "development":
        response["model"] = get_current_model_name(
            provider=provider,
            model_level=model_level,
        )

    return response