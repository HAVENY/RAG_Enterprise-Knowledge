from langchain_openai import ChatOpenAI

from .config import get_settings


settings = get_settings()


def get_llm():
    return ChatOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        model=settings.qwen_model,
        temperature=0.3,
    )


def ask_qwen(prompt: str) -> str:
    """
    调用 Qwen 模型，输入 prompt，返回模型回答。
    """
    llm = get_llm()

    response = llm.invoke(prompt)

    return response.content