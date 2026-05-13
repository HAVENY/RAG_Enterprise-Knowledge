from openai import OpenAI

from .config import get_settings


settings = get_settings()


class LLMClient:
    """
    多模型统一调用客户端。

    支持：
    - qwen
    - mimo

    支持模型档位：
    - fast
    - default
    - strong
    """

    allowed_providers = {"qwen", "mimo"}
    allowed_levels = {"fast", "default", "strong"}

    def normalize_provider(self, provider: str | None = None) -> str:
        if not provider:
            return settings.default_llm_provider

        provider = provider.lower().strip()

        if provider not in self.allowed_providers:
            return settings.default_llm_provider

        return provider

    def normalize_model_level(self, model_level: str | None = None) -> str:
        if not model_level:
            return settings.default_model_level

        model_level = model_level.lower().strip()

        if model_level not in self.allowed_levels:
            return settings.default_model_level

        return model_level

    def get_client(self, provider: str) -> OpenAI:
        """
        根据模型供应商创建 OpenAI-compatible client。
        """

        provider = self.normalize_provider(provider)

        if provider == "qwen":
            if not settings.dashscope_api_key:
                raise ValueError("缺少 DASHSCOPE_API_KEY，请检查 .env 配置")

            return OpenAI(
                api_key=settings.dashscope_api_key,
                base_url=settings.dashscope_base_url,
            )

        if provider == "mimo":
            if not settings.mimo_api_key:
                raise ValueError("缺少 MIMO_API_KEY，请检查 .env 配置")

            return OpenAI(
                api_key=settings.mimo_api_key,
                base_url=settings.mimo_base_url,
            )

        raise ValueError(f"不支持的模型供应商：{provider}")

    def select_model(
        self,
        provider: str | None = None,
        model_level: str | None = None,
    ) -> str:
        """
        根据 provider + model_level 选择具体模型名。
        """

        provider = self.normalize_provider(provider)
        model_level = self.normalize_model_level(model_level)

        model_map = {
            "qwen": {
                "fast": settings.qwen_fast_model,
                "default": settings.qwen_default_model,
                "strong": settings.qwen_strong_model,
            },
            "mimo": {
                "fast": settings.mimo_fast_model,
                "default": settings.mimo_default_model,
                "strong": settings.mimo_strong_model,
            },
        }

        return model_map[provider][model_level]

    def chat(
        self,
        prompt: str,
        provider: str | None = None,
        model_level: str | None = None,
    ) -> str:
        """
        统一聊天调用入口。
        """

        if not prompt or not prompt.strip():
            return "Prompt 不能为空。"

        provider = self.normalize_provider(provider)
        model_level = self.normalize_model_level(model_level)

        client = self.get_client(provider)
        model = self.select_model(
            provider=provider,
            model_level=model_level,
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个严谨、可靠的企业内部知识库问答助手。",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

            answer = response.choices[0].message.content

            if not answer:
                return "模型未返回有效回答。"

            return answer.strip()

        except Exception as e:
            return f"调用大模型失败：{str(e)}"


_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _llm_client

    if _llm_client is None:
        _llm_client = LLMClient()

    return _llm_client


def ask_llm(
    prompt: str,
    provider: str | None = None,
    model_level: str | None = None,
) -> str:
    """
    RAG 模块统一调用入口。
    """

    llm = get_llm_client()

    return llm.chat(
        prompt=prompt,
        provider=provider,
        model_level=model_level,
    )


def get_current_model_name(
    provider: str | None = None,
    model_level: str | None = None,
) -> str:
    """
    开发环境调试用：查看当前实际调用的模型名。
    """

    llm = get_llm_client()

    return llm.select_model(
        provider=provider,
        model_level=model_level,
    )