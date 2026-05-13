from backend.config import get_settings

settings = get_settings()

print("DEFAULT_LLM_PROVIDER =", settings.default_llm_provider)
print("DEFAULT_MODEL_LEVEL =", settings.default_model_level)

print("DASHSCOPE_BASE_URL =", settings.dashscope_base_url)
print("QWEN_FAST_MODEL =", settings.qwen_fast_model)
print("QWEN_DEFAULT_MODEL =", settings.qwen_default_model)
print("QWEN_STRONG_MODEL =", settings.qwen_strong_model)

print("MIMO_BASE_URL =", settings.mimo_base_url)
print("MIMO_FAST_MODEL =", settings.mimo_fast_model)
print("MIMO_DEFAULT_MODEL =", settings.mimo_default_model)
print("MIMO_STRONG_MODEL =", settings.mimo_strong_model)