from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # =====================
    # App
    # =====================
    app_name: str = "Enterprise Knowledge Base"
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # =====================
    # Path
    # =====================
    data_dir: Path = BASE_DIR / "data"
    uploads_dir: Path = BASE_DIR / "data" / "uploads"
    vector_store_dir: Path = BASE_DIR / "backend" / "vector_store" / "faiss_index"

    # =====================
    # Database
    # =====================
    database_url: str = f"sqlite:///{(BASE_DIR / 'enterprise_kb.db').as_posix()}"

    # =====================
    # Embedding
    # =====================
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # =====================
    # RAG
    # =====================
    top_k: int = 3
    score_threshold: float = 1.2

    # =====================
    # 默认 LLM 配置
    # =====================
    default_llm_provider: str = "qwen"
    default_model_level: str = "default"

    llm_temperature: float = 0.2
    llm_max_tokens: int = 1200

    # =====================
    # Qwen / DashScope
    # =====================
    dashscope_api_key: str | None = None
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    qwen_model: str = "qwen-plus"
    qwen_fast_model: str = "qwen-turbo"
    qwen_default_model: str = "qwen-plus"
    qwen_strong_model: str = "qwen-max"

    # =====================
    # MIMO
    # =====================
    mimo_api_key: str | None = None
    mimo_base_url: str = "https://token-plan-cn.xiaomimimo.com/v1"

    mimo_fast_model: str = "mimo-v2-omni"
    mimo_default_model: str = "mimo-v2.5"
    mimo_strong_model: str = "mimo-v2.5-pro"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()