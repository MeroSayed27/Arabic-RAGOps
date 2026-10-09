from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    """Application configuration loaded from environment variables."""

    app_name: str = "Arabic-RAGOps"
    app_version: str = "0.1.0"
    llm_provider: str = os.getenv("LLM_PROVIDER", "")
    llm_model: str = os.getenv("LLM_MODEL", "")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    vector_db_url: str = os.getenv("VECTOR_DB_URL", "")
    vector_db_api_key: str = os.getenv("VECTOR_DB_API_KEY", "")


settings = Settings()