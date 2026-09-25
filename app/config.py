from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = "Persian RAG Preprocessing API"
    app_version: str = "1.0.0"
    api_key: str | None = os.getenv("API_KEY")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "100"))
    max_records: int = int(os.getenv("MAX_RECORDS", "10000"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")


settings = Settings()
