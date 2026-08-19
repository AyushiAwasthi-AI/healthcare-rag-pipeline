from pathlib import Path            # for handling file system Paths
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

# Always resolve .env relative to this file, not the working directory
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config= SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding ="utf-8",
        extra="ignore"   # ignore any extra vars in .env
    )

    # Required — no default means the app crashes immediately at startup
    # if these are missing from .env, which is what you want
    pinecone_api_key: str
    pinecone_index: str

    # Optional — safe defaults
    groq_api_key: str
    llm_model: str = "openai/gpt-oss-120b"
    environment: str = "development"

    # inside Settings class:
    langsmith_api_key: Optional[str] = None
    langchain_tracing_v2: str = "false"
    langchain_project: str = "healthcare-rag-pipeline"

# Singleton — create once here, import everywhere
settings = Settings()  

# Set LangSmith env vars immediately at import time
# LangSmith SDK reads os.environ at import, not at runtime
if settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_API_KEY"]     = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"]     = settings.langchain_project