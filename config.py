from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Always resolve .env relative to this file, not the working directory
BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    model_config= SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        enc_file_encoding ="utf-8",
        extra="ignore"   # ignore any extra vars in .env
    )

    # Required — no default means the app crashes immediately at startup
    # if these are missing from .env, which is what you want
    pinecone_api_key: str
    pinecone_index: str

    # Optional — safe defaults
    groq_api_key: str
    llm_model: str = "llama-3.1-8b-instant"
    environment: str = "development"

# Singleton — create once here, import everywhere
settings = Settings()  