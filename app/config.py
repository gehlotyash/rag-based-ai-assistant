from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: Optional[str] = None
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_results: int = 3
    model_name: str = "claude-3-5-sonnet-20241022"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
