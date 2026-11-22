from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    openai_api_key: str = ""
    openai_extraction_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    cv_data_dir: Path = Path("data")
