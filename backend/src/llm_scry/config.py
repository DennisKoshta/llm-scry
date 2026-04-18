from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_SCRY_", env_file=".env", extra="ignore")

    default_device: str = "cuda"
    session_cache_size: int = 10
    cors_origins: list[str] = ["http://localhost:5173"]
    default_max_new_tokens: int = 32
    default_top_k: int = 10


settings = Settings()
