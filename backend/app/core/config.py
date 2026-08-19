from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Only needed by rag_core's LLM backend selection — read straight from
    # the environment there, listed here so `.env` has one obvious home.
    groq_api_key: str | None = None
    openai_api_key: str | None = None

    cors_allow_origins: list[str] = ["http://localhost:5173"]


settings = Settings()
