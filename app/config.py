from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Ollama OpenAI-compatible server
    OPENAI_BASE_URL: str = "http://localhost:11434/v1"
    OPENAI_API_KEY: str = "ollama"  # dummy for local
    OPENAI_MODEL: str = "llama3.2"

    APP_ENV: str = "dev"


settings = Settings()
