from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application name
    app_name: str = "OpenSourceMultiAgent"

    # requirements
    GITHUB_API_KEY: str
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() # type: ignore