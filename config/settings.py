from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application name
    app_name: str = "OpenSourceMultiAgent"

    # requirements
    GITHUB_API_KEY: str
    GEMINI_API_KEY: str
    GRAPHQL_API_ENDPOINT: str = "https://api.github.com/graphql"
    SERPER_API_SECRET: str
    TAVILY_API_SECRET: str
    GEMINI_MODEL: str = "gemini-2.5-flash"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings() # type: ignore