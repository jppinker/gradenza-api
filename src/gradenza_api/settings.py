from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Supabase
    supabase_url: str
    supabase_service_role_key: str

    # OpenRouter
    openrouter_api_key: str

    # Redis / ARQ
    redis_url: str = "redis://localhost:6379"

    # Server-to-server auth (shared with Next.js backend)
    internal_api_secret: str

    # CORS — comma-separated list
    allowed_origins: str = "http://localhost:3000"

    log_level: str = "INFO"

    # Storage
    submission_photos_bucket: str = "submission-photos"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()  # type: ignore[call-arg]
