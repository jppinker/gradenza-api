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
    tutor_video_bucket: str = "tutor-videos"

    # Tutor video generation
    openrouter_video_prompt_model: str = "openai/gpt-5.5"
    openrouter_video_prompt_fallback_model: str = "google/gemini-3-flash-preview"

    # ── Online lesson AI models ────────────────────────────────────────────────
    # Chat and question generation use GPT-5.5 via OpenRouter.
    # Plan, revise, and homework remain on Gemini 2.5 Flash.
    # Override any value via its corresponding environment variable (see .env.example).
    # Rollback: set ONLINE_LESSON_CHAT_MODEL / ONLINE_LESSON_QUESTIONS_MODEL to
    #   google/gemini-2.5-flash
    online_lesson_chat_model: str = "openai/gpt-5.5"
    online_lesson_plan_model: str = "google/gemini-2.5-flash"
    online_lesson_revise_model: str = "google/gemini-2.5-flash"
    online_lesson_questions_model: str = "openai/gpt-5.5"
    online_lesson_homework_model: str = "google/gemini-2.5-flash"

    @property
    def origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()  # type: ignore[call-arg]
