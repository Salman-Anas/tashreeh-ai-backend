from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Tashreeh AI"

    # Keys are tried in order; if one is rate-limited or rejected the next is used.
    gemini_api_key: str = ""
    gemini_api_key_2: str = ""
    gemini_api_keys: str = ""  # optional extra keys, comma-separated
    gemini_model: str = "gemini-3.8-flash"
    gemini_embed_model: str = "gemini-embedding-2"
    gemini_timeout_s: float = 60.0
    embed_dim: int = 768
    temperature: float = 0.2

    # Secrets belong in backend/.env, never in this file.
    supabase_url: str = ""
    supabase_service_role_key: str = ""

    # Developer API billing. Prices are USD per 1M tokens; leave unset to use the
    # built-in Gemini price table (app/services/pricing.py).
    api_markup: float = 1.25  # clients pay Gemini cost x this
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None
    price_embed_per_m: float | None = None
    api_rate_limit_per_min: int = 60

    cors_origins: str = "http://localhost:5173"
    max_input_chars: int = 5000
    max_document_chars: int = 60000
    max_upload_bytes: int = 10 * 1024 * 1024

    tm_match_count: int = 5
    tm_min_similarity: float = 0.6
    rate_limit_per_min: int = 20
    translate_concurrency: int = 4

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def gemini_keys(self) -> list[str]:
        keys = [self.gemini_api_key, self.gemini_api_key_2, *self.gemini_api_keys.split(",")]
        out: list[str] = []
        for k in (k.strip() for k in keys):
            if k and k not in out:
                out.append(k)
        return out

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
