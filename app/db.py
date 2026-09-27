"""Supabase client (service role — server side only)."""

from __future__ import annotations

import logging
from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings

log = logging.getLogger(__name__)


class DatabaseUnavailable(Exception):
    pass


@lru_cache
def _client() -> Client:
    s = get_settings()
    return create_client(s.supabase_url, s.supabase_service_role_key)


def get_db() -> Client:
    if not get_settings().supabase_configured:
        raise DatabaseUnavailable("Supabase is not configured (set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY).")
    try:
        return _client()
    except Exception as exc:  # noqa: BLE001
        raise DatabaseUnavailable(f"Could not create Supabase client: {exc}") from exc


def db_configured() -> bool:
    return get_settings().supabase_configured
