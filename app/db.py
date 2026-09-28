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


MIGRATION_HINT = "The database schema is out of date: run supabase/003_everyday_terms_and_review.sql in the Supabase SQL editor."
_NEW_COLUMNS = ("term_ur_common", "term_corrections", "review_note", "reviewed_at", "status")


def is_outdated_schema(exc: Exception) -> bool:
    """True if ``exc`` is PostgREST/Postgres complaining about a column added by migration 003."""
    msg = str(exc)
    return any(c in msg for c in _NEW_COLUMNS) and ("column" in msg.lower() or "42703" in msg or "PGRST204" in msg)
