"""Developer API keys and per-call usage records.

Keys look like ``tsh_live_<32 random chars>``. Only a SHA-256 hash and a short
display prefix are stored; the full key is shown once, at creation.

``SupabaseKeyStore`` is used when Supabase is configured; otherwise an
in-memory store keeps things working for local development (keys and usage are
lost on restart). All methods are synchronous — call them via ``asyncio.to_thread``.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from app.db import db_configured, get_db

log = logging.getLogger(__name__)

KEY_PREFIX = "tsh_live_"
DISPLAY_PREFIX_LEN = len(KEY_PREFIX) + 4

STAT_FIELDS = ("calls", "input_tokens", "output_tokens", "embedding_tokens", "total_tokens", "gemini_cost_usd", "billed_usd")


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(24)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def looks_like_key(key: str) -> bool:
    return key.startswith(KEY_PREFIX) and 20 <= len(key) <= 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_stats() -> dict[str, float]:
    return {f: 0 for f in STAT_FIELDS}


class KeyStore(Protocol):
    persistent: bool

    def create(self, name: str, owner_session: str) -> tuple[dict[str, Any], str]: ...
    def list(self, owner_session: str) -> list[dict[str, Any]]: ...
    def revoke(self, key_id: str, owner_session: str) -> bool: ...
    def find_active(self, key_hash: str) -> dict[str, Any] | None: ...
    def touch(self, key_id: str) -> None: ...
    def record_usage(self, row: dict[str, Any]) -> None: ...
    def usage(self, owner_session: str, key_id: str | None, limit: int) -> list[dict[str, Any]]: ...
    def stats(self, key_ids: list[str]) -> dict[str, dict[str, float]]: ...


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
class SupabaseKeyStore:
    persistent = True
    _key_cols = "id,name,key_prefix,owner_session,created_at,last_used_at,revoked_at"

    def create(self, name: str, owner_session: str) -> tuple[dict[str, Any], str]:
        key = generate_key()
        row = {"name": name, "key_prefix": key[:DISPLAY_PREFIX_LEN], "key_hash": hash_key(key), "owner_session": owner_session}
        res = get_db().table("api_keys").insert(row).execute()
        created = {k: v for k, v in res.data[0].items() if k != "key_hash"}
        return created, key

    def list(self, owner_session: str) -> list[dict[str, Any]]:
        res = (
            get_db().table("api_keys").select(self._key_cols)
            .eq("owner_session", owner_session).order("created_at", desc=True).execute()
        )
        return list(res.data or [])

    def revoke(self, key_id: str, owner_session: str) -> bool:
        res = (
            get_db().table("api_keys").update({"revoked_at": _now()})
            .eq("id", key_id).eq("owner_session", owner_session).is_("revoked_at", "null").execute()
        )
        return bool(res.data)

    def find_active(self, key_hash: str) -> dict[str, Any] | None:
        res = (
            get_db().table("api_keys").select(self._key_cols)
            .eq("key_hash", key_hash).is_("revoked_at", "null").limit(1).execute()
        )
        return res.data[0] if res.data else None

    def touch(self, key_id: str) -> None:
        get_db().table("api_keys").update({"last_used_at": _now()}).eq("id", key_id).execute()

    def record_usage(self, row: dict[str, Any]) -> None:
        get_db().table("api_usage").insert(row).execute()

    def usage(self, owner_session: str, key_id: str | None, limit: int) -> list[dict[str, Any]]:
        ids = [k["id"] for k in self.list(owner_session)]
        if key_id:
            ids = [i for i in ids if i == key_id]
        if not ids:
            return []
        res = (
            get_db().table("api_usage").select("*, api_keys(name,key_prefix)")
            .in_("api_key_id", ids).order("created_at", desc=True).limit(limit).execute()
        )
        rows = []
        for r in res.data or []:
            key = r.pop("api_keys", None) or {}
            rows.append({**r, "key_name": key.get("name"), "key_prefix": key.get("key_prefix")})
        return rows

    def stats(self, key_ids: list[str]) -> dict[str, dict[str, float]]:
        if not key_ids:
            return {}
        res = get_db().table("api_key_stats").select("*").in_("api_key_id", key_ids).execute()
        return {r["api_key_id"]: {f: float(r.get(f) or 0) for f in STAT_FIELDS} for r in res.data or []}


# ---------------------------------------------------------------------------
# In-memory (development / tests)
# ---------------------------------------------------------------------------
class MemoryKeyStore:
    persistent = False

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.keys: dict[str, dict[str, Any]] = {}
        self.hashes: dict[str, str] = {}
        self.rows: list[dict[str, Any]] = []

    def create(self, name: str, owner_session: str) -> tuple[dict[str, Any], str]:
        key = generate_key()
        rec = {
            "id": str(uuid.uuid4()), "name": name, "key_prefix": key[:DISPLAY_PREFIX_LEN],
            "owner_session": owner_session, "created_at": _now(), "last_used_at": None, "revoked_at": None,
        }
        with self._lock:
            self.keys[rec["id"]] = rec
            self.hashes[hash_key(key)] = rec["id"]
        return dict(rec), key

    def list(self, owner_session: str) -> list[dict[str, Any]]:
        with self._lock:
            recs = [dict(r) for r in self.keys.values() if r["owner_session"] == owner_session]
        return sorted(recs, key=lambda r: r["created_at"], reverse=True)

    def revoke(self, key_id: str, owner_session: str) -> bool:
        with self._lock:
            rec = self.keys.get(key_id)
            if not rec or rec["owner_session"] != owner_session or rec["revoked_at"]:
                return False
            rec["revoked_at"] = _now()
            return True

    def find_active(self, key_hash: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self.keys.get(self.hashes.get(key_hash, ""))
            return dict(rec) if rec and not rec["revoked_at"] else None

    def touch(self, key_id: str) -> None:
        with self._lock:
            if key_id in self.keys:
                self.keys[key_id]["last_used_at"] = _now()

    def record_usage(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows.append({"id": len(self.rows) + 1, "created_at": _now(), **row})

    def usage(self, owner_session: str, key_id: str | None, limit: int) -> list[dict[str, Any]]:
        owned = {k["id"]: k for k in self.list(owner_session)}
        with self._lock:
            rows = [r for r in self.rows if r["api_key_id"] in owned and (not key_id or r["api_key_id"] == key_id)]
        rows = sorted(rows, key=lambda r: r["id"], reverse=True)[:limit]
        return [
            {**r, "key_name": owned[r["api_key_id"]]["name"], "key_prefix": owned[r["api_key_id"]]["key_prefix"]}
            for r in rows
        ]

    def stats(self, key_ids: list[str]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            for r in self.rows:
                if r["api_key_id"] not in key_ids:
                    continue
                s = out.setdefault(r["api_key_id"], empty_stats())
                s["calls"] += 1
                for f in STAT_FIELDS[1:]:
                    s[f] += float(r.get(f) or 0)
        return out


# ---------------------------------------------------------------------------
# Access
# ---------------------------------------------------------------------------
_store: KeyStore | None = None
_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_CACHE_TTL_S = 30.0


def get_key_store() -> KeyStore:
    global _store
    if _store is None:
        if db_configured():
            _store = SupabaseKeyStore()
        else:
            log.warning("api keys: Supabase not configured; using in-memory store (lost on restart)")
            _store = MemoryKeyStore()
    return _store


def set_key_store(store: KeyStore | None) -> None:
    """Replace the store (tests)."""
    global _store
    _store = store
    _cache.clear()


def lookup_key(plaintext: str) -> dict[str, Any] | None:
    """Find an active key by its plaintext, with a short cache to spare the database."""
    h = hash_key(plaintext)
    hit = _cache.get(h)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    rec = get_key_store().find_active(h)
    _cache[h] = (time.monotonic() + _CACHE_TTL_S, rec)
    return rec


def forget_key(key_id: str) -> None:
    """Drop cached lookups for a key (after revocation)."""
    for h, (_, rec) in list(_cache.items()):
        if rec and rec.get("id") == key_id:
            _cache.pop(h, None)
