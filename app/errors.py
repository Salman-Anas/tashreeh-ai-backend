"""Consistent JSON errors: ``{"error": {"code": "...", "message": "..."}}``."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from postgrest.exceptions import APIError as PostgrestAPIError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.db import DatabaseUnavailable
from app.services.documents import DocumentError
from app.services.gemini_client import GeminiError, GeminiRateLimitError, GeminiTimeoutError

log = logging.getLogger(__name__)


class AppError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _json(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return _json(exc.status, exc.code, exc.message)

    @app.exception_handler(GeminiRateLimitError)
    async def _gemini_rate(_: Request, exc: GeminiRateLimitError) -> JSONResponse:
        return _json(429, "gemini_rate_limited", str(exc))

    @app.exception_handler(GeminiTimeoutError)
    async def _gemini_timeout(_: Request, exc: GeminiTimeoutError) -> JSONResponse:
        return _json(504, "gemini_timeout", str(exc))

    @app.exception_handler(GeminiError)
    async def _gemini(_: Request, exc: GeminiError) -> JSONResponse:
        log.error("gemini error: %s", exc)
        return _json(502, "gemini_error", str(exc))

    @app.exception_handler(DatabaseUnavailable)
    async def _db(_: Request, exc: DatabaseUnavailable) -> JSONResponse:
        log.error("database unavailable: %s", exc)
        return _json(503, "database_unavailable", "The database is unavailable right now. Please try again later.")

    @app.exception_handler(DocumentError)
    async def _doc(_: Request, exc: DocumentError) -> JSONResponse:
        return _json(422, "document_error", str(exc))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
        msg = first.get("msg", "Invalid request")
        return _json(422, "invalid_request", f"{loc}: {msg}" if loc else msg)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _json(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(PostgrestAPIError)
    async def _postgrest(_: Request, exc: PostgrestAPIError) -> JSONResponse:
        if exc.code == "PGRST205":  # table/view missing from the schema
            log.error("database table missing: %s", exc.message)
            return _json(
                503,
                "database_not_migrated",
                "A database table is missing. Run supabase/002_developer_api.sql in the Supabase SQL editor.",
            )
        log.error("database error: %s", exc)
        return _json(503, "database_error", "The database rejected the request. Please try again later.")

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _json(500, "internal_error", "Something went wrong on the server.")
