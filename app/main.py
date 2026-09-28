from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from app.config import get_settings
from app.db import db_configured
from app.deps import glossary, load_glossary
from app.errors import register_error_handlers
from app.models import HealthResponse
from app.routers import developer, explain, feedback, glossary as glossary_router, history, public_api, review, translate
from app.services.gemini_client import get_gemini

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

PUBLIC_PREFIX = "/v1"


class ScopedCORS:
    """CORS for the website's origins everywhere, except the public API which any origin may call.

    ``/v1`` is authenticated by API key (no cookies), so allowing all origins is safe;
    the rest of the API is only for the Tashreeh AI website.
    """

    def __init__(self, app: ASGIApp, site_origins: list[str]) -> None:
        self.site = CORSMiddleware(
            app, allow_origins=site_origins, allow_credentials=False, allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["*"]
        )
        self.public = CORSMiddleware(
            app, allow_origins=["*"], allow_credentials=False, allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key"],
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith(PUBLIC_PREFIX):
            await self.public(scope, receive, send)
        else:
            await self.site(scope, receive, send)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_glossary()
    yield


settings = get_settings()
app = FastAPI(
    title=f"{settings.app_name} API",
    description="English ⇄ Urdu legal translation for Pakistani law. The public, API-key authenticated endpoints are under /v1.",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(ScopedCORS, site_origins=settings.cors_origin_list)
register_error_handlers(app)

app.include_router(translate.router)
app.include_router(explain.router)
app.include_router(glossary_router.router)
app.include_router(history.router)
app.include_router(feedback.router)
app.include_router(review.router)
app.include_router(developer.router)
app.include_router(public_api.router)


@app.get("/api/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    gemini = get_gemini()
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        model=settings.gemini_model,
        embed_model=settings.gemini_embed_model,
        supabase=db_configured(),
        gemini=gemini.configured,
        gemini_keys=len(gemini.keys),
        glossary_terms=len(glossary),
        glossary_source="supabase" if glossary.source == "supabase" else "csv",
    )
