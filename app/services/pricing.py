"""Gemini cost and client price for Developer API calls.

Prices are Gemini API paid-tier list prices in USD per 1M tokens
(https://ai.google.dev/gemini-api/docs/pricing, checked 2026-09-27). Free-tier
keys are not charged by Google, but we still report the list-price cost so the
margin is realistic. Override with PRICE_*_PER_M in .env if prices change.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

from app.config import Settings
from app.services.usage import UsageMeter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenPrice:
    input_per_m: float
    output_per_m: float  # includes thinking tokens


# model -> [(effective_from, price)], newest first
GENERATION_PRICES: dict[str, list[tuple[date, GenPrice]]] = {
    "gemini-3.8-flash": [
        (date(2027, 1, 1), GenPrice(1.50, 7.50)),
        (date.min, GenPrice(0.75, 3.75)),
    ],
    "gemini-3.5-flash-lite": [(date.min, GenPrice(0.30, 2.50))],
}
EMBED_PRICES: dict[str, float] = {
    "gemini-embedding-2": 0.20,
    "gemini-embedding-2-preview": 0.20,
}
FALLBACK_MODEL = "gemini-3.8-flash"


@dataclass(frozen=True)
class PriceSheet:
    model: str
    embed_model: str
    input_per_m: float
    output_per_m: float
    embed_per_m: float
    markup: float

    def charge(self, per_m: float) -> float:
        return round(per_m * self.markup, 4)


def price_sheet(settings: Settings, today: date | None = None) -> PriceSheet:
    today = today or date.today()
    table = GENERATION_PRICES.get(settings.gemini_model)
    if table is None:
        log.warning("pricing: no price for %s, using %s prices", settings.gemini_model, FALLBACK_MODEL)
        table = GENERATION_PRICES[FALLBACK_MODEL]
    gen = next(p for start, p in table if today >= start)
    return PriceSheet(
        model=settings.gemini_model,
        embed_model=settings.gemini_embed_model,
        input_per_m=settings.price_input_per_m if settings.price_input_per_m is not None else gen.input_per_m,
        output_per_m=settings.price_output_per_m if settings.price_output_per_m is not None else gen.output_per_m,
        embed_per_m=(
            settings.price_embed_per_m
            if settings.price_embed_per_m is not None
            else EMBED_PRICES.get(settings.gemini_embed_model, 0.20)
        ),
        markup=settings.api_markup,
    )


@dataclass(frozen=True)
class Cost:
    gemini_cost_usd: float
    price_usd: float


def _round_up(x: float, places: int = 6) -> float:
    f = 10**places
    return math.ceil(x * f - 1e-9) / f


def compute_cost(usage: UsageMeter, sheet: PriceSheet) -> Cost:
    gemini = (
        usage.input_tokens * sheet.input_per_m
        + usage.output_tokens * sheet.output_per_m
        + usage.embedding_tokens * sheet.embed_per_m
    ) / 1_000_000
    return Cost(gemini_cost_usd=round(gemini, 6), price_usd=_round_up(gemini * sheet.markup))
