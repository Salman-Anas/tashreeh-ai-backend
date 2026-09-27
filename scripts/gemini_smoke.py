"""Manual smoke test for Gemini (and, if configured, the TM vector search).

Usage (from backend/):
    python -m scripts.gemini_smoke ["The accused was granted bail."]
"""

from __future__ import annotations

import asyncio
import math
import sys

from app.config import get_settings
from app.db import db_configured
from app.models import LLMTranslation
from app.services.gemini_client import GeminiClient
from app.services.pipeline import SYSTEM_INSTRUCTION, build_prompt
from app.services.tm import SupabaseTM


async def main(text: str) -> None:
    s = get_settings()
    g = GeminiClient()
    print(f"model={s.gemini_model} embed_model={s.gemini_embed_model}\n")

    q = await g.embed_query(text)
    d1, d2 = await g.embed_documents(["The accused was granted bail.", "Weather is nice today."])
    norm = math.sqrt(sum(x * x for x in q))
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731 - vectors are unit-norm
    print(f"embedding dims={len(q)} norm={norm:.3f}")
    print(f"  cos(query, legal doc)   = {cos(q, d1):.3f}")
    print(f"  cos(query, weather doc) = {cos(q, d2):.3f}\n")

    out = await g.generate_structured(
        system_instruction=SYSTEM_INSTRUCTION,
        contents=build_prompt(text, "en-ur", [], []),
        schema=LLMTranslation,
    )
    print("translation:", out.translation)
    print("notes:", out.notes, "\n")

    if db_configured():
        examples = await SupabaseTM(g).retrieve(text, "en-ur")
        print(f"TM matches ({len(examples)}):")
        for ex in examples:
            print(f"  {ex.similarity:.3f}  {ex.text_en}  |  {ex.text_ur}")
    else:
        print("Supabase not configured; skipping TM search.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "The accused was granted bail by the High Court."))
