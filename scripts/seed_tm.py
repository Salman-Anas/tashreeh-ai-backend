"""Seed ``translation_memory`` from a CSV (text_en,text_ur,source) with embeddings.

Usage (from backend/):
    python -m scripts.seed_tm [--csv data/tm_pairs.csv] [--batch 50] [--limit N] [--dry-run]

Steps: validate → deduplicate → normalize Urdu → embed both sides in batches
(gemini-embedding-2, 768 dims, retry + backoff) → upsert → summary.
Re-running is safe: rows upsert on (text_en, text_ur).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from collections import Counter
from pathlib import Path

from app.config import BACKEND_DIR
from app.db import DatabaseUnavailable, get_db
from app.services.gemini_client import GeminiClient, GeminiError
from app.services.normalize import normalize_for_matching, normalize_urdu, urdu_ratio
from app.services.tm import embedding_text

MIN_RATIO, MAX_RATIO = 0.4, 2.5  # len(ur) / len(en), in characters
MAX_CHARS = 3000


def validate(en: str, ur: str) -> str | None:
    """Return a rejection reason, or None if the pair looks usable."""
    if not en or not ur:
        return "empty side"
    if len(en) > MAX_CHARS or len(ur) > MAX_CHARS:
        return "too long"
    ratio = len(ur) / len(en)
    if not MIN_RATIO <= ratio <= MAX_RATIO:
        return "length ratio out of range"
    if urdu_ratio(ur) < 0.6:
        return "text_ur is not mostly Urdu"
    if urdu_ratio(en) > 0.2:
        return "text_en contains Urdu script"
    return None


def read_pairs(path: Path) -> tuple[list[dict[str, str | None]], Counter[str]]:
    rejected: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str | None]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"text_en", "text_ur"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            en = " ".join((row.get("text_en") or "").split())
            ur = normalize_urdu(row.get("text_ur") or "", preserve_newlines=False)
            reason = validate(en, ur)
            if reason:
                rejected[reason] += 1
                continue
            key = (en.lower(), normalize_for_matching(ur))
            if key in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"text_en": en, "text_ur": ur, "source": (row.get("source") or "").strip() or None})
    return rows, rejected


async def embed_rows(rows: list[dict[str, str | None]], batch: int) -> None:
    gemini = GeminiClient()
    if not gemini.configured:
        sys.exit("Error: GEMINI_API_KEY is not set in backend/.env")
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        en_vecs = await gemini.embed_documents([embedding_text(r["text_en"] or "", "en") for r in chunk], batch)
        ur_vecs = await gemini.embed_documents([embedding_text(r["text_ur"] or "", "ur") for r in chunk], batch)
        for r, ve, vu in zip(chunk, en_vecs, ur_vecs, strict=True):
            r["embedding_en"] = ve  # type: ignore[assignment]
            r["embedding_ur"] = vu  # type: ignore[assignment]
        print(f"  embedded {min(i + batch, len(rows))}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=BACKEND_DIR / "data" / "tm_pairs.csv")
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="Only seed the first N valid rows")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no embeddings, no writes")
    args = parser.parse_args()

    started = time.perf_counter()
    rows, rejected = read_pairs(args.csv)
    if args.limit:
        rows = rows[: args.limit]

    print(f"Read {args.csv}: {len(rows)} valid pairs, {sum(rejected.values())} rejected")
    for reason, n in rejected.most_common():
        print(f"  rejected ({reason}): {n}")
    if args.dry_run or not rows:
        return

    try:
        db = get_db()
    except DatabaseUnavailable as exc:
        sys.exit(f"Error: {exc}")

    try:
        asyncio.run(embed_rows(rows, args.batch))
    except GeminiError as exc:
        sys.exit(f"Embedding failed: {exc}")

    for i in range(0, len(rows), 100):
        db.table("translation_memory").upsert(rows[i : i + 100], on_conflict="text_en,text_ur").execute()
    total = db.table("translation_memory").select("id", count="exact").limit(1).execute().count

    print("\nTranslation memory seed summary")
    print(f"  valid pairs       : {len(rows)}")
    print(f"  rejected          : {sum(rejected.values())}")
    print(f"  embedded + upserted: {len(rows)}")
    print(f"  total in database : {total}")
    print(f"  took              : {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
