"""Seed ``glossary_terms`` from a CSV (term_en,term_ur,term_ur_common,category,notes,source).

``term_ur`` is the official legal Urdu; ``term_ur_common`` (optional) is the
everyday Urdu that translations use instead, e.g. FIR → ایف آئی آر.

Usage (from backend/):
    python -m scripts.seed_glossary [--csv data/glossary.csv] [--dry-run] [--overwrite-common]

Re-running is safe: rows upsert on (term_en, term_ur). An everyday form already
in the database (e.g. approved by a reviewer) is kept unless --overwrite-common
is given. After seeding a running server, call POST /api/glossary/reload.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from app.config import BACKEND_DIR
from app.db import DatabaseUnavailable, get_db
from app.services.normalize import normalize_for_matching, normalize_urdu, urdu_ratio


def read_rows(path: Path) -> tuple[list[dict[str, str | None]], list[str]]:
    rows: list[dict[str, str | None]] = []
    problems: list[str] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"term_en", "term_ur"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV is missing columns: {', '.join(sorted(missing))}")
        for line_no, row in enumerate(reader, start=2):
            en = " ".join((row.get("term_en") or "").split())
            ur = normalize_urdu(row.get("term_ur") or "", preserve_newlines=False)
            if not en or not ur:
                problems.append(f"line {line_no}: empty term_en or term_ur")
                continue
            if urdu_ratio(ur) < 0.8:
                problems.append(f"line {line_no}: term_ur does not look like Urdu: {ur!r}")
                continue
            if urdu_ratio(en) > 0.2:
                problems.append(f"line {line_no}: term_en contains Urdu script: {en!r}")
                continue
            common = normalize_urdu(row.get("term_ur_common") or "", preserve_newlines=False) or None
            if common and urdu_ratio(common) < 0.8:
                problems.append(f"line {line_no}: term_ur_common does not look like Urdu: {common!r}")
                continue
            if common and normalize_for_matching(common) == normalize_for_matching(ur):
                common = None
            key = (en, ur)
            if key in seen:
                problems.append(f"line {line_no}: duplicate of an earlier row ({en} / {ur})")
                continue
            seen.add(key)
            rows.append(
                {
                    "term_en": en,
                    "term_ur": ur,
                    "term_ur_common": common,
                    "category": (row.get("category") or "").strip().lower() or None,
                    "notes": (row.get("notes") or "").strip() or None,
                    "source": (row.get("source") or "").strip() or None,
                }
            )
    return rows, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=BACKEND_DIR / "data" / "glossary.csv")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not write to Supabase")
    parser.add_argument(
        "--overwrite-common", action="store_true", help="Replace everyday Urdu forms already in the database with the CSV's"
    )
    args = parser.parse_args()

    rows, problems = read_rows(args.csv)
    for p in problems:
        print(f"  skipped — {p}")

    if args.dry_run:
        print(f"\nDry run: {len(rows)} valid rows, {len(problems)} skipped.")
        return

    kept = 0
    try:
        db = get_db()
        if not args.overwrite_common:
            existing = {
                (r["term_en"], r["term_ur"]): r["term_ur_common"]
                for r in db.table("glossary_terms").select("term_en,term_ur,term_ur_common").limit(10000).execute().data or []
                if r.get("term_ur_common")
            }
            for r in rows:
                current = existing.get((r["term_en"], r["term_ur"]))  # type: ignore[arg-type]
                if current and current != r["term_ur_common"]:
                    r["term_ur_common"] = current
                    kept += 1
        for i in range(0, len(rows), 200):
            db.table("glossary_terms").upsert(rows[i : i + 200], on_conflict="term_en,term_ur").execute()
        total = db.table("glossary_terms").select("id", count="exact").limit(1).execute().count
    except DatabaseUnavailable as exc:
        sys.exit(f"Error: {exc}")

    print("\nGlossary seed summary")
    print(f"  CSV file          : {args.csv}")
    print(f"  valid rows        : {len(rows)}")
    print(f"  skipped rows      : {len(problems)}")
    print(f"  upserted          : {len(rows)}")
    print(f"  everyday forms kept from database: {kept}")
    print(f"  total in database : {total}")
    print("\nIf the API is running, reload it with: POST /api/glossary/reload")


if __name__ == "__main__":
    main()
