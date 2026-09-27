"""Benchmark Tashreeh AI against Google Translate.

Input: eval/test_set.csv with columns ``source,reference,direction,google_output``
(paste Google Translate outputs by hand; leave google_output empty to skip it).

For each row we run our full pipeline, then compute for both systems:
  * chrF++  (sacrebleu CHRF, word_order=2) — sentence-level and corpus-level
  * Terminology accuracy — % of glossary terms found in the source whose correct
    target term (or an accepted alternative) appears in the output

Urdu hypotheses and references are matching-normalized the same way for both
systems (Arabic → Urdu letter forms, no diacritics) so neither is penalised for
encoding differences.

Usage (from backend/):
    python -m eval.benchmark [--csv eval/test_set.csv] [--limit N] [--concurrency 2]
Outputs: eval/results.csv and eval/summary.md
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sacrebleu.metrics import CHRF

from app.config import BACKEND_DIR, get_settings
from app.deps import get_pipeline, glossary, load_glossary
from app.models import Direction, RequiredTerm
from app.services.gemini_client import GeminiError
from app.services.normalize import normalize_for_matching
from app.services.term_check import check_terms

EVAL_DIR = BACKEND_DIR / "eval"
chrf = CHRF(word_order=2)


@dataclass
class Row:
    source: str
    reference: str
    direction: Direction
    google: str
    ours: str = ""
    error: str = ""
    required: list[RequiredTerm] | None = None


def prep(text: str, direction: Direction) -> str:
    return normalize_for_matching(text) if direction == "en-ur" else " ".join(text.split())


def sentence_chrf(hyp: str, ref: str, direction: Direction) -> float | None:
    if not hyp:
        return None
    return chrf.sentence_score(prep(hyp, direction), [prep(ref, direction)]).score


def corpus_chrf(hyps: list[str], refs: list[str], direction: Direction) -> float | None:
    if not hyps:
        return None
    return chrf.corpus_score([prep(h, direction) for h in hyps], [[prep(r, direction) for r in refs]]).score


def term_counts(output: str, required: list[RequiredTerm], direction: Direction) -> tuple[int, int]:
    if not required or not output:
        return 0, len(required)
    res = check_terms(output, required, direction)
    return len(res.terms_found), len(required)


def read_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            direction = (r.get("direction") or "").strip()
            if direction not in ("en-ur", "ur-en"):
                print(f"  line {i}: skipped (direction must be en-ur or ur-en)")
                continue
            src, ref = (r.get("source") or "").strip(), (r.get("reference") or "").strip()
            if not src or not ref:
                print(f"  line {i}: skipped (empty source or reference)")
                continue
            rows.append(Row(src, ref, direction, (r.get("google_output") or "").strip()))  # type: ignore[arg-type]
    return rows


async def run_ours(rows: list[Row], concurrency: int) -> None:
    pipeline = get_pipeline()
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(row: Row) -> None:
        nonlocal done
        async with sem:
            try:
                res = await pipeline.translate(row.source, row.direction)
                row.ours = res.translation
            except GeminiError as exc:
                row.error = str(exc)
        done += 1
        print(f"  translated {done}/{len(rows)}", end="\r", flush=True)

    await asyncio.gather(*(one(r) for r in rows))
    print()


def fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "—"
    return f"{x:.1f}%" if pct else f"{x:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=EVAL_DIR / "test_set.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(f"{args.csv} not found. Copy eval/test_set.example.csv to eval/test_set.csv and fill it in.")
    if not get_settings().gemini_api_key:
        sys.exit("GEMINI_API_KEY is not set in backend/.env")

    load_glossary()
    rows = read_rows(args.csv)
    if args.limit:
        rows = rows[: args.limit]
    print(f"Glossary: {len(glossary)} terms ({glossary.source}). Test rows: {len(rows)}")

    started = time.perf_counter()
    asyncio.run(run_ours(rows, args.concurrency))

    for row in rows:
        src_lang = "en" if row.direction == "en-ur" else "ur"
        row.required = glossary.required_terms(glossary.match(row.source, src_lang), row.direction)

    # -- per-row results ---------------------------------------------------
    out_rows = []
    for row in rows:
        assert row.required is not None
        of, ot = term_counts(row.ours, row.required, row.direction)
        gf, gt = term_counts(row.google, row.required, row.direction)
        out_rows.append(
            {
                "direction": row.direction,
                "source": row.source,
                "reference": row.reference,
                "ours": row.ours,
                "google": row.google,
                "chrf_ours": fmt(sentence_chrf(row.ours, row.reference, row.direction)),
                "chrf_google": fmt(sentence_chrf(row.google, row.reference, row.direction)),
                "terms_required": ot,
                "terms_ours": of,
                "terms_google": gf if row.google else "",
                "error": row.error,
            }
        )
    results_csv = EVAL_DIR / "results.csv"
    with results_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else ["direction"])
        writer.writeheader()
        writer.writerows(out_rows)

    # -- aggregate ---------------------------------------------------------
    table: list[tuple[str, int, str, str, str, str]] = []
    for label, dirs in (("EN → UR", ["en-ur"]), ("UR → EN", ["ur-en"]), ("Overall", ["en-ur", "ur-en"])):
        subset = [r for r in rows if r.direction in dirs]
        if not subset:
            continue
        # chrF++ is computed per direction; "Overall" is the row-weighted mean of directions.
        def chrf_for(get) -> float | None:  # type: ignore[no-untyped-def]
            scores, weights = [], []
            for d in dirs:
                pairs = [(get(r), r.reference) for r in subset if r.direction == d and get(r)]
                if pairs:
                    s = corpus_chrf([p[0] for p in pairs], [p[1] for p in pairs], d)  # type: ignore[arg-type]
                    if s is not None:
                        scores.append(s * len(pairs))
                        weights.append(len(pairs))
            return sum(scores) / sum(weights) if weights else None

        def term_acc(get) -> float | None:  # type: ignore[no-untyped-def]
            found = total = 0
            for r in subset:
                if not get(r):
                    continue
                f_, t_ = term_counts(get(r), r.required or [], r.direction)
                found, total = found + f_, total + t_
            return 100 * found / total if total else None

        table.append(
            (
                label,
                len(subset),
                fmt(chrf_for(lambda r: r.ours)),
                fmt(chrf_for(lambda r: r.google)),
                fmt(term_acc(lambda r: r.ours), pct=True),
                fmt(term_acc(lambda r: r.google), pct=True),
            )
        )

    header = ("Direction", "N", "chrF++ (Tashreeh AI)", "chrF++ (Google)", "Term acc. (Tashreeh AI)", "Term acc. (Google)")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *table)]
    line = lambda cells: "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    print()
    print(line(header))
    print(line(["-" * w for w in widths]))
    for t in table:
        print(line(t))

    errors = sum(1 for r in rows if r.error)
    md = [
        "## Benchmark: Tashreeh AI vs Google Translate",
        "",
        f"Test set: {len(rows)} Pakistani legal sentences · model `{get_settings().gemini_model}` · "
        f"glossary {len(glossary)} terms · run {time.strftime('%Y-%m-%d')}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
        *["| " + " | ".join(str(c) for c in t) + " |" for t in table],
        "",
        "chrF++ = sacrebleu CHRF with word_order=2 (higher is better). Terminology accuracy = share of glossary "
        "terms in the source whose required target term appears in the output.",
    ]
    if errors:
        md.append(f"\n{errors} row(s) failed to translate and were excluded from our scores.")
    (EVAL_DIR / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"\nSaved {results_csv} and {EVAL_DIR / 'summary.md'} ({time.perf_counter() - started:.0f}s)")
    if errors:
        print(f"Warning: {errors} row(s) failed; see the 'error' column in results.csv")


if __name__ == "__main__":
    main()
