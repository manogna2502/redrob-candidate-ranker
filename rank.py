#!/usr/bin/env python3
"""
CLI entrypoint for producing the submission CSV.

Exact reproduction command (per submission_spec.docx section 10.3):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Runtime/compute budget (enforced by the grading sandbox, not by this script):
    - <= 5 minutes wall-clock
    - <= 16 GB RAM
    - CPU only, no GPU
    - No network access during the ranking step itself
    - <= 5 GB intermediate disk state

This script does not download anything at run time. If sentence-transformers
and a cached model checkpoint are available, it uses them; otherwise it falls
back to a TF-IDF semantic backend automatically (see semantic.py). Either way,
no network call is made here -- model downloading, if you want the SBERT path,
must happen ahead of time via scripts/precompute_embeddings.py or simply by
having the model cached in the standard sentence-transformers cache dir before
the timed run starts.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

# Allow running as `python rank.py` from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from redrob_ranker.data_loader import load_candidates  # noqa: E402
from redrob_ranker.pipeline import run_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rank")

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob hackathon JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl (or sample_candidates.json)")
    parser.add_argument("--out", required=True, help="Path to write the submission CSV")
    parser.add_argument("--top-k", type=int, default=100, help="Number of ranked rows to output (spec requires 100)")
    parser.add_argument(
        "--backend", choices=["sbert", "tfidf"], default=None,
        help="Force a specific semantic backend; default auto-tries sbert then falls back to tfidf",
    )
    return parser.parse_args()


def write_submission_csv(results: list[dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REQUIRED_HEADER)
        for rank, r in enumerate(results, start=1):
            writer.writerow([
                r["candidate_id"],
                rank,
                f"{r['final_score']:.4f}",
                r.get("reasoning", ""),
            ])


def main() -> None:
    args = parse_args()
    t0 = time.time()

    logger.info("Loading candidates from %s ...", args.candidates)
    candidates = load_candidates(args.candidates)
    logger.info("Loaded %d candidates in %.1fs", len(candidates), time.time() - t0)

    t1 = time.time()
    results, stats = run_pipeline(candidates, top_k=args.top_k, backend_name=args.backend)
    logger.info("Pipeline complete in %.1fs. Stats: %s", time.time() - t1, stats)

    write_submission_csv(results, args.out)
    logger.info("Wrote %d ranked rows to %s", len(results), args.out)
    logger.info("Total wall-clock time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
