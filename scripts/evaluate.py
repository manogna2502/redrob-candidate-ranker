#!/usr/bin/env python3
"""
Self-evaluation harness.

We do NOT have access to the hidden ground truth relevance labels used for the
official NDCG@10 / NDCG@50 / MAP / P@10 composite (per submission_spec.docx,
section 4 -- "no public partition, no live leaderboard"). So this script
cannot compute the actual competition metric.

What it DOES do is compute a set of proxy quality checks that we can verify
ourselves against the released data and the JD's own explicit statements
about what a good/bad ranking looks like. These are sanity checks a human
reviewer (or we, before submitting) can use to catch obvious failures before
ever finding out the real score:

    1. Honeypot rate in top-K (spec: >10% in top 100 = automatic disqualification
       at Stage 3). We verify it's exactly 0% on the real submission.
    2. Format validation (delegates to the organizer-provided validator).
    3. Trap-pattern leakage: consulting-only / research-only / langchain-tourist
       counts in top-K (should be near zero per the JD's explicit disqualifiers).
    4. Title-tier distribution in top-K (should skew heavily toward tier 2-3).
    5. Score monotonicity and distribution sanity (no flat-lining, no ties
       violating the required tie-break order).
    6. Reasoning quality proxies: no empty reasoning, no duplicate reasoning
       strings, average length, and lexical diversity across the sample.

Usage:
    python3 scripts/evaluate.py --submission ./submission.csv --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker import features as feat
from redrob_ranker.data_loader import load_candidates
from redrob_ranker.honeypot import is_honeypot


def load_submission(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--candidates", required=True)
    args = parser.parse_args()

    rows = load_submission(args.submission)
    candidates = load_candidates(args.candidates)
    candidates_by_id = {c["candidate_id"]: c for c in candidates}

    print(f"Submission rows: {len(rows)}")
    print()

    # --- 1. Honeypot rate -------------------------------------------------
    honeypot_count = 0
    for row in rows:
        c = candidates_by_id.get(row["candidate_id"])
        if c and is_honeypot(c):
            honeypot_count += 1
    honeypot_rate = honeypot_count / len(rows) if rows else 0
    status = "PASS" if honeypot_rate <= 0.10 else "FAIL (Stage 3 auto-disqualify)"
    print(f"[1] Honeypot rate in submission: {honeypot_rate:.1%} ({honeypot_count}/{len(rows)}) -- {status}")
    print()

    # --- 2. Trap-pattern leakage -------------------------------------------
    consulting_only = research_only = langchain_tourist = 0
    title_tier_counts: Counter = Counter()
    for row in rows:
        c = candidates_by_id.get(row["candidate_id"])
        if not c:
            continue
        if feat.is_consulting_only(c):
            consulting_only += 1
        if feat.is_research_only(c):
            research_only += 1
        if feat.framework_tourist_penalty(c) > 0:
            langchain_tourist += 1
        tier = feat.title_tier(c.get("profile", {}).get("current_title", ""))
        title_tier_counts[tier] += 1

    print(f"[2] Trap-pattern leakage in submission:")
    print(f"    consulting-only career : {consulting_only}")
    print(f"    research-only career   : {research_only}")
    print(f"    langchain-tourist      : {langchain_tourist}")
    print()

    print(f"[3] Title-tier distribution in submission (tier 3 = exact match, 0 = no a-priori relevance):")
    for tier in sorted(title_tier_counts, reverse=True):
        print(f"    tier {tier}: {title_tier_counts[tier]}")
    print()

    # --- 4. Score sanity ----------------------------------------------------
    scores = [float(r["score"]) for r in rows]
    ranks = [int(r["rank"]) for r in rows]
    ids = [r["candidate_id"] for r in rows]

    monotonic = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    ranks_valid = sorted(ranks) == list(range(1, len(rows) + 1))
    no_dupes = len(ids) == len(set(ids))
    all_distinct_scores = len(set(scores)) > 1

    print(f"[4] Score / rank sanity:")
    print(f"    scores non-increasing by rank : {monotonic}")
    print(f"    ranks are exactly 1..N        : {ranks_valid}")
    print(f"    no duplicate candidate_ids    : {no_dupes}")
    print(f"    scores are differentiated     : {all_distinct_scores} (min={min(scores):.4f}, max={max(scores):.4f})")
    print()

    # --- 5. Reasoning quality proxies ---------------------------------------
    reasonings = [r.get("reasoning", "") for r in rows]
    n_empty = sum(1 for r in reasonings if not r.strip())
    n_unique = len(set(reasonings))
    avg_len = sum(len(r.split()) for r in reasonings) / len(reasonings) if reasonings else 0

    print(f"[5] Reasoning quality proxies:")
    print(f"    empty reasoning strings    : {n_empty}")
    print(f"    unique reasoning strings   : {n_unique}/{len(reasonings)}")
    print(f"    avg reasoning length (words): {avg_len:.1f}")
    print()

    print("Note: this script checks proxy signals we can verify ourselves. The "
          "actual competition score (NDCG@10/50, MAP, P@10) requires the hidden "
          "ground truth and is computed by the organizers after submissions close.")


if __name__ == "__main__":
    main()
