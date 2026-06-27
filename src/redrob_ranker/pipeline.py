"""
Pipeline orchestrator — the Stage A -> D funnel described in the architecture
writeup / README.md.

Stage A (full pool, all N candidates, milliseconds-to-seconds):
    - Drop honeypots (honeypot.py)
    - Compute a cheap structural "plausibility" score (title tier + experience
      band + location) to narrow the pool before the (relatively) expensive
      semantic stage. We keep a generous cutoff so we never silently discard a
      genuine Tier-1 candidate whose title happens to be unusual.

Stage B (narrowed pool, up to a few thousand, seconds):
    - Semantic similarity between JD query text and each candidate's narrative
      document (semantic.py), via sentence-transformers or TF-IDF fallback.

Stage C (same narrowed pool):
    - Full hybrid composite scoring (scoring.py) combining semantic_fit with
      all the deterministic feature components, then behavioral modifier.

Stage D (top K only, e.g. top 100):
    - Grounded reasoning text generation (reasoning.py).

Output: a list of dicts ready to be written to the submission CSV by cli.py,
already sorted by final_score descending with the required tie-break by
candidate_id ascending.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np

from . import features as feat
from . import scoring
from .honeypot import honeypot_flags
from .jd_query import JD_QUERY_TEXT
from .reasoning import build_reasoning
from .semantic import candidate_document, get_semantic_backend

logger = logging.getLogger(__name__)

# Stage A cutoff: keep any candidate whose cheap structural prior is at least
# this fraction of the max possible, OR who has zero a-priori title relevance
# but enough years of plausible experience that the JD's "may not use the
# right words" warning could apply -- we intentionally do not require title
# tier > 0, because that is exactly the trap the JD warns about (a Tier 5
# candidate may have an unrelated-sounding title with a strong career history).
# Instead Stage A only removes candidates who are *clearly* out of scope on
# experience and have zero title relevance, which is a much safer cut.
STAGE_A_MIN_PLAUSIBILITY = 0.05


def stage_a_filter(
    candidates: list[dict[str, Any]],
    min_survivors: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Drop honeypots and obviously out-of-scope candidates. Returns (kept, stats).

    Honeypots are ALWAYS dropped, with no exception -- the submission spec
    disqualifies any submission with >10% honeypot rate in the top 100, so
    this is a hard gate, not a tunable preference.

    The plausibility cut, by contrast, is a soft efficiency filter purely to
    keep Stage B/C fast. min_survivors guarantees we never end up with fewer
    than `min_survivors` candidates even in a pathological JD/dataset
    combination -- if the plausibility cut would drop the pool below that
    floor, we back off and keep the highest-plausibility remainder instead of
    risking an under-filled submission (which is an automatic Stage 1
    rejection: "exactly 100 rows" is non-negotiable).
    """
    non_honeypots = []
    n_honeypots = 0
    for c in candidates:
        if honeypot_flags(c):
            n_honeypots += 1
        else:
            non_honeypots.append(c)

    stats = {
        "input": len(candidates),
        "honeypots_dropped": n_honeypots,
        "implausible_dropped": 0,
    }

    scored_for_plausibility = []
    for c in non_honeypots:
        profile = c.get("profile", {})
        title_score = feat.title_tier_score(profile.get("current_title", ""))
        exp_score = feat.experience_fit_score(profile.get("years_of_experience"))
        # Keep if there's ANY plausible relevance signal: decent title tier,
        # OR experience is in-band even with an unrelated-looking title (the
        # JD's own "may not use the right words" caveat -- semantic stage
        # gets the final say on these).
        plausibility = max(title_score, exp_score * 0.3)
        scored_for_plausibility.append((plausibility, c))

    kept = [c for p, c in scored_for_plausibility if p >= STAGE_A_MIN_PLAUSIBILITY]
    stats["implausible_dropped"] = len(non_honeypots) - len(kept)

    if len(kept) < min_survivors:
        # Back off: keep the top `min_survivors` by plausibility instead of a
        # hard threshold, so we never under-fill the required 100 output rows.
        scored_for_plausibility.sort(key=lambda x: -x[0])
        kept = [c for _, c in scored_for_plausibility[:min_survivors]]
        stats["implausible_dropped"] = len(non_honeypots) - len(kept)
        stats["plausibility_backoff_applied"] = True

    stats["kept"] = len(kept)
    return kept, stats


def stage_b_semantic(
    candidates: list[dict[str, Any]],
    backend_name: str | None = None,
) -> tuple[np.ndarray, str]:
    """Compute semantic_fit scores for all candidates in the narrowed pool."""
    backend = get_semantic_backend(force=backend_name)
    documents = [candidate_document(c) for c in candidates]
    scores = backend.score(JD_QUERY_TEXT, documents)
    return scores, backend.name


def stage_c_score(
    candidates: list[dict[str, Any]],
    semantic_scores: np.ndarray,
    today: date,
) -> list[dict[str, Any]]:
    """Full hybrid composite score for every candidate in the narrowed pool."""
    results = []
    for c, sem in zip(candidates, semantic_scores):
        result = scoring.score_candidate(c, semantic_fit=float(sem), today=today)
        results.append(result)
    return results


def stage_d_reasoning(
    candidates_by_id: dict[str, dict[str, Any]],
    top_results: list[dict[str, Any]],
) -> None:
    """Mutate top_results in place, adding a 'reasoning' field for the top K only."""
    for r in top_results:
        candidate = candidates_by_id[r["candidate_id"]]
        r["reasoning"] = build_reasoning(candidate, r)


def run_pipeline(
    candidates: list[dict[str, Any]],
    top_k: int = 100,
    backend_name: str | None = None,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Run the full Stage A->D funnel.

    Returns (top_k_results, run_stats) where top_k_results is sorted by
    final_score descending, candidate_id ascending on ties, each entry having
    candidate_id / final_score / reasoning / full score breakdown.
    """
    today = today or date(2026, 6, 26)
    run_stats: dict[str, Any] = {}

    kept, stage_a_stats = stage_a_filter(candidates, min_survivors=top_k)
    run_stats["stage_a"] = stage_a_stats
    logger.info("Stage A: %s", stage_a_stats)

    semantic_scores, backend_used = stage_b_semantic(kept, backend_name=backend_name)
    run_stats["semantic_backend"] = backend_used
    logger.info("Stage B: semantic backend = %s", backend_used)

    results = stage_c_score(kept, semantic_scores, today=today)
    run_stats["stage_c_scored"] = len(results)

    # Sort using the score rounded to the same precision written to the CSV
    # (4 decimal places). Sorting on the raw float can produce two candidates
    # that *look* tied in the output (same 4dp string) but weren't treated as
    # tied during the sort -- which fails the validator's "ties broken by
    # candidate_id ascending" rule. Rounding before sorting makes the in-memory
    # ordering match the on-disk ordering exactly.
    for r in results:
        r["final_score"] = round(r["final_score"], 4)
    results.sort(key=lambda r: (-r["final_score"], r["candidate_id"]))

    effective_top_k = top_k
    if len(results) < top_k:
        # Only realistic on tiny test fixtures (e.g. the 50-row sample file) --
        # on the real 100K dataset there are always >= top_k honeypot-free
        # candidates. Rather than fabricate filler rows, we output everything
        # available and log loudly: a short file run is for local testing, and
        # silently padding with fake duplicate-looking rows would be a worse
        # bug to ship than an honest short output during dev.
        logger.warning(
            "Only %d candidates survived filtering, fewer than top_k=%d. "
            "Outputting all %d available rows. This is expected on small test "
            "fixtures and should never happen on the full 100K dataset.",
            len(results), top_k, len(results),
        )
        effective_top_k = len(results)

    top_results = results[:effective_top_k]
    candidates_by_id = {c["candidate_id"]: c for c in candidates}
    stage_d_reasoning(candidates_by_id, top_results)
    run_stats["stage_d_reasoned"] = len(top_results)

    return top_results, run_stats
