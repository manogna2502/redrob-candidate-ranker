"""
Hybrid composite scoring.

final_score = clamp01(
    base_fit_score             [weighted sum of components below]
) * behavioral_modifier        [multiplicative, see features.behavioral_modifier]

base_fit_score components (weights sum to 1.0):
    semantic_fit          0.32   -- JD<->candidate narrative similarity (Stage B)
    skills_match          0.28   -- must-have coverage, trust-adjusted vs Redrob assessment
    title_relevance       0.16   -- title tier (0-3), cheap but reliable prior
    experience_fit        0.08   -- soft band around 5-9 yrs
    location_fit          0.08   -- Pune/Noida > other India > non-India
    notice_period_fit     0.04   -- sub-30-day preferred
    career_trajectory     0.04   -- inverse of penalty (consulting-only, title-chasing, etc.)

Honeypots are excluded entirely upstream (Stage A) -- they never reach this
scorer. Career-trajectory disqualifiers (consulting-only, research-only,
non-coding-senior, langchain-tourist, title-chaser) are NOT hard-excluded here;
per the JD's own "case-by-case... if other signals are strong" framing, we
apply them as a steep but not infinite penalty, then let the final ranking
sort them appropriately. A candidate who is consulting-only AND otherwise a
perfect semantic/skills match still ranks below a genuine product-company
candidate, but isn't artificially zeroed out in a way that would look like a
bug if a borderline case sits at the cutoff line.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from . import features as feat

WEIGHTS = {
    "semantic_fit": 0.32,
    "skills_match": 0.28,
    "title_relevance": 0.16,
    "experience_fit": 0.08,
    "location_fit": 0.08,
    "notice_period_fit": 0.04,
    "career_trajectory": 0.04,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def score_candidate(
    candidate: dict[str, Any],
    semantic_fit: float,
    today: date,
) -> dict[str, Any]:
    """
    Compute the full score breakdown for one candidate.

    semantic_fit is passed in (already computed in batch by the semantic
    backend in pipeline.py) rather than computed here, since embedding /
    TF-IDF scoring is far more efficient batched across many candidates than
    called once per candidate.
    """
    profile = candidate.get("profile", {})

    title_score = feat.title_tier_score(profile.get("current_title", ""))
    experience_score = feat.experience_fit_score(profile.get("years_of_experience"))
    skills_score, skills_debug = feat.skills_match_score(candidate)
    location_score, location_reason = feat.location_fit_score(candidate)
    notice_score = feat.notice_period_score(
        candidate.get("redrob_signals", {}).get("notice_period_days")
    )
    trajectory_penalty, trajectory_reasons = feat.career_trajectory_penalty(candidate)
    trajectory_score = max(0.0, 1.0 - trajectory_penalty)

    components = {
        "semantic_fit": semantic_fit,
        "skills_match": skills_score,
        "title_relevance": title_score,
        "experience_fit": experience_score,
        "location_fit": location_score,
        "notice_period_fit": notice_score,
        "career_trajectory": trajectory_score,
    }

    base_score = sum(WEIGHTS[k] * v for k, v in components.items())
    base_score = float(np.clip(base_score, 0.0, 1.0))

    behavioral_mult, behavioral_reasons = feat.behavioral_modifier(candidate, today)
    final_score = float(np.clip(base_score * behavioral_mult, 0.0, 1.0))

    return {
        "candidate_id": candidate.get("candidate_id"),
        "final_score": round(final_score, 6),
        "base_score": round(base_score, 6),
        "behavioral_multiplier": round(behavioral_mult, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "skills_debug": skills_debug,
        "location_reason": location_reason,
        "trajectory_reasons": trajectory_reasons,
        "behavioral_reasons": behavioral_reasons,
    }
