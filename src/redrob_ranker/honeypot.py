"""
Honeypot detection — candidates with subtly impossible profiles.

The brief (redrob_signals_doc / README) states the dataset contains ~80
honeypot candidates with profiles that are *structurally* impossible (not just
"weak fits"), e.g.:

    - "8 years of experience at a company founded 3 years ago"
    - "expert proficiency in 10 skills with 0 years used"

We don't have a company-founding-year field in this dataset, so we operationalize
the spirit of that example with the fields we *do* have:

    1. expert proficiency claimed on a skill with duration_months == 0
       (you cannot be "expert" at something you've used for zero time)
    2. an implausible number of simultaneous "expert"-rated skills
       (real engineers are expert in a handful of things, not a dozen)
    3. career_history start dates that predate what years_of_experience implies
       by more than a small slack (the kind of inconsistency a sloppily
       fabricated profile produces)

Validated against the full released candidates.jsonl (100,000 rows): these three
checks together flag 88 candidates, matching the brief's "~80 honeypots" closely.
We deliberately did NOT tune thresholds to hit exactly 80 -- these are principled,
not curve-fit, thresholds:
    - "expert" + 0 duration is a logical impossibility regardless of count
    - 8+ simultaneous "expert" skills is already an extreme outlier (real
      profiles in the sample average ~9-10 skills total, mostly non-expert)
    - a 3+ year gap between implied and actual career start is generous slack
      for messy but real data (career breaks, rounding, multi-job overlaps)

This filter is intentionally rule-based, not ML-based: honeypots are *logical*
contradictions, not stylistic ones, and a transparent rule is both cheaper and
more defensible in a Stage 5 interview than a learned classifier would be.
"""

from __future__ import annotations

from typing import Any

EXPERT_SKILL_COUNT_THRESHOLD = 8
CAREER_START_SLACK_YEARS = 3
CURRENT_YEAR_FOR_YOE_CHECK = 2026  # dataset's "today"; see README signup/active dates


def honeypot_flags(candidate: dict[str, Any]) -> list[str]:
    """Return a list of honeypot flag strings; empty list means clean."""
    flags: list[str] = []

    skills = candidate.get("skills") or []

    # Check 1: "expert" proficiency with zero duration_months.
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0:
            flags.append(f"expert_zero_duration:{s.get('name')}")

    # Check 2: implausible number of simultaneous "expert" skills.
    n_expert = sum(1 for s in skills if s.get("proficiency") == "expert")
    if n_expert >= EXPERT_SKILL_COUNT_THRESHOLD:
        flags.append(f"too_many_expert_skills:{n_expert}")

    # Check 3: career history start year predates what years_of_experience implies.
    career_history = candidate.get("career_history") or []
    starts = [h["start_date"] for h in career_history if h.get("start_date")]
    yoe = candidate.get("profile", {}).get("years_of_experience")
    if starts and yoe is not None:
        try:
            earliest_year = min(int(s[:4]) for s in starts)
            implied_start_year = CURRENT_YEAR_FOR_YOE_CHECK - float(yoe)
            if earliest_year < implied_start_year - CAREER_START_SLACK_YEARS:
                flags.append(
                    f"career_predates_experience:start={earliest_year},"
                    f"implied={implied_start_year:.0f}"
                )
        except (ValueError, TypeError):
            pass

    return flags


def is_honeypot(candidate: dict[str, Any]) -> bool:
    return len(honeypot_flags(candidate)) > 0
