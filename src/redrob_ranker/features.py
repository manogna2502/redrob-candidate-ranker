"""
Deterministic, explainable feature extraction.

Every function here takes a raw candidate dict (as loaded from candidates.jsonl)
and returns a small, named numeric/boolean signal. None of these require a model
or network call -- they're pure Python over the JSON structure, which is what
makes Stage A/C fast enough to run over the full 100K pool inside the 5-minute
budget.

Score components are deliberately kept in [0, 1] (or a known bounded range) so
they can be linearly combined in scoring.py without one feature silently
dominating because of scale.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from . import jd_config as cfg

PROFICIENCY_RANK = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}


# ---------------------------------------------------------------------------
# Title relevance
# ---------------------------------------------------------------------------

def title_tier(current_title: str) -> int:
    key = (current_title or "").strip().lower()
    return cfg.TITLE_TIERS.get(key, cfg.DEFAULT_TITLE_TIER)


def title_tier_score(current_title: str) -> float:
    """Normalize tier (0-3) to a [0, 1] component score."""
    return title_tier(current_title) / 3.0


# ---------------------------------------------------------------------------
# Experience band fit
# ---------------------------------------------------------------------------

def experience_fit_score(years_of_experience: float) -> float:
    """1.0 inside [MIN, MAX], smooth penalty outside, floored at 0."""
    yoe = float(years_of_experience or 0)
    if cfg.EXPERIENCE_MIN <= yoe <= cfg.EXPERIENCE_MAX:
        return 1.0
    if yoe < cfg.EXPERIENCE_MIN:
        gap = cfg.EXPERIENCE_MIN - yoe
    else:
        gap = yoe - cfg.EXPERIENCE_MAX
    penalty = gap * cfg.EXPERIENCE_SOFT_PENALTY_PER_YEAR
    return max(0.0, 1.0 - penalty)


# ---------------------------------------------------------------------------
# Skill match: claimed vs Redrob-assessed (this is the keyword-stuffer detector)
# ---------------------------------------------------------------------------

def _flatten_must_have_keywords() -> list[str]:
    return [kw for group in cfg.MUST_HAVE_SKILL_GROUPS for kw in group]


def skills_match_score(candidate: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """
    Score how well a candidate's skill set covers the JD's must-have groups,
    discounted by the gap between claimed proficiency and Redrob's measured
    skill_assessment_scores when both exist for the same skill.

    Returns (score in [0,1], debug dict) so callers/reasoning can explain the
    number without recomputing it.
    """
    skills = candidate.get("skills") or []
    assessed = candidate.get("redrob_signals", {}).get("skill_assessment_scores") or {}

    skill_names_lower = {s["name"].lower(): s for s in skills}

    must_have_keywords = _flatten_must_have_keywords()
    groups_covered = 0
    trust_adjustments: list[float] = []
    matched_skill_names: list[str] = []

    for group in cfg.MUST_HAVE_SKILL_GROUPS:
        hit = None
        for kw in group:
            for name_lower, s in skill_names_lower.items():
                if kw in name_lower or name_lower in kw:
                    hit = s
                    break
            if hit:
                break
        if hit:
            groups_covered += 1
            matched_skill_names.append(hit["name"])
            # Trust adjustment: if claimed proficiency is advanced/expert but the
            # Redrob-assessed score for the *same* skill is low, distrust the claim.
            prof_rank = PROFICIENCY_RANK.get(hit.get("proficiency"), 2)
            assessed_score = assessed.get(hit["name"])
            if assessed_score is not None:
                if prof_rank >= 3 and assessed_score < 45:
                    trust_adjustments.append(0.4)  # heavy discount: claim contradicted
                elif prof_rank >= 3 and assessed_score < 60:
                    trust_adjustments.append(0.75)
                else:
                    trust_adjustments.append(1.0)
            else:
                # No assessment on file -- mild uncertainty discount, not a penalty.
                trust_adjustments.append(0.9)

    base_coverage = groups_covered / max(1, len(cfg.MUST_HAVE_SKILL_GROUPS))
    avg_trust = sum(trust_adjustments) / len(trust_adjustments) if trust_adjustments else 1.0

    # Nice-to-have bonus, small and capped, never enough to compensate for
    # missing must-haves.
    nice_hits = sum(
        1 for kw in cfg.NICE_TO_HAVE_SKILLS
        for name_lower in skill_names_lower
        if kw in name_lower
    )
    nice_bonus = min(0.10, 0.02 * nice_hits)

    score = min(1.0, base_coverage * avg_trust + nice_bonus)

    debug = {
        "groups_covered": groups_covered,
        "total_groups": len(cfg.MUST_HAVE_SKILL_GROUPS),
        "avg_trust": round(avg_trust, 3),
        "matched_skill_names": matched_skill_names,
        "nice_hits": nice_hits,
    }
    return score, debug


def framework_tourist_penalty(candidate: dict[str, Any]) -> float:
    """
    JD: AI experience that's *primarily* recent LangChain-calls-OpenAI, without
    deeper pre-LLM-era production ML, is a soft disqualifier.

    Heuristic: candidate's only AI-ish skill is langchain-flavored AND total
    years_of_experience that overlap with "production ML" skills (embeddings,
    vector db, ranking) is near zero.
    """
    skills = candidate.get("skills") or []
    names_lower = [s["name"].lower() for s in skills]
    has_langchain = any(kw in n for n in names_lower for kw in cfg.FRAMEWORK_TOURIST_SKILL_KEYWORDS)
    if not has_langchain:
        return 0.0

    infra_keywords = [kw for group in cfg.MUST_HAVE_SKILL_GROUPS[:2] for kw in group]
    has_real_infra = any(kw in n for n in names_lower for kw in infra_keywords)
    if has_real_infra:
        return 0.0  # has genuine retrieval/vector-db depth -- not a tourist

    # Only langchain-flavored AI skill, no infra depth -> penalty.
    return 0.25


# ---------------------------------------------------------------------------
# Career trajectory: title-chasing, consulting-only, research-only, non-coding
# ---------------------------------------------------------------------------

def is_consulting_only(candidate: dict[str, Any]) -> bool:
    history = candidate.get("career_history") or []
    companies = [h.get("company", "").lower() for h in history]
    if not companies:
        return False
    return all(any(firm in c for firm in cfg.CONSULTING_FIRMS) for c in companies)


def is_research_only(candidate: dict[str, Any]) -> bool:
    history = candidate.get("career_history") or []
    industries = [h.get("industry", "").lower() for h in history]
    if not industries:
        return False
    return all(any(kw in i for kw in cfg.RESEARCH_ONLY_INDUSTRY_KEYWORDS) for i in industries)


def is_title_chaser(candidate: dict[str, Any]) -> bool:
    history = candidate.get("career_history") or []
    if len(history) < cfg.TITLE_CHASER_MIN_TOTAL_STINTS:
        return False
    short_stints = sum(
        1 for h in history if h.get("duration_months", 999) <= cfg.TITLE_CHASER_MAX_STINT_MONTHS
    )
    return short_stints >= cfg.TITLE_CHASER_MIN_SHORT_STINTS


def is_non_coding_senior(candidate: dict[str, Any]) -> bool:
    title = (candidate.get("profile", {}).get("current_title") or "").lower()
    return any(kw in title for kw in cfg.NON_CODING_TITLE_KEYWORDS)


def career_trajectory_penalty(candidate: dict[str, Any]) -> tuple[float, list[str]]:
    """Returns (multiplicative-style penalty in [0, ~0.6], list of reason tags)."""
    penalty = 0.0
    reasons: list[str] = []

    if is_consulting_only(candidate):
        penalty += 0.30
        reasons.append("consulting_only_career")
    if is_research_only(candidate):
        penalty += 0.35
        reasons.append("research_only_no_production")
    if is_title_chaser(candidate):
        penalty += cfg.TITLE_CHASER_PENALTY
        reasons.append("frequent_job_switching")
    if is_non_coding_senior(candidate):
        penalty += 0.15
        reasons.append("non_coding_senior_title")

    fw_penalty = framework_tourist_penalty(candidate)
    if fw_penalty:
        penalty += fw_penalty
        reasons.append("langchain_only_no_infra_depth")

    return min(penalty, 0.65), reasons


# ---------------------------------------------------------------------------
# Location / logistics fit
# ---------------------------------------------------------------------------

def location_fit_score(candidate: dict[str, Any]) -> tuple[float, str]:
    profile = candidate.get("profile", {})
    location = (profile.get("location") or "").lower()
    country = (profile.get("country") or "").lower()
    willing_to_relocate = candidate.get("redrob_signals", {}).get("willing_to_relocate", False)

    if any(loc in location for loc in cfg.PREFERRED_LOCATIONS_TIER1):
        return 1.0, "tier1_location"
    if any(loc in location for loc in cfg.PREFERRED_LOCATIONS_TIER2):
        return 0.85, "tier2_india_location"
    if country == cfg.TARGET_COUNTRY:
        if willing_to_relocate:
            return 0.70, "india_other_city_willing_to_relocate"
        return 0.55, "india_other_city_not_relocating"
    # Outside India
    if willing_to_relocate:
        return max(0.0, 1.0 - cfg.NON_INDIA_RELOCATE_PENALTY - 0.5), "non_india_willing_to_relocate"
    return max(0.0, 1.0 - cfg.NON_INDIA_PENALTY - 0.5), "non_india_not_relocating"


# ---------------------------------------------------------------------------
# Notice period fit
# ---------------------------------------------------------------------------

def notice_period_score(notice_period_days: int) -> float:
    days = notice_period_days or 0
    if days <= cfg.NOTICE_PERIOD_IDEAL_DAYS:
        return 1.0
    over_units = (days - cfg.NOTICE_PERIOD_IDEAL_DAYS) / 30.0
    penalty = over_units * cfg.NOTICE_PERIOD_PENALTY_PER_30_DAYS_OVER
    return max(0.5, 1.0 - penalty)


# ---------------------------------------------------------------------------
# Behavioral availability modifier (multiplicative, per JD's explicit instruction)
# ---------------------------------------------------------------------------

def behavioral_modifier(candidate: dict[str, Any], today: date) -> tuple[float, list[str]]:
    """
    Returns a multiplier in roughly [0.55, 1.05] and the reasons that drove it.
    Applied multiplicatively to the base fit score -- never additive -- so it
    can meaningfully demote an inactive "perfect-on-paper" candidate without
    ever letting raw activity outrank genuine skill/role fit.
    """
    sig = candidate.get("redrob_signals", {})
    reasons: list[str] = []
    mult = 1.0

    last_active = sig.get("last_active_date")
    if last_active:
        try:
            y, m, d = (int(x) for x in last_active.split("-"))
            days_inactive = (today - date(y, m, d)).days
        except (ValueError, TypeError):
            days_inactive = 0
        if days_inactive >= cfg.INACTIVITY_DAYS_SEVERE:
            mult *= 0.55
            reasons.append(f"inactive_{days_inactive}d")
        elif days_inactive >= cfg.INACTIVITY_DAYS_MODERATE:
            mult *= 0.80
            reasons.append(f"inactive_{days_inactive}d")

    if not sig.get("open_to_work_flag", False):
        mult *= 0.85
        reasons.append("not_marked_open_to_work")

    response_rate = sig.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate < 0.15:
            mult *= 0.80
            reasons.append(f"low_response_rate_{response_rate:.2f}")
        elif response_rate > 0.6:
            mult *= 1.05
            reasons.append(f"high_response_rate_{response_rate:.2f}")

    interview_rate = sig.get("interview_completion_rate")
    if interview_rate is not None and interview_rate < 0.4:
        mult *= 0.85
        reasons.append(f"low_interview_completion_{interview_rate:.2f}")

    github_score = sig.get("github_activity_score", -1)
    if github_score is not None and github_score >= 50:
        mult *= 1.05
        reasons.append(f"strong_github_{github_score:.0f}")

    return max(0.50, min(1.10, mult)), reasons
