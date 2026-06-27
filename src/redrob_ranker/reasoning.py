"""
Grounded reasoning generation.

This deliberately does NOT call an LLM. Two reasons:

1. Compute budget: generating free-text reasoning for the top ~100 via a local
   LLM is feasible time-wise, but adds a heavyweight dependency and nontrivial
   latency for marginal benefit, when the same explanatory power can be built
   from facts we've already extracted as structured data.
2. Trust: the brief's Stage 4 review explicitly penalizes reasoning that's
   hallucinated, generic, or template-with-name-swapped. An LLM asked to
   "explain why this candidate is a good fit" will cheerfully invent specifics
   if the prompt budget runs thin or the candidate is a weaker match -- which
   is exactly the failure mode we cannot afford to risk on a graded artifact.

Instead, we build reasoning out of small, fact-grounded clauses, each one only
emitted when the underlying structured signal that supports it is actually
present and above a threshold, then keep only the highest-priority clauses per
candidate (bounded by MAX_CLAUSES) to match the spec's "1-2 sentence
justification" guidance -- submission_spec.docx's own worked examples are
single sentences of ~15-25 words, chaining facts with semicolons. Different
candidates surface different clauses based on what's actually true about them
and which clauses rank highest in priority for that candidate, so two
reasonings naturally read differently without name-swap templating risk, while
staying short.

Priority order when truncating to the clause budget:
    1. Title/role relevance (always included -- it's the anchor fact)
    2. The single strongest concern, if any -- rank-consistency requires
       lower-ranked candidates to surface their concerns, not just strengths
    3. Skill/semantic match evidence
    4. Logistics (location / notice period)
    5. A positive behavioral signal, only if there was no concern to report
"""

from __future__ import annotations

from typing import Any

MAX_CLAUSES = 4


def _title_clause(candidate: dict[str, Any], comps: dict[str, float]) -> str:
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "this role")
    company = profile.get("current_company", "current company")
    yoe = profile.get("years_of_experience")
    yoe_str = f"{yoe:g}" if isinstance(yoe, (int, float)) else str(yoe)
    if comps["title_relevance"] >= 0.66:
        return f"{title} at {company} with {yoe_str} yrs experience, directly in scope for the role"
    elif comps["title_relevance"] >= 0.33:
        return f"{title} at {company} with {yoe_str} yrs experience, an adjacent role to what we need"
    return f"{title} at {company} with {yoe_str} yrs experience -- title alone isn't a strong signal here"


def _concern_clause(candidate: dict[str, Any], score_result: dict[str, Any]) -> str | None:
    """Return the single highest-priority concern clause, or None if clean."""
    trajectory_reasons = score_result.get("trajectory_reasons", [])
    concern_text = {
        "research_only_no_production": "career history is pure research/academic with no production deployment on file",
        "consulting_only_career": "entire career history is at consulting/services firms, no product-company experience on file",
        "frequent_job_switching": "career shows a pattern of short stints, matching the JD's title-chaser concern",
        "non_coding_senior_title": "title suggests a move away from hands-on coding (architecture/tech-lead), which the JD flags",
        "langchain_only_no_infra_depth": "AI skills lean on LangChain/LLM-API usage without deeper retrieval/vector-db depth",
    }
    # Highest-severity concerns first (matches feat.career_trajectory_penalty weighting).
    for key in ("research_only_no_production", "consulting_only_career",
                "langchain_only_no_infra_depth", "frequent_job_switching",
                "non_coding_senior_title"):
        if key in trajectory_reasons:
            return concern_text[key]

    avg_trust = score_result["skills_debug"].get("avg_trust", 1.0)
    if avg_trust < 0.85:
        return "Redrob's measured skill-assessment scores run notably below the self-claimed proficiency on a key skill"

    loc_reason = score_result.get("location_reason", "")
    if "non_india" in loc_reason and "willing_to_relocate" not in loc_reason:
        location = candidate.get("profile", {}).get("location", "outside India")
        return f"based in {location} with no relocation signal; JD does not sponsor visas"

    behavioral_reasons = score_result.get("behavioral_reasons", [])
    inactive_reason = next((r for r in behavioral_reasons if "inactive" in r), None)
    if inactive_reason:
        days = inactive_reason.split("_")[-1].replace("d", "")
        return f"platform activity is stale (inactive ~{days} days), down-weighted for likely unavailability"

    notice_days = candidate.get("redrob_signals", {}).get("notice_period_days")
    if notice_days is not None and notice_days > 30:
        return f"notice period of {notice_days} days exceeds the JD's preferred sub-30-day window"

    return None


def _skills_clause(score_result: dict[str, Any]) -> str | None:
    matched = score_result["skills_debug"].get("matched_skill_names", [])
    groups_covered = score_result["skills_debug"].get("groups_covered", 0)
    total_groups = score_result["skills_debug"].get("total_groups", 4)
    if matched:
        skill_list = ", ".join(matched[:3])
        return f"covers {groups_covered}/{total_groups} must-have skill areas including {skill_list}"
    if groups_covered == 0:
        return "no clear coverage of the JD's must-have skill areas in the listed skills"
    return f"covers {groups_covered}/{total_groups} must-have skill areas"


def _logistics_clause(candidate: dict[str, Any], score_result: dict[str, Any]) -> str | None:
    loc_reason = score_result.get("location_reason", "")
    location = candidate.get("profile", {}).get("location", "")
    if "tier1" in loc_reason:
        return f"{location}-based, matching the team's preferred Pune/Noida hub"
    notice_days = candidate.get("redrob_signals", {}).get("notice_period_days")
    if notice_days is not None and notice_days <= 30:
        return f"{notice_days}-day notice period fits the JD's preferred window"
    return None


def _positive_signal_clause(score_result: dict[str, Any]) -> str | None:
    behavioral_reasons = score_result.get("behavioral_reasons", [])
    if any("strong_github" in r for r in behavioral_reasons):
        return "strong public GitHub activity adds independent evidence of hands-on building"
    if any("high_response_rate" in r for r in behavioral_reasons):
        return "strong historical recruiter response rate"
    return None


def build_reasoning(candidate: dict[str, Any], score_result: dict[str, Any]) -> str:
    comps = score_result["components"]

    clauses: list[str] = [_title_clause(candidate, comps)]

    concern = _concern_clause(candidate, score_result)
    if concern:
        clauses.append(concern)

    skills = _skills_clause(score_result)
    if skills and len(clauses) < MAX_CLAUSES:
        clauses.append(skills)

    if len(clauses) < MAX_CLAUSES:
        logistics = _logistics_clause(candidate, score_result)
        if logistics:
            clauses.append(logistics)

    if len(clauses) < MAX_CLAUSES and not concern:
        positive = _positive_signal_clause(score_result)
        if positive:
            clauses.append(positive)

    text = "; ".join(clauses)
    return text[0].upper() + text[1:] + "."
