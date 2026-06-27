"""
Structured encoding of job_description.docx — "Senior AI Engineer, Founding Team" @ Redrob AI.

This module exists so the JD's requirements are auditable data, not magic numbers
buried inside scoring.py. Every constant below traces back to an explicit sentence
in the JD. Where we made a judgment call interpreting ambiguous JD language, it's
flagged with a comment.

If you re-run this system against a *different* JD, this is the only file (plus
the free-text JD_SUMMARY_FOR_EMBEDDING string in semantic.py) that should change.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Role identity — used for the semantic-retrieval query and for reasoning text
# ---------------------------------------------------------------------------

ROLE_TITLE = "Senior AI Engineer — Founding Team"
COMPANY = "Redrob AI"

# ---------------------------------------------------------------------------
# Experience band
# ---------------------------------------------------------------------------
# JD: "Experience Required: 5-9 years ... This is a range, not a requirement.
#      We'll seriously consider candidates outside the band if other signals
#      are strong." -> soft band, not a hard filter. We score a smooth penalty
#      outside [MIN, MAX] rather than excluding candidates.
EXPERIENCE_MIN = 5.0
EXPERIENCE_MAX = 9.0
EXPERIENCE_SOFT_PENALTY_PER_YEAR = 0.06  # penalty per year outside the band

# ---------------------------------------------------------------------------
# Title relevance tiers
# ---------------------------------------------------------------------------
# Built from the *actual* title census of candidates.jsonl (100K scan) cross-
# referenced against the JD's described ideal candidate ("6-8 years ... applied
# ML/AI roles at product companies"). Tier 0 titles are pure noise (mechanical
# engineer, accountant, etc.) and should not survive Stage A at all once the
# pool is narrowed by Stage B; they exist here mainly to support honest
# disqualification reasoning, not as something we expect to rank.
TITLE_TIERS: dict[str, int] = {
    # Tier 3 — exact or near-exact role match
    "senior ai engineer": 3,
    "lead ai engineer": 3,
    "staff machine learning engineer": 3,
    "senior machine learning engineer": 3,
    "senior applied scientist": 3,
    "senior nlp engineer": 3,
    "senior data scientist": 3,
    # Tier 2 — strong adjacent: applied ML/AI IC roles
    "ai engineer": 2,
    "machine learning engineer": 2,
    "ml engineer": 2,
    "applied ml engineer": 2,
    "nlp engineer": 2,
    "ai research engineer": 2,
    "ai specialist": 2,
    "computer vision engineer": 2,
    "recommendation systems engineer": 2,
    "search engineer": 2,
    "data scientist": 2,
    "junior ml engineer": 2,
    "senior software engineer (ml)": 2,
    # Tier 1 — plausible if career history shows real ML/retrieval/ranking work
    "data engineer": 1,
    "senior data engineer": 1,
    "analytics engineer": 1,
    "data analyst": 1,
    "backend engineer": 1,
    "senior software engineer": 1,
    "software engineer": 1,
    "full stack developer": 1,
    "devops engineer": 1,
    "cloud engineer": 1,
}
# Anything not listed defaults to tier 0 (no a-priori relevance; must earn fit
# entirely through semantic/career-history signal, which will almost always
# fail for genuinely irrelevant titles like "Mechanical Engineer").
DEFAULT_TITLE_TIER = 0

# ---------------------------------------------------------------------------
# Hard / near-hard disqualifiers, straight from the JD's "explicitly do NOT
# want" and "disqualifiers we actually apply" sections.
# ---------------------------------------------------------------------------

# "If you've spent your career in pure research environments (academic labs,
#  research-only roles) without any production deployment -- we will not move
#  forward."
RESEARCH_ONLY_INDUSTRY_KEYWORDS = ("research", "academia", "university", "academic")

# "People who have only worked at consulting firms ... in their entire career."
CONSULTING_FIRMS = (
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
)

# "People whose primary expertise is computer vision, speech, or robotics
#  without significant NLP/IR exposure."
NON_NLP_SPECIALIST_TITLES = ("computer vision engineer",)  # speech/robotics not seen as titles in data

# "Title-chasers ... switching companies every 1.5 years."
TITLE_CHASER_MAX_STINT_MONTHS = 18
TITLE_CHASER_MIN_SHORT_STINTS = 3
TITLE_CHASER_MIN_TOTAL_STINTS = 4
TITLE_CHASER_PENALTY = 0.15

# "If your 'AI experience' consists primarily of recent (<12mo) projects using
#  LangChain to call OpenAI -- we will probably not move forward, unless you
#  can demonstrate substantial pre-LLM-era ML production experience."
FRAMEWORK_TOURIST_SKILL_KEYWORDS = ("langchain",)
FRAMEWORK_TOURIST_MAX_MONTHS = 12

# "Senior engineer who hasn't written production code in 18 months because
#  they moved into 'architecture'/'tech lead' roles -- we will probably not
#  move forward. This role writes code."
NON_CODING_TITLE_KEYWORDS = ("architect", "tech lead", "engineering manager", "director")

# ---------------------------------------------------------------------------
# Must-have technical surface area (semantic + lexical matching target)
# ---------------------------------------------------------------------------
MUST_HAVE_SKILL_GROUPS: list[list[str]] = [
    # embeddings-based retrieval, production
    ["sentence-transformers", "openai embeddings", "bge", "e5", "embeddings"],
    # vector db / hybrid search infra
    ["pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch", "faiss",
     "vector database", "vector db", "hybrid search"],
    # python
    ["python"],
    # eval rigor
    ["ndcg", "mrr", "map", "a/b test", "a/b testing", "offline evaluation",
     "evaluation framework", "ranking metrics"],
]

NICE_TO_HAVE_SKILLS: list[str] = [
    "lora", "qlora", "peft", "fine-tuning llms", "fine-tuning",
    "learning to rank", "xgboost", "ltr",
    "recruiting tech", "hr-tech", "marketplace",
    "distributed systems", "large-scale inference", "inference optimization",
    "open source", "open-source",
]

# ---------------------------------------------------------------------------
# Location / logistics
# ---------------------------------------------------------------------------
# JD: "Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply."
PREFERRED_LOCATIONS_TIER1 = ("pune", "noida")
PREFERRED_LOCATIONS_TIER2 = ("hyderabad", "mumbai", "delhi", "new delhi", "gurugram", "gurgaon", "ncr")
TARGET_COUNTRY = "india"
# "Outside India: case-by-case, but we don't sponsor work visas." -> non-India
# candidates are penalized, not hard-excluded, unless willing_to_relocate=True.
NON_INDIA_PENALTY = 0.20
NON_INDIA_RELOCATE_PENALTY = 0.08

# ---------------------------------------------------------------------------
# Notice period
# ---------------------------------------------------------------------------
# "We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice
#  candidates are still in scope but the bar gets higher."
NOTICE_PERIOD_IDEAL_DAYS = 30
NOTICE_PERIOD_PENALTY_PER_30_DAYS_OVER = 0.05

# ---------------------------------------------------------------------------
# Behavioral signal modifier weights
# ---------------------------------------------------------------------------
# JD: "weigh behavioral signals -- a perfect-on-paper candidate who hasn't
#  logged in for 6 months and has a 5% recruiter response rate is, for hiring
#  purposes, not actually available. Down-weight them appropriately."
# Implemented as a MULTIPLIER (range ~[0.55, 1.05]) applied to the base fit
# score, never as an additive bonus -- a behaviorally "perfect" candidate
# should not outrank a strong-fit candidate purely on activity.
INACTIVITY_DAYS_SEVERE = 180   # ~6 months, matches JD's own example
INACTIVITY_DAYS_MODERATE = 60
