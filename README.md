# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

A hybrid ranking system that ranks candidates from a 100,000-profile pool
against the "Senior AI Engineer — Founding Team" job description, without
relying on keyword matching.

> TL;DR reproduction command:
> ```bash
> python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
> ```

---

## 1. The problem, in one paragraph

This dataset is adversarial by design. The JD explicitly states: *"The right
answer to this JD is not 'find candidates whose skills section contains the
most AI keywords.' That's a trap we've explicitly built into the dataset."*
Out of 100,000 candidates, roughly 1% have an ML/AI-adjacent title at all, the
dataset contains ~80 honeypot profiles with logically impossible claims
(verified: our detector finds **88**), and several deliberate trap patterns
exist — keyword-stuffers whose claimed skill proficiency contradicts Redrob's
own measured `skill_assessment_scores`, consulting-only careers, title-chasers,
and "LangChain tourists" with no real retrieval/vector-DB depth. A system that
just embeds skill lists and ranks by cosine similarity will rank these traps
highly. This system is built specifically to not do that.

## 2. Architecture: a four-stage funnel

The compute budget (≤5 min wall-clock, ≤16GB RAM, CPU-only, no network, no
GPU, for 100,000 candidates — see `submission_spec.docx` section 3) rules out
calling any LLM per-candidate. The system is a funnel instead:

```
Stage A (full 100K pool, milliseconds)
  -> drop honeypots (logically-impossible profiles)
  -> drop candidates with zero plausible relevance signal
     (with a guaranteed-floor backoff so we never under-fill the
     required top-100 output)

Stage B (narrowed pool, seconds)
  -> semantic similarity between a JD "ideal candidate" narrative
     and each candidate's own narrative (summary + career history
     descriptions + skills)
  -> sentence-transformers bi-encoder if available, automatic
     TF-IDF + cosine-similarity fallback if not (see semantic.py)

Stage C (same narrowed pool)
  -> full hybrid composite score: semantic_fit (0.32) + skills_match (0.28)
     + title_relevance (0.16) + experience_fit (0.08) + location_fit (0.08)
     + notice_period_fit (0.04) + career_trajectory (0.04)
  -> multiplied by a behavioral availability modifier (0.50-1.10x) per the
     JD's explicit instruction to down-weight inactive/unresponsive
     candidates rather than exclude them additively

Stage D (top 100 only)
  -> grounded, template-based reasoning generation pulling only verified
     profile fields -- no LLM call, so no hallucination risk
```

See `src/redrob_ranker/pipeline.py` for the orchestrator and
`docs/architecture.md` for the full design rationale.

## 3. Why this design, specifically

- **Funnel, not brute force.** An LLM-per-candidate approach cannot fit the
  5-minute budget at 100K candidates even with a small local model (the spec
  says so explicitly). A cheap structural gate (Stage A) → batched semantic
  retrieval (Stage B) → full feature scoring on a much smaller pool (Stage C)
  is the standard real-world pattern for this, and is also literally what the
  JD's own roadmap describes building ("v2 ranking system... embeddings,
  hybrid retrieval, probably LLM re-ranking").
- **Honeypots get a rule, not a model.** Honeypots are *logically* impossible
  (e.g. "expert" proficiency in a skill used for 0 months), not stylistically
  weak. A transparent, auditable rule is cheaper, more reliable, and easier to
  defend at a Stage 5 interview than a learned classifier for something that
  doesn't require learning. Verified against the real `candidates.jsonl`: our
  three checks catch exactly 88 candidates, matching the spec's "~80" closely.
- **The claim-vs-assessment gap is the core anti-keyword-stuffing signal.**
  Redrob's own `redrob_signals.skill_assessment_scores` field lets us check
  whether a candidate's *self-reported* "advanced/expert" skill proficiency is
  backed up by an independently measured score. When it isn't, we discount the
  skill-match contribution for that skill rather than rewarding the keyword
  match at face value. This was found by directly inspecting
  `sample_candidates.json` candidate #1, whose "Fine-tuning LLMs: advanced (21
  endorsements)" claim is contradicted by an assessed score of 41.6/100.
- **Behavioral signals are multiplicative, never additive.** The JD says to
  "down-weight" inactive/unresponsive candidates, not to disqualify them. A
  multiplier preserves relative ordering among similarly-skilled candidates
  while still demoting stale profiles — an additive penalty risks letting a
  mediocre-but-active candidate outrank a strong-but-quiet one, which isn't
  what the JD is asking for.
- **Reasoning is templated from verified facts, not LLM-generated.** Stage 4
  manual review explicitly penalizes hallucinated, generic, or
  name-swap-templated reasoning. A template that only emits a clause when the
  underlying structured fact is actually true, and prioritizes the
  highest-relevance clauses per candidate, produces short (~1-2 sentence),
  honest, candidate-specific reasoning without any hallucination risk and
  without an LLM call (which would also cost compute budget we don't have).

## 4. Repository structure

```
.
├── rank.py                          # CLI entrypoint (the required reproduction command)
├── requirements.txt
├── submission_metadata.yaml
├── README.md                        # this file
├── docs/
│   └── architecture.md              # full design writeup, dataset analysis, justification
├── src/redrob_ranker/
│   ├── jd_config.py                 # JD requirements as structured, auditable config
│   ├── jd_query.py                  # free-text JD narrative used for semantic matching
│   ├── data_loader.py               # candidates.jsonl / sample_candidates.json loading
│   ├── honeypot.py                  # impossible-profile detection (Stage A hard gate)
│   ├── features.py                  # deterministic feature extraction
│   ├── semantic.py                  # SBERT bi-encoder + TF-IDF fallback (Stage B)
│   ├── scoring.py                   # hybrid composite scorer (Stage C)
│   ├── reasoning.py                 # grounded reasoning generation (Stage D)
│   └── pipeline.py                  # Stage A->D orchestrator
├── scripts/
│   ├── precompute_embeddings.py     # one-time SBERT model cache (run with network, ahead of time)
│   └── evaluate.py                  # self-evaluation harness (proxy quality checks)
├── sandbox_app/
│   └── app.py                       # Streamlit demo app (sandbox/reproduction requirement)
├── tests/
│   ├── test_honeypot.py
│   ├── test_features.py
│   └── test_pipeline.py
└── data/
    └── sample_candidates.json       # small fixture for local testing
```

## 5. Setup and exact reproduction command

```bash
# 1. Clone and enter the repo
git clone <this-repo-url>
cd <this-repo>

# 2. Install dependencies
pip install -r requirements.txt --break-system-packages   # or use a venv

# 3. (Optional, recommended) Pre-cache the sentence-transformers model.
#    This step needs network access and should be run ONCE, ahead of time.
#    If skipped, rank.py automatically falls back to the TF-IDF backend --
#    nothing breaks, semantic match quality is just somewhat lower.
python scripts/precompute_embeddings.py

# 4. Run the ranker (no network needed at this point)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 5. (Optional) Validate the output against the official spec
python scripts/validate_submission.py submission.csv

# 6. (Optional) Run our own proxy quality checks
python scripts/evaluate.py --submission ./submission.csv --candidates ./data/candidates.jsonl
```

To force a specific semantic backend (useful for reproducibility testing):

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --backend tfidf
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --backend sbert
```

### Measured runtime (this machine: 1 CPU core, ~4GB RAM available)

| Stage | Time |
|---|---|
| Load 100,000 candidates from JSONL | ~7-30s |
| Stage A (honeypot + plausibility filter) | <1s |
| Stage B (TF-IDF semantic backend, ~99,900 candidates) | ~35-45s |
| Stage C (hybrid scoring, same pool) | <5s |
| Stage D (reasoning, top 100) | <1s |
| **Total** | **~50-85s**, well within the 5-minute / 16GB / CPU-only budget |

The TF-IDF backend was used for all reported numbers above, since this
development environment has no network access to download the SBERT
checkpoint. The SBERT path adds embedding-encode time for the narrowed pool
but is still expected to comfortably fit the budget at this pool size.

## 6. Running the tests

No `pytest` dependency is required to run them:

```bash
python3 -m unittest discover -s tests -v
```

(They are also auto-discovered by `pytest tests/` if you have it installed.)

## 7. Sandbox / demo

A small Streamlit app (`sandbox_app/app.py`) accepts a CSV/JSON sample of
candidates and runs the full pipeline end-to-end, producing a ranked CSV in
the browser. See that folder's own README for hosting instructions
(HuggingFace Spaces / Streamlit Cloud, both free-tier).

## 8. Known limitations / honest caveats

- We do not have the hidden ground-truth relevance labels, so we cannot
  compute the actual NDCG@10/NDCG@50/MAP/P@10 composite ourselves —
  `scripts/evaluate.py` only checks proxy signals we can verify (honeypot
  rate, trap-pattern leakage, score/rank structural validity, reasoning
  quality proxies).
- Stage A's plausibility filter is currently very permissive (it kept 99,912
  of 100,000 candidates in our test run) — almost all of the actual ranking
  separation happens in Stage B/C. This is a deliberate choice to avoid
  accidentally dropping a genuine "Tier 5 candidate whose title doesn't use
  the right words" (the JD's own example of what NOT to do), at the cost of
  some wasted compute on clearly-irrelevant candidates in Stage B. There is
  headroom to tighten Stage A further if runtime ever became a binding
  constraint, but at ~1 minute total runtime against the real 100K dataset,
  it currently isn't.
- The location/title taxonomies in `jd_config.py` were built by directly
  reading `job_description.docx` and cross-referencing the real title census
  in `candidates.jsonl`; they are not exhaustive of every possible synonym a
  candidate might use, and the semantic backend is the safety net for titles
  that don't appear in our tier table.
