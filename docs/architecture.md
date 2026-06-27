# Architecture & Design Rationale

This document is the detailed version of README.md section 3, written for
anyone (a judge, a teammate, future-us) who wants to understand *why* every
design choice was made, not just *what* the code does.

## 1. Dataset analysis (what we actually found, not assumed)

Everything below was verified by direct inspection of the released
`candidates.jsonl` (100,000 records) and `sample_candidates.json` (50
records), not assumed from the schema alone.

### 1.1 Title distribution is dominated by noise

A full census of `profile.current_title` across all 100,000 candidates shows
the most common titles are: Mechanical Engineer, HR Manager, Content Writer,
Business Analyst, Sales Executive, Customer Support, Accountant, Civil
Engineer, Graphic Designer, Operations Manager -- each appearing roughly
5,700-5,800 times. Genuinely ML/AI-adjacent titles (ML Engineer, AI Research
Engineer, Data Scientist, NLP Engineer, Senior AI Engineer, etc.) total
roughly **1,070 candidates combined** -- about **1%** of the pool. The JD
states this directly: "We are aware this is a narrow profile. We're not
expecting to find many matches in a 100K candidate pool."

### 1.2 Honeypots are present and detectable

The brief states the dataset contains "~80 honeypot candidates with subtly
impossible profiles." We operationalized this with three structural checks
(see `src/redrob_ranker/honeypot.py`):

1. `proficiency == "expert"` with `duration_months == 0` (a logical
   impossibility)
2. 8 or more simultaneous "expert"-rated skills (real profiles in the sample
   average far fewer; this is an extreme outlier threshold, not curve-fit)
3. `career_history` start dates that predate what `years_of_experience`
   implies by more than 3 years of slack

Run against the full 100,000-record dataset, these three checks together flag
**88 candidates** -- closely matching the brief's stated "~80." Notably, when
we sampled 5 of these flagged candidates directly, 4 of the 5 had titles of
exactly "Machine Learning Engineer," "Senior AI Engineer," "Senior Data
Scientist," and "NLP Engineer" -- i.e., honeypots are deliberately given
titles that would otherwise place them near the top of a naive ranking. We
confirmed our pipeline correctly excludes all 5 of these injected honeypots in
a direct test (see `tests/test_pipeline.py::TestStageA::test_drops_honeypots`
and the manual validation described in the project's development log).

### 1.3 The keyword-stuffer trap is real and present

`sample_candidates.json` candidate `CAND_0000001` has title "Backend
Engineer," a narrative describing data-engineering work transitioning toward
ML, but a skills list stuffed with "NLP," "Image Classification," "Fine-tuning
LLMs," "Speech Recognition," "TTS," and "LoRA," several with high endorsement
counts. Critically, the same candidate's `redrob_signals.skill_assessment_scores`
shows "Fine-tuning LLMs" assessed at only **41.6/100**, despite the candidate
claiming "advanced" proficiency with 21 endorsements. This is the dataset's
own built-in lie detector: claimed proficiency vs. independently measured
proficiency, available as data we don't have to infer.

We built `features.skills_match_score` to discount the skill-match
contribution whenever a skill is claimed at advanced/expert level but its
measured assessment score is low (<45 -> 0.4x trust multiplier on that
component; <60 -> 0.75x), rather than crediting the keyword match at face
value.

### 1.4 Consulting-only and other explicit JD disqualifiers are present at scale

A direct scan found:
- **7,034 candidates** whose entire career history is at consulting/services
  firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) -- of which
  **18** carry ML/AI-sounding titles (e.g. "AI Specialist" at Wipro), exactly
  matching the JD's explicit disqualifier: "People who have only worked at
  consulting firms... in their entire career."
- **3,716 candidates** showing a job-hopping pattern (3+ stints of <=18 months
  out of 4+ total stints), matching the JD's "title-chaser" concern.
- 0 candidates with a pure research/academic-only career history in our
  sample scan -- this particular trap may be less prevalent in the released
  data, or expressed differently (e.g. mixed industry labels); our detector
  (`features.is_research_only`) still checks for it per the JD's explicit
  language, and showed 0 leakage into our actual top-100 output.

### 1.5 Sentinel values for missing behavioral data

`redrob_signals.github_activity_score` is `-1` for ~64% of candidates, and
`offer_acceptance_rate` is `-1` for ~60%, in a 20,000-row scan. These are
explicit "no data" sentinels, not "zero activity." Our `behavioral_modifier`
function only applies a positive boost when `github_activity_score >= 50` and
is otherwise neutral on missing values -- never penalizing a candidate just
because GitHub linking is absent, since absence of data is not evidence of
absence of skill.

## 2. Why a funnel architecture, not single-pass scoring

The compute budget -- <=5 minutes wall-clock, <=16GB RAM, CPU-only, no GPU, no
network, for the full 100,000-candidate pool (submission_spec.docx section 3)
-- makes an LLM call per candidate mathematically infeasible: even at an
optimistic 100ms/candidate for a tiny local model, that's 167 minutes, 33x
over budget. The JD's own roadmap for the real product describes this exact
pattern: "Ship a v2 ranking system... This will involve embeddings, hybrid
retrieval, and probably some LLM-based re-ranking." We built the hackathon
submission as a working instance of that roadmap:

- **Stage A** (full pool, milliseconds): a cheap, purely structural filter.
  Honeypots are dropped unconditionally (this is a hard correctness
  requirement -- the spec disqualifies any submission with honeypot rate
  >10% in the top 100). A "plausibility" pre-filter then removes candidates
  with both a zero title-relevance prior AND wildly out-of-band experience,
  with a guaranteed floor (`min_survivors`) so the pool entering Stage B
  never drops below what's needed to fill the required top-100 output --
  this matters because "exactly 100 rows" is a hard format requirement and
  an under-filled submission is an automatic Stage 1 rejection.
- **Stage B** (narrowed pool, seconds): batched semantic similarity between
  one JD "ideal candidate" narrative and every surviving candidate's own
  narrative document (summary + career history descriptions + skill names).
  Batching here matters: encoding documents in bulk through either a
  bi-encoder or a TF-IDF vectorizer is far cheaper than scoring one document
  against the query at a time.
- **Stage C** (same narrowed pool): the full hybrid composite -- see section 3
  below -- runs over every surviving candidate. This is pure pandas/numpy-level
  arithmetic over already-extracted features, which is why it adds negligible
  time even at ~100K rows.
- **Stage D** (top 100 only): grounded reasoning generation. This is the only
  stage restricted to exactly the output size, since reasoning text is only
  needed for what's actually shown to a human reviewer.

Measured runtime on this development machine (TF-IDF backend, 1 CPU core):
**~50-85 seconds end-to-end against the real 100,000-candidate dataset** --
comfortably inside the 5-minute budget, with significant headroom even if the
grading sandbox is slower per-core than this environment.

## 3. The hybrid composite score

```
base_score = 0.32 * semantic_fit
           + 0.28 * skills_match
           + 0.16 * title_relevance
           + 0.08 * experience_fit
           + 0.08 * location_fit
           + 0.04 * notice_period_fit
           + 0.04 * career_trajectory

final_score = base_score * behavioral_modifier   # multiplier in [0.50, 1.10]
```

**Why these weights.** `semantic_fit` and `skills_match` together make up 60%
of the base score because the JD's own framing of the "ideal candidate" is
almost entirely about what the candidate actually built and knows, not where
they currently sit on an org chart -- the JD explicitly downweights
title-matching as a sufficient signal ("a Tier 5 candidate may not use the
words 'RAG' or 'Pinecone' in their profile... if their career history shows
they built a recommendation system at a product company, they're a fit").
`title_relevance` still carries real weight (0.16) because it remains a
useful, cheap prior -- most genuine matches do have a relevant title -- but
deliberately not enough weight to let a keyword-stuffed "Marketing Manager"
profile compete with a real ML engineer, nor enough to fully exclude an
unusually-titled but genuinely strong candidate. `experience_fit`,
`location_fit`, and `notice_period_fit` are real JD requirements but are
explicitly described as soft (the JD calls the 5-9yr band "a range, not a
requirement" and treats non-India location and long notice periods as
case-by-case, not auto-disqualifying) -- hence smaller weights and smooth
penalty curves rather than hard cutoffs. `career_trajectory` (consulting-only,
research-only, title-chasing, non-coding-senior, langchain-tourist) gets a
direct 0.04 weight in the composite and feeds the reasoning generator's
"concern" clause, since the JD treats these less as binary disqualifiers and
more as "we will probably not move forward, unless..." -- i.e. strong enough
to matter, not strong enough to deserve a hard zero.

**Why a multiplicative behavioral modifier, not an additive bonus/penalty.**
The JD's own instruction is explicit: "a perfect-on-paper candidate who hasn't
logged in for 6 months and has a 5% recruiter response rate is, for hiring
purposes, not actually available. Down-weight them appropriately." "Down-weight"
suggests scaling, not adding/subtracting a fixed amount. A multiplier
preserves the relative ordering established by the base score (a
strong-fit-but-quiet candidate still outranks a weak-fit-but-active one, since
0.55 x 0.85 still beats 1.0 x 0.30) while still meaningfully demoting
genuinely stale profiles -- which is closer to what "still a good candidate,
just less reachable right now" should mean in a ranking, versus an additive
penalty that could let raw activity dominate skill/fit entirely if miscalibrated.

## 4. Why reasoning is templated, not LLM-generated

Two independent reasons converged on the same design choice:

1. **Compute.** Generating free-text reasoning for the top 100 via even a
   small local LLM adds meaningful latency and a heavy dependency for a
   feature the grading rubric treats as "optional but heavily recommended" --
   not worth the budget risk for marginal gain.
2. **Trust.** submission_spec.docx section 3's manual-review checklist
   explicitly penalizes: empty reasoning, all-identical reasoning,
   name-swap-templated reasoning, reasoning that mentions skills not in the
   candidate's profile (hallucination), and reasoning that contradicts the
   rank. An LLM prompted to "explain why this is a good fit" will, under time
   or context pressure, sometimes invent plausible-sounding specifics -- which
   is precisely the failure mode that gets a submission flagged at Stage 4. A
   template that only ever emits a clause when the underlying structured
   signal is verified true cannot hallucinate by construction.

Our reasoning generator (`src/redrob_ranker/reasoning.py`) builds 2-4 clauses
per candidate from a priority-ordered set of fact-checked options (title/role
fit, the single most relevant concern if one exists, skill-match evidence,
logistics, and a positive behavioral signal only when there was no concern to
report), then joins them into one sentence -- matching the spec's own worked
examples, which are single ~15-25 word sentences chaining facts with
semicolons. Because different candidates trigger different subsets of
available clauses based on what's actually true about them, two reasonings
read distinctly differently without name-swap templating risk, while staying
short and rank-consistent (lower-ranked candidates' reasoning surfaces a
concern; higher-ranked candidates' reasoning that found no concern surfaces a
positive behavioral signal instead).

## 5. Semantic backend: graceful degradation, not a single point of failure

`src/redrob_ranker/semantic.py` implements a primary/fallback pair:

- **Primary:** `sentence-transformers/all-MiniLM-L6-v2`, a small (~80MB),
  CPU-friendly bi-encoder. Cosine similarity between the JD narrative and each
  candidate's narrative document, batched for efficiency.
- **Fallback:** scikit-learn `TfidfVectorizer` + cosine similarity. Zero extra
  dependencies, fully deterministic, noticeably weaker on paraphrase/synonym
  matches but still effective on shared vocabulary.

`get_semantic_backend()` tries the SBERT path first and catches any exception
(missing package, no network to fetch the model, OOM, corrupted cache, etc.),
falling back to TF-IDF automatically and logging which backend actually ran.
We made this choice because the grading environment's exact package/network
configuration is not fully knowable in advance -- submitting a system that can
only run if a specific optional dependency happens to be present and a
specific model happens to be cached is a real reproduction risk given the
Stage 3 sandboxed re-run requirement. The fallback path was tested end-to-end
against the full 100,000-candidate dataset and produces a valid,
qualitatively sound ranking on its own (see README.md section 5's measured
runtime table, all of which used the TF-IDF path).

## 6. What we deliberately did NOT build, and why

- **No learned ranking model (XGBoost/LTR) on top of the hybrid score.** The
  JD lists learning-to-rank experience as a nice-to-have for the candidate
  being hired, not necessarily a requirement for the hackathon submission
  itself, and we have no labeled relevance data to train one against (the
  ground truth is hidden). Building one without labels would mean designing
  our own pseudo-labels and risk encoding the same blind spots into a model
  that a hand-built scorer at least keeps auditable. A natural v2 extension,
  once some labeled feedback exists, would be exactly this.
- **No vector database (Pinecone/FAISS/etc.) for retrieval.** At <=100,000
  candidates and a <=5-minute budget, brute-force cosine similarity over an
  in-memory matrix is already fast enough (measured: ~35-45s for the TF-IDF
  path over ~99,900 documents) that introducing an external index would add
  operational complexity without a measurable speed benefit at this scale.
  This would become necessary at a materially larger candidate pool or a
  tighter latency budget -- which is exactly the JD's own description of the
  real production system's evolution path.
