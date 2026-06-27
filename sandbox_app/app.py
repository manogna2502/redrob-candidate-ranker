"""
Streamlit sandbox demo app.

Lets a reviewer upload a small candidate sample (JSON array or JSONL) and see
the full Stage A->D pipeline run live, with the ranked output rendered as a
table plus a downloadable CSV in the exact submission format.

Run locally:
    streamlit run sandbox_app/app.py

Deploy (free tier, both on the spec's accepted-platforms list):
    - Streamlit Community Cloud: connect this GitHub repo, set the main file
      path to `sandbox_app/app.py`, deploy.
    - HuggingFace Spaces: create a new Space (SDK: Streamlit), push this repo,
      it auto-detects `sandbox_app/app.py` if you set it as the entry point in
      the Space's app_file config (or copy this file to the Space root as
      app.py).

This app intentionally limits the candidate pool size accepted via upload (a
few thousand rows) to keep demo runs fast and within free-tier compute/memory
limits -- it is a reproducibility/demo surface, not a replacement for running
the full 100K-candidate `rank.py` locally for the actual submission.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.pipeline import run_pipeline  # noqa: E402

MAX_DEMO_CANDIDATES = 5000

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")

st.title("Redrob Hackathon — Candidate Ranker (Sandbox)")
st.caption(
    "Upload a sample of candidates (JSON array or JSONL, matching candidate_schema.json) "
    "to see the full ranking pipeline run end-to-end. For the real 100,000-candidate "
    "submission, run `python rank.py --candidates ... --out ...` locally -- see README.md."
)

with st.expander("How this works (architecture summary)"):
    st.markdown(
        """
        **Stage A** — drop honeypots (logically impossible profiles) and
        clearly out-of-scope candidates.

        **Stage B** — semantic similarity between the JD's "ideal candidate"
        narrative and each candidate's own narrative (summary, career
        history, skills), via a sentence-transformers bi-encoder, or a
        TF-IDF fallback if that model isn't available in this environment.

        **Stage C** — hybrid composite score: semantic fit, skills match
        (discounted against Redrob's own measured skill-assessment scores),
        title relevance, experience fit, location fit, notice period fit,
        and career-trajectory penalties — multiplied by a behavioral
        availability modifier.

        **Stage D** — grounded, template-based reasoning for the top
        results, built only from verified profile facts (no LLM call, no
        hallucination risk).

        See `docs/architecture.md` in the repo for the full design rationale.
        """
    )

uploaded = st.file_uploader(
    "Upload candidates (.json or .jsonl)", type=["json", "jsonl"],
)

backend_choice = st.radio(
    "Semantic backend",
    options=["auto (try sentence-transformers, fall back to TF-IDF)", "force TF-IDF"],
    index=0,
)
backend_arg = None if backend_choice.startswith("auto") else "tfidf"

top_k = st.slider("How many ranked results to show", min_value=5, max_value=100, value=20)

run_clicked = st.button("Run ranking pipeline", type="primary")

if run_clicked:
    if uploaded is None:
        st.error("Please upload a candidate file first.")
    else:
        raw = uploaded.read().decode("utf-8")
        try:
            if uploaded.name.endswith(".jsonl"):
                candidates = [json.loads(line) for line in raw.splitlines() if line.strip()]
            else:
                candidates = json.loads(raw)
        except json.JSONDecodeError as exc:
            st.error(f"Could not parse the uploaded file as JSON/JSONL: {exc}")
            candidates = None

        if candidates is not None:
            if len(candidates) > MAX_DEMO_CANDIDATES:
                st.warning(
                    f"File has {len(candidates)} candidates; this demo caps at "
                    f"{MAX_DEMO_CANDIDATES} to keep the hosted sandbox responsive. "
                    f"Truncating to the first {MAX_DEMO_CANDIDATES}."
                )
                candidates = candidates[:MAX_DEMO_CANDIDATES]

            effective_top_k = min(top_k, len(candidates))

            with st.spinner(f"Running Stage A-D over {len(candidates)} candidates..."):
                results, stats = run_pipeline(
                    candidates, top_k=effective_top_k, backend_name=backend_arg,
                )

            st.success(
                f"Done. Semantic backend used: **{stats['semantic_backend']}**. "
                f"Honeypots dropped: {stats['stage_a']['honeypots_dropped']}."
            )

            st.subheader(f"Top {len(results)} ranked candidates")
            table_rows = [
                {
                    "rank": i + 1,
                    "candidate_id": r["candidate_id"],
                    "score": r["final_score"],
                    "reasoning": r.get("reasoning", ""),
                }
                for i, r in enumerate(results)
            ]
            st.dataframe(table_rows, use_container_width=True)

            csv_buffer = io.StringIO()
            csv_buffer.write("candidate_id,rank,score,reasoning\n")
            for row in table_rows:
                reasoning_escaped = row["reasoning"].replace('"', '""')
                csv_buffer.write(f'{row["candidate_id"]},{row["rank"]},{row["score"]:.4f},"{reasoning_escaped}"\n')

            st.download_button(
                "Download as submission CSV",
                data=csv_buffer.getvalue(),
                file_name="submission.csv",
                mime="text/csv",
            )

            with st.expander("Run stats (Stage A details)"):
                st.json(stats)
