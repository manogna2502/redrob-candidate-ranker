# Sandbox Demo App

A small Streamlit app that runs the full ranking pipeline on an uploaded
candidate sample and lets you download a submission-format CSV from the
browser. This satisfies the hackathon's mandatory "hosted sandbox" submission
requirement (see `submission_spec.docx` section 10.5 — accepted platforms
include HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, Binder).

## Run locally

```bash
pip install streamlit -r ../requirements.txt --break-system-packages
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Deploy to Streamlit Community Cloud (free tier)

1. Push this repo to GitHub.
2. Go to https://share.streamlit.io, sign in, click "New app."
3. Select this repo and branch.
4. Set the main file path to `sandbox_app/app.py`.
5. Deploy. You'll get a public URL like
   `https://your-app-name.streamlit.app` — use this as `sandbox_link` in
   `submission_metadata.yaml`.

## Deploy to HuggingFace Spaces (free tier)

1. Create a new Space at https://huggingface.co/new-space.
2. Choose SDK: **Streamlit**.
3. Either:
   - Push this whole repo to the Space's git remote and set the Space's
     `app_file` to `sandbox_app/app.py` in the Space's `README.md` front
     matter, **or**
   - Copy `sandbox_app/app.py` to the Space's root as `app.py` and copy
     `src/redrob_ranker/` alongside it (adjust the `sys.path.insert` line at
     the top of `app.py` if you flatten the directory structure).
4. Make sure `requirements.txt` is present at the Space root (the one in this
   repo's root works as-is).
5. The Space will build and give you a public URL like
   `https://huggingface.co/spaces/your-username/redrob-ranker` — use this as
   `sandbox_link` in `submission_metadata.yaml`.

## What to upload for the demo

Use `data/sample_candidates.json` (the 50-candidate sample bundled in this
repo) for a quick smoke test, or any subset of the real `candidates.jsonl`
(capped at 5,000 rows by the app itself, to keep free-tier hosting
responsive). The app is a reproducibility/demo surface — the actual graded
submission should be produced by running `rank.py` locally against the full
100,000-candidate dataset, per the main README.
