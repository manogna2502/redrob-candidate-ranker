"""Redrob Hackathon — Intelligent Candidate Discovery & Ranking.

A hybrid ranking system for the "Senior AI Engineer — Founding Team" JD.

Pipeline (see pipeline.py for the orchestrator):
    Stage A — cheap structural gate   (title relevance, honeypot/consulting filters)
    Stage B — semantic retrieval      (bi-encoder cosine sim, TF-IDF fallback)
    Stage C — hybrid composite scoring + behavioral modifier
    Stage D — grounded reasoning generation (top 100 only)
"""

__version__ = "1.0.0"
