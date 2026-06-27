"""
Semantic retrieval — JD <-> candidate similarity.

Design constraint: the grading environment is CPU-only, no network access, with
a 16GB RAM / ~5 minute budget for 100K candidates (see submission_spec.docx).
That rules out calling any hosted LLM/embedding API at scoring time, and it
means we cannot assume `sentence-transformers` + a pretrained checkpoint will
successfully import/load in the grading container (pip install may be blocked,
or a model file may not be cached on disk there even if it is here).

So this module implements two interchangeable backends behind one interface:

    1. SBERT_BACKEND   -- sentence-transformers bi-encoder, cosine similarity.
                          Best semantic quality: catches a candidate whose
                          summary says "built a personalization engine using
                          dense retrieval" without ever using the literal words
                          in the JD.
    2. TFIDF_BACKEND    -- scikit-learn TF-IDF + cosine similarity. Zero extra
                          dependencies beyond what's already in requirements.txt,
                          always available, noticeably weaker on paraphrase /
                          synonym matches but still solid on shared vocabulary.

get_semantic_backend() tries SBERT first and falls back to TF-IDF on *any*
failure (missing package, no network to fetch the model, OOM, etc.), logging
which backend was actually used. The rest of the pipeline never needs to know
which one is active -- it just calls `.score(jd_text, candidate_texts)`.
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Small, CPU-friendly, well-regarded general-purpose sentence embedding model.
# ~80MB, runs comfortably under the 16GB/5-min budget even for a few thousand
# candidates surviving the Stage A funnel.
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticBackend(Protocol):
    name: str

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        """Return a [len(documents)] array of similarity scores in [0, 1]."""
        ...


class SbertBackend:
    name = "sentence-transformers (all-MiniLM-L6-v2)"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # local import: optional dep

        self._model = SentenceTransformer(SBERT_MODEL_NAME)

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        from sentence_transformers import util

        query_emb = self._model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        doc_embs = self._model.encode(
            documents, convert_to_tensor=True, normalize_embeddings=True, batch_size=64,
            show_progress_bar=False,
        )
        sims = util.cos_sim(query_emb, doc_embs).cpu().numpy().flatten()
        # cosine sim in [-1, 1] -> rescale to [0, 1]
        return (sims + 1.0) / 2.0


class TfidfBackend:
    name = "TF-IDF + cosine similarity (fallback)"

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # always available (requirements.txt)

        self._vectorizer_cls = TfidfVectorizer

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [query] + list(documents)
        vectorizer = self._vectorizer_cls(
            stop_words="english", ngram_range=(1, 2), max_features=20000, sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(corpus)
        query_vec, doc_vecs = matrix[0:1], matrix[1:]
        sims = cosine_similarity(query_vec, doc_vecs).flatten()
        return np.clip(sims, 0.0, 1.0)


def get_semantic_backend(force: str | None = None) -> SemanticBackend:
    """
    Resolve the best available semantic backend.

    force: "sbert" | "tfidf" | None (auto: try sbert, fall back to tfidf)
    """
    if force == "tfidf":
        logger.info("Semantic backend forced to TF-IDF.")
        return TfidfBackend()

    if force in (None, "sbert"):
        try:
            backend = SbertBackend()
            logger.info("Semantic backend: %s", backend.name)
            return backend
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: any failure -> fallback
            if force == "sbert":
                raise
            logger.warning(
                "sentence-transformers unavailable (%s: %s). Falling back to TF-IDF backend.",
                type(exc).__name__, exc,
            )

    backend = TfidfBackend()
    logger.info("Semantic backend: %s", backend.name)
    return backend


def candidate_document(candidate: dict) -> str:
    """
    Build the text blob used to represent a candidate for semantic matching.

    We deliberately weight career_history descriptions heavily -- per the JD's
    own framing, a great candidate's *narrative* of what they built matters far
    more than their skills list, which is exactly the field keyword-stuffers
    game the hardest.
    """
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
    ]
    for job in candidate.get("career_history", []) or []:
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for skill in candidate.get("skills", []) or []:
        parts.append(skill.get("name", ""))
    return " . ".join(p for p in parts if p)
