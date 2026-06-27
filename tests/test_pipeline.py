"""
Tests for the pipeline orchestrator (src/redrob_ranker/pipeline.py).

Run with: python3 -m unittest discover -s tests

These use backend_name="tfidf" exclusively, since sentence-transformers is an
optional dependency that may not be installed in every environment these
tests run in (including, deliberately, this repo's own CI-less sandbox dev
loop). The TF-IDF path is fully deterministic and dependency-light, which
makes these tests fast and reliable; the SBERT path is covered separately by
scripts/precompute_embeddings.py's own smoke check, not by this suite.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.pipeline import run_pipeline, stage_a_filter


def _make_candidate(cid, title="AI Engineer", yoe=6.0, expert_skills=None):
    return {
        "candidate_id": cid,
        "profile": {
            "current_title": title,
            "years_of_experience": yoe,
            "location": "Pune, Maharashtra",
            "country": "India",
            "current_company": "TestCo",
        },
        "skills": expert_skills or [
            {"name": "Python", "proficiency": "advanced", "duration_months": 48},
            {"name": "Pinecone", "proficiency": "advanced", "duration_months": 24},
        ],
        "career_history": [
            {"company": "TestCo", "title": title, "start_date": "2020-01-01",
             "duration_months": 36, "description": "Built a ranking system.", "industry": "Tech"},
        ],
        "redrob_signals": {
            "skill_assessment_scores": {"Python": 80, "Pinecone": 75},
            "last_active_date": "2026-06-20",
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.5,
            "interview_completion_rate": 0.7,
            "notice_period_days": 30,
            "github_activity_score": 40,
            "willing_to_relocate": True,
        },
    }


class TestStageA(unittest.TestCase):
    def test_drops_honeypots(self):
        clean = _make_candidate("CAND_0000001")
        honeypot = _make_candidate(
            "CAND_0000002",
            expert_skills=[
                {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 24}
                for i in range(9)
            ],
        )
        kept, stats = stage_a_filter([clean, honeypot], min_survivors=1)
        kept_ids = {c["candidate_id"] for c in kept}
        self.assertNotIn("CAND_0000002", kept_ids)
        self.assertEqual(stats["honeypots_dropped"], 1)

    def test_backoff_guarantees_min_survivors(self):
        # All candidates have a weak title (tier 0) and experience far enough
        # outside the band that experience_fit_score floors at 0.0 (verified:
        # yoe=30 -> exp_score=0.0, since the penalty saturates), so plausibility
        # is exactly 0 for all of them -- the plausibility cut alone would drop
        # everyone, and the backoff must still return at least min_survivors.
        candidates = [
            _make_candidate(f"CAND_{i:07d}", title="Mechanical Engineer", yoe=30.0)
            for i in range(10)
        ]
        kept, stats = stage_a_filter(candidates, min_survivors=5)
        self.assertGreaterEqual(len(kept), 5)
        self.assertTrue(stats.get("plausibility_backoff_applied"))


class TestRunPipeline(unittest.TestCase):
    def test_end_to_end_small_pool(self):
        candidates = [_make_candidate(f"CAND_{i:07d}", yoe=5.0 + i * 0.1) for i in range(20)]
        results, stats = run_pipeline(candidates, top_k=10, backend_name="tfidf")
        self.assertEqual(len(results), 10)
        # Every result must have a non-empty reasoning string.
        self.assertTrue(all(r.get("reasoning") for r in results))

    def test_scores_are_non_increasing_with_tiebreak(self):
        candidates = [_make_candidate(f"CAND_{i:07d}") for i in range(15)]
        results, _ = run_pipeline(candidates, top_k=15, backend_name="tfidf")
        scores = [r["final_score"] for r in results]
        self.assertTrue(all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)))
        # For any adjacent equal-score pair, candidate_id must be ascending.
        for i in range(len(results) - 1):
            if results[i]["final_score"] == results[i + 1]["final_score"]:
                self.assertLess(results[i]["candidate_id"], results[i + 1]["candidate_id"])

    def test_no_duplicate_candidate_ids_in_output(self):
        candidates = [_make_candidate(f"CAND_{i:07d}") for i in range(30)]
        results, _ = run_pipeline(candidates, top_k=20, backend_name="tfidf")
        ids = [r["candidate_id"] for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_handles_pool_smaller_than_top_k(self):
        candidates = [_make_candidate(f"CAND_{i:07d}") for i in range(5)]
        results, _ = run_pipeline(candidates, top_k=100, backend_name="tfidf")
        # Should not crash, and should not fabricate more rows than exist.
        self.assertEqual(len(results), 5)


if __name__ == "__main__":
    unittest.main()
