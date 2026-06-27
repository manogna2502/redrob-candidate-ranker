"""
Tests for honeypot detection (src/redrob_ranker/honeypot.py).

Written as unittest.TestCase so they run with zero extra dependencies via:
    python3 -m unittest discover -s tests
They are also auto-discovered by pytest if it happens to be installed.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker.honeypot import honeypot_flags, is_honeypot


def _base_candidate(**overrides):
    base = {
        "candidate_id": "CAND_0000000",
        "profile": {"years_of_experience": 6.0, "current_title": "AI Engineer"},
        "skills": [
            {"name": "Python", "proficiency": "advanced", "duration_months": 60},
            {"name": "Embeddings", "proficiency": "intermediate", "duration_months": 24},
        ],
        "career_history": [
            {"company": "Acme", "start_date": "2020-01-01", "duration_months": 36},
            {"company": "Beta", "start_date": "2023-01-01", "duration_months": 36},
        ],
    }
    base.update(overrides)
    return base


class TestHoneypotDetection(unittest.TestCase):
    def test_clean_candidate_is_not_a_honeypot(self):
        c = _base_candidate()
        self.assertFalse(is_honeypot(c))
        self.assertEqual(honeypot_flags(c), [])

    def test_expert_with_zero_duration_is_flagged(self):
        c = _base_candidate(skills=[
            {"name": "LangChain", "proficiency": "expert", "duration_months": 0},
        ])
        flags = honeypot_flags(c)
        self.assertTrue(any("expert_zero_duration" in f for f in flags))
        self.assertTrue(is_honeypot(c))

    def test_too_many_expert_skills_is_flagged(self):
        skills = [
            {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 24}
            for i in range(9)
        ]
        c = _base_candidate(skills=skills)
        flags = honeypot_flags(c)
        self.assertTrue(any("too_many_expert_skills" in f for f in flags))
        self.assertTrue(is_honeypot(c))

    def test_seven_expert_skills_is_not_flagged_by_count_alone(self):
        # Threshold is >= 8; exactly 7 (with nonzero duration) should not trip
        # the count-based check on its own.
        skills = [
            {"name": f"Skill{i}", "proficiency": "expert", "duration_months": 24}
            for i in range(7)
        ]
        c = _base_candidate(skills=skills)
        flags = honeypot_flags(c)
        self.assertFalse(any("too_many_expert_skills" in f for f in flags))

    def test_career_predates_experience_is_flagged(self):
        # years_of_experience=3 implies a 2023 start, but career_history
        # starts in 2010 -- a 13-year gap, well beyond the 3-year slack.
        c = _base_candidate(
            profile={"years_of_experience": 3.0, "current_title": "AI Engineer"},
            career_history=[{"company": "Acme", "start_date": "2010-01-01", "duration_months": 36}],
        )
        flags = honeypot_flags(c)
        self.assertTrue(any("career_predates_experience" in f for f in flags))

    def test_small_career_start_slack_is_tolerated(self):
        # years_of_experience=6 implies a 2020 start; career_history starts in
        # 2018 -- a 2-year gap, within the 3-year slack, should NOT be flagged.
        c = _base_candidate(
            profile={"years_of_experience": 6.0, "current_title": "AI Engineer"},
            career_history=[{"company": "Acme", "start_date": "2018-01-01", "duration_months": 36}],
        )
        flags = honeypot_flags(c)
        self.assertFalse(any("career_predates_experience" in f for f in flags))

    def test_missing_fields_do_not_crash(self):
        c = {"candidate_id": "CAND_0000001", "profile": {}}
        self.assertEqual(honeypot_flags(c), [])
        self.assertFalse(is_honeypot(c))


class TestRealHoneypotFixtures(unittest.TestCase):
    """
    Regression test using 5 real honeypot candidates pulled directly from the
    full candidates.jsonl release (not synthetic). These specific IDs were
    flagged by our detector during dataset analysis and confirmed to have
    ML/AI-adjacent titles (e.g. "Senior AI Engineer", "NLP Engineer") --
    exactly the kind of profile that would otherwise rank highly. This test
    guards against a future code change silently breaking detection on real
    data, which the synthetic unit tests above can't catch.
    """

    @classmethod
    def setUpClass(cls):
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "real_honeypot_examples.json"
        with open(fixture_path, "r", encoding="utf-8") as f:
            cls.real_honeypots = json.load(f)

    def test_all_real_examples_are_detected(self):
        for c in self.real_honeypots:
            with self.subTest(candidate_id=c["candidate_id"]):
                self.assertTrue(
                    is_honeypot(c),
                    f"{c['candidate_id']} ({c['profile']['current_title']}) "
                    f"should be flagged as a honeypot but was not.",
                )

    def test_real_examples_have_attractive_titles(self):
        # Sanity-check our own assumption: these honeypots should mostly have
        # ML/AI-adjacent titles, confirming they're designed to look like
        # strong matches on the surface.
        ai_titles = {"machine learning engineer", "senior ai engineer",
                     "senior data scientist", "nlp engineer", "ai engineer"}
        matching = sum(
            1 for c in self.real_honeypots
            if c["profile"]["current_title"].lower() in ai_titles
        )
        self.assertGreaterEqual(matching, 4)  # 4 of 5 in our actual fixture


if __name__ == "__main__":
    unittest.main()
