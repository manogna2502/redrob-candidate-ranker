"""
Tests for deterministic feature extraction (src/redrob_ranker/features.py).

Run with: python3 -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from redrob_ranker import features as feat


class TestTitleTier(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(feat.title_tier("Senior AI Engineer"), 3)
        self.assertEqual(feat.title_tier("senior ai engineer"), 3)  # case insensitive

    def test_unknown_defaults_to_zero(self):
        self.assertEqual(feat.title_tier("Mechanical Engineer"), 0)
        self.assertEqual(feat.title_tier(""), 0)


class TestExperienceFit(unittest.TestCase):
    def test_inside_band_is_perfect(self):
        self.assertEqual(feat.experience_fit_score(7.0), 1.0)
        self.assertEqual(feat.experience_fit_score(5.0), 1.0)
        self.assertEqual(feat.experience_fit_score(9.0), 1.0)

    def test_penalizes_outside_band(self):
        below = feat.experience_fit_score(2.0)
        above = feat.experience_fit_score(15.0)
        self.assertLess(below, 1.0)
        self.assertLess(above, 1.0)
        self.assertGreaterEqual(below, 0.0)
        self.assertGreaterEqual(above, 0.0)


class TestSkillsMatch(unittest.TestCase):
    def test_rewards_must_have_coverage(self):
        candidate = {
            "skills": [
                {"name": "Pinecone", "proficiency": "advanced", "duration_months": 24},
                {"name": "Python", "proficiency": "expert", "duration_months": 60},
            ],
            "redrob_signals": {"skill_assessment_scores": {"Pinecone": 80, "Python": 90}},
        }
        score, debug = feat.skills_match_score(candidate)
        self.assertGreater(score, 0)
        self.assertGreaterEqual(debug["groups_covered"], 2)

    def test_discounts_claim_vs_assessment_gap(self):
        candidate_trusted = {
            "skills": [{"name": "Pinecone", "proficiency": "expert", "duration_months": 24}],
            "redrob_signals": {"skill_assessment_scores": {"Pinecone": 90}},
        }
        candidate_untrusted = {
            "skills": [{"name": "Pinecone", "proficiency": "expert", "duration_months": 24}],
            "redrob_signals": {"skill_assessment_scores": {"Pinecone": 20}},
        }
        score_trusted, _ = feat.skills_match_score(candidate_trusted)
        score_untrusted, _ = feat.skills_match_score(candidate_untrusted)
        self.assertGreater(score_trusted, score_untrusted)


class TestCareerTrajectory(unittest.TestCase):
    def test_consulting_only_detection(self):
        consulting_candidate = {
            "career_history": [
                {"company": "TCS", "industry": "IT Services"},
                {"company": "Infosys", "industry": "IT Services"},
            ]
        }
        mixed_candidate = {
            "career_history": [
                {"company": "TCS", "industry": "IT Services"},
                {"company": "Flipkart", "industry": "E-commerce"},
            ]
        }
        self.assertTrue(feat.is_consulting_only(consulting_candidate))
        self.assertFalse(feat.is_consulting_only(mixed_candidate))

    def test_title_chaser_detection(self):
        chaser = {
            "career_history": [
                {"duration_months": 12}, {"duration_months": 14},
                {"duration_months": 10}, {"duration_months": 16},
            ]
        }
        stable = {
            "career_history": [
                {"duration_months": 36}, {"duration_months": 48},
            ]
        }
        self.assertTrue(feat.is_title_chaser(chaser))
        self.assertFalse(feat.is_title_chaser(stable))

    def test_framework_tourist_penalty_only_without_infra_depth(self):
        tourist = {"skills": [{"name": "LangChain", "proficiency": "advanced", "duration_months": 6}]}
        non_tourist = {
            "skills": [
                {"name": "LangChain", "proficiency": "advanced", "duration_months": 6},
                {"name": "Pinecone", "proficiency": "advanced", "duration_months": 36},
            ]
        }
        self.assertGreater(feat.framework_tourist_penalty(tourist), 0)
        self.assertEqual(feat.framework_tourist_penalty(non_tourist), 0)


class TestLocationFit(unittest.TestCase):
    def test_location_tiers_ordered_correctly(self):
        pune_candidate = {"profile": {"location": "Pune, Maharashtra", "country": "India"}, "redrob_signals": {}}
        other_india = {"profile": {"location": "Bhopal, MP", "country": "India"}, "redrob_signals": {}}
        non_india = {"profile": {"location": "London", "country": "UK"}, "redrob_signals": {"willing_to_relocate": False}}

        pune_score, _ = feat.location_fit_score(pune_candidate)
        other_score, _ = feat.location_fit_score(other_india)
        non_india_score, _ = feat.location_fit_score(non_india)

        self.assertGreater(pune_score, other_score)
        self.assertGreater(other_score, non_india_score)


class TestBehavioralModifier(unittest.TestCase):
    def test_penalizes_inactivity(self):
        today = date(2026, 6, 26)
        active = {"redrob_signals": {"last_active_date": "2026-06-20", "open_to_work_flag": True}}
        inactive = {"redrob_signals": {"last_active_date": "2025-01-01", "open_to_work_flag": True}}

        mult_active, _ = feat.behavioral_modifier(active, today)
        mult_inactive, _ = feat.behavioral_modifier(inactive, today)

        self.assertGreater(mult_active, mult_inactive)

    def test_bounded_at_floor(self):
        today = date(2026, 6, 26)
        worst_case = {
            "redrob_signals": {
                "last_active_date": "2020-01-01",
                "open_to_work_flag": False,
                "recruiter_response_rate": 0.0,
                "interview_completion_rate": 0.0,
            }
        }
        mult, _ = feat.behavioral_modifier(worst_case, today)
        self.assertGreaterEqual(mult, 0.50)  # never below the floor


if __name__ == "__main__":
    unittest.main()
