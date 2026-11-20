"""Unit tests for the FastAPI helpers in main.py (no HTTP calls)."""

import unittest
from types import SimpleNamespace

from amazon_crewai.app import (
    NO_MATCH_PATTERNS,
    _describe_step,
    _describe_task,
    _indicates_no_match,
    _sse,
    RecommendRequest,
)
from pydantic import ValidationError


class IndicatesNoMatchTests(unittest.TestCase):
    def test_detects_no_match_phrases(self):
        cases = [
            "No laptops found matching the criteria",
            "There are no matching laptops available",
            "no laptops match your budget",
            "Sorry, no matching models were found",
            "We did not find any laptops",
            "could not find any laptop that fits",
        ]
        for text in cases:
            self.assertTrue(_indicates_no_match(text), f"should detect: {text}")

    def test_ignores_normal_reports(self):
        cases = [
            "Here are the top 3 laptops for you",
            "Recommendation report for a Finance student",
            "",
            None,
        ]
        for text in cases:
            self.assertFalse(_indicates_no_match(text), f"should ignore: {text!r}")

    def test_case_insensitive(self):
        self.assertTrue(_indicates_no_match("No LAPTOPS FOUND!"))

    def test_patterns_list_non_empty_and_lowercase(self):
        self.assertTrue(NO_MATCH_PATTERNS)
        for pattern in NO_MATCH_PATTERNS:
            self.assertEqual(pattern, pattern.lower())


class RecommendRequestTests(unittest.TestCase):
    def test_valid_payload(self):
        req = RecommendRequest(
            provider="nvidia", major="Data Science", budget=80000, brand="Apple"
        )
        self.assertEqual(req.provider, "nvidia")
        self.assertEqual(req.brand, "Apple")
        self.assertIsNone(req.ram)

    def test_missing_required_fields_raises(self):
        with self.assertRaises(ValidationError):
            RecommendRequest(provider="nvidia", major="Data Science")

    def test_defaults(self):
        req = RecommendRequest(provider="gemini", major="MBA", budget=100000)
        self.assertEqual(req.brand, "Any")
        self.assertEqual(req.os_name, "Any")
        self.assertEqual(req.details, "")


class SseHelpersTests(unittest.TestCase):
    def test_sse_frame_wraps_json(self):
        frame = _sse({"type": "status", "message": "hi"})
        self.assertTrue(frame.startswith("data: "))
        self.assertIn('"type": "status"', frame)
        self.assertTrue(frame.endswith("\n\n"))

    def test_describe_step_for_tool_action(self):
        step = SimpleNamespace(tool="query_laptops_tool", log="")
        message = _describe_step(step)
        self.assertIn("Searching the laptop catalog", message)
        self.assertLessEqual(len(message), 72)

    def test_describe_step_for_stats_tool(self):
        message = _describe_step(SimpleNamespace(tool="get_dataset_stats_tool", log=""))
        self.assertIn("Reviewing the dataset stats", message)

    def test_describe_step_from_log_is_short(self):
        long_log = "I should call the database to find laptops. " * 10
        step = SimpleNamespace(tool=None, log=long_log)
        message = _describe_step(step)
        self.assertIn("find laptops", message)
        self.assertLessEqual(len(message), 72)

    def test_describe_step_fallback(self):
        message = _describe_step(SimpleNamespace(tool=None, log=""))
        self.assertTrue(message)

    def test_describe_task_maps_search(self):
        output = SimpleNamespace(name="Search the database", agent=SimpleNamespace(role="Laptop Database Specialist"))
        message = _describe_task(output)
        self.assertIn("Searching the laptop catalog", message)
        self.assertIn("Laptop Database Specialist", message)

    def test_describe_task_maps_recommendation(self):
        output = SimpleNamespace(name="Recommendation task", agent=None)
        message = _describe_task(output)
        self.assertIn("Writing your personalized recommendation", message)


if __name__ == "__main__":
    unittest.main()
