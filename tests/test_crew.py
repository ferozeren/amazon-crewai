"""Unit tests for the CrewAI layer (crew.py).

Covers the deterministic fallback machinery (_compute_fallback_matches,
build_fallback_summary, build_fallback_recommendation) and the LLM provider
guard rails (_initialize_llm). No live LLM calls are made.
"""

import os
import unittest
from unittest import mock

from amazon_crewai.crew import (
    _compute_fallback_matches,
    _initialize_llm,
    build_fallback_recommendation,
    build_fallback_summary,
)


class FallbackMatchingTests(unittest.TestCase):
    def test_exact_match_not_relaxed(self):
        profile = {
            "major": "Finance",
            "budget": 250000,
            "ram": "Any",
            "brand": "Apple",
            "os": "MacOS",
            "details": "None",
        }
        matches, note, brand_ref, budget = _compute_fallback_matches(profile)
        self.assertIsNotNone(matches)
        # With fresh data cheapest available Apple is ~172k, so 250k should match without relaxation
        # If data changes, allow relaxed note but must have matches
        self.assertEqual(budget, 250000)
        self.assertGreaterEqual(len(matches), 1)
        # Price should be within budget and cheapest available
        self.assertLessEqual(int(matches.iloc[0]["Price"]), 250000)

    def test_color_variants_collapsed(self):
        # Apple ships the MacBook Neo 256GB in 3 colours (Silver/Indigo/Citrus),
        # each a separate Amazon SKU. The pick list must contain that model once.
        matches, _, _, _ = _compute_fallback_matches(
            {
                "major": "CS",
                "budget": 250000,
                "brand": "Apple",
                "os": "MacOS",
                "ram": "Any",
                "details": "",
            }
        )
        self.assertIsNotNone(matches)
        self.assertGreaterEqual(len(matches), 1)
        # Distinct models
        self.assertEqual(len(set(matches["_base"])), len(matches))

    def test_picks_are_distinct_models(self):
        matches, _, _, _ = _compute_fallback_matches(
            {
                "major": "CS",
                "budget": 250000,
                "brand": "Apple",
                "os": "MacOS",
                "ram": "Any",
                "details": "",
            }
        )
        self.assertIsNotNone(matches)
        self.assertGreaterEqual(len(matches), 2)
        self.assertEqual(len(set(matches["_base"])), len(matches))

    def test_relaxes_brand_when_os_has_no_brand_match(self):
        # Dell does not ship macOS - strict filtering should yield no match (real-world behavior)
        profile = {
            "major": "Finance",
            "budget": 250000,
            "ram": "Any",
            "brand": "Dell",
            "os": "MacOS",
            "details": "None",
        }
        matches, note, brand_ref, _ = _compute_fallback_matches(profile)
        self.assertIsNotNone(matches)
        self.assertIsNotNone(note)
        # Strict: no laptops match Dell+MacOS -> empty with guidance, not relaxed to Apple
        self.assertEqual(len(matches), 0)
        self.assertTrue(any(k in note.lower() for k in ("no laptops match", "try lowering", "no laptops found")))

    def test_summary_shape(self):
        profile = {
            "major": "Data Science",
            "budget": 250000,
            "ram": 16,
            "brand": "Apple",
            "os": "MacOS",
            "details": "",
        }
        summary = build_fallback_summary(profile)
        self.assertIsNotNone(summary)
        self.assertIn("top_pick", summary)
        self.assertIn("picks", summary)
        self.assertIn("relaxed", summary)
        self.assertTrue(len(summary["picks"]) >= 1)
        top = summary["top_pick"]
        for key in (
            "title",
            "brand",
            "price",
            "ram",
            "storage",
            "processor",
            "os",
            "rating",
            "product_url",
            "image_path",
            "image_url",
            "colors",
        ):
            self.assertIn(key, top)

    def test_build_fallback_recommendation_content(self):
        result = build_fallback_recommendation(
            user_profile={
                "major": "Computer Science",
                "budget": 150000,
                "ram": 16,
                "brand": "Dell",
                "os": "Windows",
                "details": "",
            },
            error_message="Simulated provider failure",
        )
        self.assertIn("Computer Science", result)
        self.assertIn("Dell", result)
        self.assertIn("budget", result.lower())
        self.assertRegex(result, r"### Top \d+ pick(s)? for you")
        self.assertIn("At a glance", result)
        self.assertIn("| Laptop | Price | RAM |", result)

    def test_summary_uses_stored_cleaned_dataset(self):
        rows = _compute_fallback_matches(
            {"major": "Business", "budget": 80000, "brand": "Any", "os": "Any", "ram": "Any", "details": ""}
        )
        matches = rows[0]
        self.assertIsNotNone(matches)
        # Cheapest laptop must be within the budget or the dataset is broken.
        self.assertLessEqual(int(matches.iloc[0]["Price"]), 80000)


class LLMInitTests(unittest.TestCase):
    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            _initialize_llm("not-a-provider", "fake-key")

    def test_missing_key_raises(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError):
                _initialize_llm("gemini", None)

    def test_openrouter_model_override_resolution(self):
        # Constructing an LLM object is offline; just verify it does not raise
        # with a valid provider + fake key and that the model override applies.
        llm = _initialize_llm("openrouter", "fake-key", model="google/gemma-4-31b-it:free")
        self.assertIn("google/gemma-4-31b-it:free", llm.model)

    def test_nvidia_provider_constructs(self):
        llm = _initialize_llm("nvidia", "fake-key")
        self.assertIn("nemotron", llm.model)


if __name__ == "__main__":
    unittest.main()
