"""Integration tests for the FastAPI application (main.py).

These spin up the real app with FastAPI's TestClient and hit the HTTP
endpoints. The only thing mocked is `create_laptop_crew` so we never make
live LLM calls; the fallback/grounded paths run against the real clean
dataset, exactly as they do in production.
"""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from fastapi.testclient import TestClient

from amazon_crewai import app as main
from amazon_crewai.data_manager import base_model

client = TestClient(main.app)

FAKE_KEY = "test-api-key"


def _crew_returns_markdown():
    """Fake crew whose kickoff() returns a plausible AI report."""
    report = (
        "# Recommendation Report\n"
        "## Top pick\n"
        "| Model | Price | RAM | Processor | OS | Rating |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        "| Apple MacBook | 63,990 | 8GB | Apple | MacOS | 4.8 |\n"
    )
    fake_crew = mock.MagicMock()
    fake_crew.kickoff.return_value = SimpleNamespace(raw=report)
    return fake_crew


def _stream_result(payload=None, headers=None, body=None):
    """POST /api/recommend and parse the SSE stream.

    Returns (status_code, final_result_dict) where final_result_dict is the
    parsed 'result' frame, or the error detail dict for non-200 responses.
    """
    json_body = body if body is not None else {
        "provider": "nvidia",
        "major": "Finance",
        "budget": 250000,
        "brand": "Apple",
        "os_name": "MacOS",
        **(payload or {}),
    }
    resp = client.post("/api/recommend", json=json_body, headers=headers)
    if resp.status_code != 200:
        try:
            return resp.status_code, resp.json()
        except Exception:
            return resp.status_code, {"detail": resp.text}

    result = None
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            try:
                frame = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict) and frame.get("type") == "result":
                result = frame
    return resp.status_code, result


class DataEndpointsTests(unittest.TestCase):
    def test_stats_endpoint(self):
        resp = client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertGreater(body["total_records"], 0)
        self.assertIn("brands", body)

    def test_laptops_endpoint(self):
        resp = client.get("/api/laptops")
        self.assertEqual(resp.status_code, 200)
        rows = resp.json()
        self.assertIsInstance(rows, list)
        self.assertGreater(len(rows), 0)
        self.assertIn("Brand", rows[0])
        self.assertIn("Price", rows[0])
        self.assertIn("Title", rows[0])
        self.assertIn("Product_URL", rows[0])
        self.assertIn("Image_URL", rows[0])

    def test_assets_served_for_laptop_images(self):
        # Pick a laptop with a local image path and confirm it is served.
        resp = client.get("/api/laptops")
        rows = resp.json()
        image_row = next((r for r in rows if r.get("Image_Path")), None)
        if not image_row:
            self.skipTest("No laptop row has a local image path.")
        img_resp = client.get(image_row["Image_Path"])
        self.assertEqual(img_resp.status_code, 200)
        self.assertGreater(len(img_resp.content), 100)

    def test_clean_endpoint(self):
        resp = client.post("/api/clean")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "success")
        self.assertGreater(body["clean_count"], 0)

    def test_root_serves_frontend(self):
        resp = client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Find Your Perfect Laptop", resp.text)


class RecommendEndpointTests(unittest.TestCase):
    VALID_BODY = {
        "provider": "nvidia",
        "major": "Finance",
        "budget": 250000,
        "brand": "Apple",
        "os_name": "MacOS",
    }

    def test_success_path_returns_ai_report(self):
        with mock.patch("amazon_crewai.app.create_laptop_crew", return_value=_crew_returns_markdown()):
            status_code, result = _stream_result(headers={"X-Api-Key": FAKE_KEY})
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "success")
        self.assertIn("Apple MacBook", result["recommendation"])

    def test_local_provider_needs_no_api_key_and_skips_llm(self):
        # Local mode must work with no key and never try to build an LLM.
        with mock.patch("amazon_crewai.app.create_laptop_crew") as create_crew:
            status_code, result = _stream_result(
                payload={"provider": "local"}, headers={"X-Api-Key": ""}
            )
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["fallback_reason"], "local")
        self.assertGreater(len(result["summary"]["picks"]), 0)
        create_crew.assert_not_called()

    def test_crew_none_falls_back(self):
        with mock.patch("amazon_crewai.app.create_laptop_crew", return_value=None):
            status_code, result = _stream_result(headers={"X-Api-Key": FAKE_KEY})
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "fallback")
        self.assertIsNotNone(result["summary"])
        self.assertIn("warning", result)
        self.assertGreater(len(result["recommendation"]), 100)
        # Overview picks now carry the product title + links.
        for pick in result["summary"]["picks"]:
            self.assertIn("title", pick)
            self.assertIn("product_url", pick)
            self.assertIn("image_path", pick)
            self.assertIsInstance(pick.get("colors"), list)

    def test_fallback_picks_are_distinct_models(self):
        # Apple + 250k budget: fallback must never list the same colour
        # SKU of a model multiple times; picks are distinct base models.
        with mock.patch("amazon_crewai.app.create_laptop_crew", return_value=None):
            status_code, result = _stream_result(headers={"X-Api-Key": FAKE_KEY})
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "fallback")
        titles = [p["title"] for p in result["summary"]["picks"]]
        self.assertGreaterEqual(len(titles), 1)
        bases = [base_model(t) for t in titles]
        self.assertEqual(len(set(bases)), len(bases))

    def test_ai_no_match_result_falls_back(self):
        fake_crew = mock.MagicMock()
        fake_crew.kickoff.return_value = SimpleNamespace(
            raw="No laptops found matching the criteria of Apple brand, MacOS OS, "
                "and a maximum budget of 100,000 INR."
        )
        with mock.patch("amazon_crewai.app.create_laptop_crew", return_value=fake_crew):
            status_code, result = _stream_result(headers={"X-Api-Key": FAKE_KEY})
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["fallback_reason"], "no_match")
        self.assertIn("summary", result)
        # Grounded picks must exist, never the hallucinated empty report.
        self.assertGreater(len(result["summary"]["picks"]), 0)

    def test_exception_during_kickoff_falls_back(self):
        def boom(api_key, provider, user_profile, model=None, step_callback=None, task_callback=None):
            raise RuntimeError("simulated LLM outage")

        with mock.patch("amazon_crewai.app.create_laptop_crew", side_effect=boom):
            status_code, result = _stream_result(headers={"X-Api-Key": FAKE_KEY})
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["fallback_reason"], "ai_error")
        self.assertIn("simulated LLM outage", result["details"])

    def test_missing_api_key_returns_local_fallback(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": ""}, clear=False):
            with mock.patch("amazon_crewai.app.create_laptop_crew") as m:
                status_code, result = _stream_result()
        self.assertEqual(status_code, 200)
        self.assertEqual(result["status"], "fallback")
        self.assertEqual(result["fallback_reason"], "missing_api_key")
        self.assertIn("local laptop database", result["warning"])
        self.assertGreater(len(result["recommendation"]), 100)
        self.assertGreater(len(result["summary"]["picks"]), 0)
        m.assert_not_called()

    def test_header_api_key_is_used(self):
        with mock.patch("amazon_crewai.app.create_laptop_crew", return_value=_crew_returns_markdown()) as m:
            _stream_result(headers={"X-Api-Key": FAKE_KEY})
        m.assert_called_once()
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs["api_key"], FAKE_KEY)

    def test_invalid_body_returns_422(self):
        resp = client.post(
            "/api/recommend",
            json={"provider": "nvidia", "major": "Finance"},  # missing budget
            headers={"X-Api-Key": FAKE_KEY},
        )
        self.assertEqual(resp.status_code, 422)

    def test_profile_dict_built_from_request(self):
        with mock.patch("amazon_crewai.app.create_laptop_crew", return_value=_crew_returns_markdown()) as m:
            _stream_result(headers={"X-Api-Key": FAKE_KEY})
        profile = m.call_args.kwargs["user_profile"]
        self.assertEqual(profile["major"], "Finance")
        self.assertEqual(profile["brand"], "Apple")
        self.assertEqual(profile["os"], "MacOS")
        self.assertEqual(profile["budget"], 250000)


if __name__ == "__main__":
    unittest.main()
