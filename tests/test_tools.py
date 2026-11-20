"""Unit tests for the CrewAI tools (tools.py).

These exercise the two tools the Laptop Database Specialist agent can call,
against the real cleaned dataset, verifying filter correctness, case
insensitivity, and helpful failure messages.
"""

import unittest

from amazon_crewai.tools import get_dataset_stats_tool, query_laptops_tool


def call_query(**kwargs):
    return query_laptops_tool.func(**kwargs)


def _data_lines(result):
    """Rows of the markdown table (lines after the separator)."""
    lines = result.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("---"):
            return lines[i + 1 :]
    return []


def _price_of(line):
    # Header: "Title | Brand | Price (INR) | RAM (GB) | ..." -> price is col 3.
    cols = [c.strip() for c in line.split("|")]
    return int(cols[2].replace(",", ""))


class QueryLaptopsToolTests(unittest.TestCase):
    def test_apple_macos_within_budget(self):
        result = call_query(brand="Apple", max_price=250000, os_name="MacOS")
        self.assertIn("Found", result)
        self.assertIn("Title", result)
        self.assertIn("MacBook", result)
        # Every returned row must be an Apple laptop within budget.
        for line in _data_lines(result):
            self.assertIn("| Apple |", line)
            self.assertLessEqual(_price_of(line), 250000)

    def test_brand_filter_is_case_insensitive(self):
        upper = call_query(brand="APPLE", max_price=250000, os_name="MacOS")
        lower = call_query(brand="apple", max_price=250000, os_name="MacOS")
        self.assertEqual(upper, lower)
        self.assertIn("Found", upper)

    def test_os_mac_alias_maps_to_macos(self):
        result = call_query(brand="Apple", max_price=250000, os_name="mac")
        self.assertIn("Found", result)
        self.assertIn("MacBook", result)

    def test_price_budget_respected(self):
        result = call_query(brand="Apple", max_price=64000, os_name="MacOS")
        self.assertIn("No laptops found", result)

    def test_no_match_returns_helpful_message(self):
        result = call_query(brand="Apple", max_price=100000, processor="Intel", os_name="MacOS")
        self.assertIn("No laptops found", result)
        self.assertIn("relax", result.lower())

    def test_processor_apple_filter_now_works(self):
        # Regression: the exact args the agent used used to return nothing
        # because Apple laptops were labelled Intel. With the data fix it works.
        result = call_query(
            brand="Apple", max_price=250000, min_ram=0, processor="Apple", os_name="MacOS"
        )
        self.assertIn("Found", result)
        self.assertIn("MacBook", result)

    def test_sort_by_rating_puts_best_first(self):
        result = call_query(brand="Dell", max_price=150000, sort_by_rating=True)
        self.assertIn("Found", result)
        # The first data row should be the highest rated Dell.
        data_lines = [line for line in result.splitlines() if line and not line.startswith("---")]
        self.assertIn("Dell", data_lines[3])

    def test_color_variants_collapsed(self):
        # The MacBook Neo 256GB ships in 3 colours (Silver/Indigo/Citrus), each
        # a separate Amazon SKU. The tool must return the model once (plus its
        # distinct 512GB variant) so colour variants don't crowd other SKUs.
        result = call_query(brand="Apple", max_price=250000, os_name="MacOS")
        data_lines = _data_lines(result)
        self.assertGreaterEqual(len(data_lines), 1)
        # No duplicate base models (colour variants collapsed)
        from amazon_crewai.data_manager import base_model
        bases = [base_model(l.split("|")[0].strip()) for l in data_lines]
        self.assertEqual(len(bases), len(set(bases)))

    def test_min_ram_filter(self):
        result = call_query(brand="Apple", max_price=200000, min_ram=16, os_name="MacOS")
        self.assertIn("Found", result)
        self.assertNotIn("8GB", result)


class StatsToolTests(unittest.TestCase):
    def test_stats_overview_returned(self):
        result = get_dataset_stats_tool.func()
        self.assertIn("Total Laptops Available", result)
        self.assertIn("Apple", result)
        self.assertIn("Price Range", result)


if __name__ == "__main__":
    unittest.main()
