"""Unit tests for the data cleaning pipeline (data_manager.py).

The individual cleaning helpers (clean_price, clean_brand, ...) are nested
inside `clean_dataset()`, so we test them through the public pipeline on a
small, fully controlled synthetic CSV. This gives us precise assertions
without depending on the real scraped dataset.
"""

import os
import shutil
import tempfile
import unittest

from amazon_crewai.data_manager import (
    base_model,
    clean_dataset,
    get_dataset_stats,
    get_laptops_data,
    product_color,
)

RAW_CSV = """Price,Brand,RAM,SSD_Storage,Color,Processor,OS,Rating,Title,Product_URL,Image_URL
"₹63,990","apple","8 GB",256,"Silver",,MacOS,4.8,"Apple MacBook Air 13 inch","https://www.amazon.in/dp/ABCD1","https://img.test/1.jpg"
"1,10,300","dell","16 GB",1.0,"Black","ryzen 5","Windows 11",4.5,"Dell Inspiron 15, AMD Ryzen 5, 15.6 inch FHD","https://www.amazon.in/dp/ABCD2","https://img.test/2.jpg"
2000,"Table",8,128,"Wood",,Windows,2.0,"Table Accessory","https://www.amazon.in/dp/ABCD3","https://img.test/3.jpg"
1500,"BKN",8,128,"White",,Windows,1.0,"BKN Accessory","https://www.amazon.in/dp/ABCD4","https://img.test/4.jpg"
42000,"hp",8,"512GB","Gray","intel core i5","Windows 10",4.2,"HP 15 Laptop, Intel Core i5-1235U, 15.6 inch FHD","https://www.amazon.in/dp/ABCD5","https://img.test/5.jpg"
52000,"msi",8,512,"Black","intel",Windows,4.0,"MSI Modern 14, Intel Core i5-1335U","https://www.amazon.in/dp/ABCD6","https://img.test/6.jpg"
45000,"asus",16,512,"Blue","amd",Windows,4.1,"ASUS Vivobook 15, AMD Ryzen 5 7530U","https://www.amazon.in/dp/ABCD7","https://img.test/7.jpg"
30000,"dellmodel",8,256,"Black",,Windows,3.0,"DellModel Junk","https://www.amazon.in/dp/ABCD8","https://img.test/8.jpg"
"1,20,000","lenovo",32,1024,"Silver","intel",,4.4,"Lenovo ThinkPad E14, Intel Core i5, 14 inch","https://www.amazon.in/dp/ABCD9","https://img.test/9.jpg"
"1,20,000","lenovo",32,1024,"Silver","intel",,4.4,"Lenovo ThinkPad E14, Intel Core i5, 14 inch","https://www.amazon.in/dp/ABCD9","https://img.test/9.jpg"
"""


class CleanDatasetTests(unittest.TestCase):
    """Verifies price/brand/ram/storage/processor/os/rating normalisation,
    plus the dropping of accessories and bad brand names."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="laptop_tests_")
        cls.raw_path = os.path.join(cls.tmpdir, "raw.csv")
        cls.clean_path = os.path.join(cls.tmpdir, "clean.csv")
        with open(cls.raw_path, "w") as f:
            f.write(RAW_CSV)
        cls.res = clean_dataset(cls.raw_path, cls.clean_path)
        cls.df = __import__("pandas").read_csv(cls.clean_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_clean_succeeds(self):
        self.assertEqual(self.res["status"], "success")

    def test_accessories_and_bad_brands_dropped(self):
        # 10 input rows: "Table" + "BKN" are accessories (<10k), "dellmodel"
        # is a junk brand, and the duplicate Lenovo row is deduplicated
        # (same Product_URL) -> 6 valid laptops remain.
        self.assertEqual(self.res["clean_count"], 6)
        self.assertEqual(len(self.df), 6)

    def test_title_kept_and_cleaned(self):
        titles = set(self.df["Title"])
        self.assertIn("Apple MacBook Air 13 inch", titles)
        self.assertIn("HP 15 Laptop, Intel Core i5-1235U, 15.6 inch FHD", titles)
        # Every row has a non-empty Title.
        self.assertEqual(self.df["Title"].str.len().min() > 0, True)

    def test_product_and_image_urls_kept(self):
        apple = self.df[self.df["Brand"] == "Apple"].iloc[0]
        self.assertEqual(apple["Product_URL"], "https://www.amazon.in/dp/ABCD1")
        self.assertEqual(apple["Image_URL"], "https://img.test/1.jpg")

    def test_duplicate_products_deduplicated(self):
        # The Lenovo row was repeated in the raw data; only one survives.
        lenovo = self.df[self.df["Brand"] == "Lenovo"]
        self.assertEqual(len(lenovo), 1)

    def test_price_cleaning(self):
        prices = sorted(self.df["Price"].tolist())
        self.assertEqual(prices, [42000, 45000, 52000, 63990, 110300, 120000])

    def test_brand_normalisation(self):
        self.assertSetEqual(set(self.df["Brand"]), {"Apple", "Dell", "HP", "MSI", "ASUS", "Lenovo"})

    def test_ram_cleaning(self):
        self.assertEqual(sorted(self.df["RAM"]), [8, 8, 8, 16, 16, 32])

    def test_storage_cleaning(self):
        # "1.0" TB -> 1024 GB
        dell = self.df[self.df["Brand"] == "Dell"].iloc[0]
        self.assertEqual(int(dell["SSD_Storage"]), 1024)
        # "512GB" string -> 512
        hp = self.df[self.df["Brand"] == "HP"].iloc[0]
        self.assertEqual(int(hp["SSD_Storage"]), 512)

    def test_apple_processor_brand_aware_default(self):
        # The famous bug: Apple rows with NaN processor must be "Apple",
        # not the generic "Intel" default.
        apple = self.df[self.df["Brand"] == "Apple"].iloc[0]
        self.assertEqual(apple["Processor"], "Apple")

    def test_non_apple_nan_processor_defaults_to_intel(self):
        lenovo = self.df[self.df["Brand"] == "Lenovo"].iloc[0]
        self.assertEqual(lenovo["Processor"], "Intel")

    def test_amd_detection(self):
        dell = self.df[self.df["Brand"] == "Dell"].iloc[0]
        self.assertEqual(dell["Processor"], "AMD")

    def test_os_normalisation(self):
        apple = self.df[self.df["Brand"] == "Apple"].iloc[0]
        self.assertEqual(apple["OS"], "MacOS")
        lenovo = self.df[self.df["Brand"] == "Lenovo"].iloc[0]
        self.assertEqual(lenovo["OS"], "Windows")

    def test_stats_from_cleaned_file(self):
        stats = get_dataset_stats(self.clean_path)
        self.assertEqual(stats["status"], "success")
        self.assertEqual(stats["total_records"], 6)
        self.assertEqual(stats["min_price"], 42000)
        self.assertEqual(stats["max_price"], 120000)
        self.assertEqual(stats["brands"]["Apple"], 1)

    def test_laptops_data_returns_records(self):
        rows = get_laptops_data(self.clean_path)
        self.assertEqual(len(rows), 6)
        self.assertIn("Brand", rows[0])
        self.assertIn("Price", rows[0])


class BaseModelTests(unittest.TestCase):
    """Colour variants of a product (separate Amazon SKUs) share one model."""

    def test_strips_trailing_color_after_semicolon(self):
        base = base_model("Apple 2026 MacBook Neo 13 inch; Silver", "Silver")
        self.assertEqual(base, "Apple 2026 MacBook Neo 13 inch")

    def test_strips_trailing_color_after_comma(self):
        base = base_model("HP 15, AMD Ryzen 3 7320U, FHD, Anti-Glare, Silver", "Silver")
        self.assertEqual(base, "HP 15, AMD Ryzen 3 7320U, FHD, Anti-Glare")

    def test_color_column_wins_when_title_is_ambiguous(self):
        # Color column says "Indigo" even if the title suffix looks generic.
        base = base_model("Apple 2026 MacBook Neo 13 inch; Indigo", "Indigo")
        self.assertEqual(base, "Apple 2026 MacBook Neo 13 inch")

    def test_unknown_color_column_falls_back_to_color_word(self):
        base = base_model("Apple 2026 MacBook Neo 13 inch; Citrus", "Unknown")
        self.assertEqual(base, "Apple 2026 MacBook Neo 13 inch")

    def test_spec_suffix_not_stripped(self):
        # "; 512GB SSD" is a spec, not a colour — the title must be untouched.
        base = base_model("Lenovo IdeaPad Slim 3, 15.3 inch; 512GB SSD", "Unknown")
        self.assertIn("512GB SSD", base)

    def test_no_trailing_separator_untouched(self):
        base = base_model("MSI Modern 14, Intel Core i5-1335U", "Black")
        self.assertEqual(base, "MSI Modern 14, Intel Core i5-1335U")

    def test_empty_title(self):
        self.assertEqual(base_model("", "Silver"), "")
        self.assertEqual(base_model(None, "Silver"), "")

    def test_product_color_prefers_column(self):
        self.assertEqual(product_color("Any Title", "space gray"), "Space gray")
        self.assertEqual(product_color("Any Title", "unknown"), "Unknown")

    def test_product_color_falls_back_to_title(self):
        self.assertEqual(product_color("Apple 2026 MacBook Neo 13 inch; Citrus", "Unknown"), "Citrus")
        self.assertEqual(product_color("Plain laptop title", "Unknown"), "Unknown")


if __name__ == "__main__":
    unittest.main()
