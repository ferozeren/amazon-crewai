"""Scrape Amazon India laptop listings into data/raw_data/amazon_laptops.csv."""

import argparse
import os
import re
import sys
import time
from urllib.parse import unquote, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .paths import RAW_CSV

BASE_URL = "https://www.amazon.in"
SEARCH_URL = BASE_URL + "/s"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Brands already supported by the frontend form; each gets its own search
# query so we can reach well beyond the ~20 result pages Amazon allows for a
# single keyword. Expanded to double catalog size.
BRANDS = [
    "HP",
    "Lenovo",
    "Dell",
    "ASUS",
    "Acer",
    "MSI",
    "Apple",
    "Samsung",
    "Xiaomi",
    "Alienware",
    "Infinix",
    "Primebook",
    "Thomson",
    "Honor",
    "Realme",
    "Gigabyte",
    "Jiobook",
    "Chuwi",
    "Avita",
    "Nokia",
]

# Known-brand keywords matched anywhere in a title (word-boundary safe), used
# for brand inference instead of the naive "first word is the brand" rule.
BRAND_KEYWORDS = {
    "hp": "HP",
    "lenovo": "Lenovo",
    "dell": "Dell",
    "asus": "ASUS",
    "acer": "Acer",
    "msi": "MSI",
    "apple": "Apple",
    "macbook": "Apple",
    "samsung": "Samsung",
    "xiaomi": "Xiaomi",
    "razer": "Razer",
    "alienware": "Alienware",
    "microsoft": "Microsoft",
    "primebook": "Primebook",
    "thomson": "Thomson",
    "infinix": "Infinix",
    "honor": "Honor",
    "realme": "Realme",
    "gigabyte": "Gigabyte",
    "aorus": "Gigabyte",
    "jiobook": "Jiobook",
    "chuwi": "Chuwi",
    "avita": "Avita",
    "nokia": "Nokia",
    "jio": "Jiobook",
}

QUERIES = ["laptops", "gaming laptop", "business laptop", "student laptop", "ultrabook", "chromebook"] + [f"{b} laptop" for b in BRANDS]


def _make_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


def _fetch(session, url, params, retries=3, delay=1.5):
    """Fetch a page with retries + backoff. Returns response or None."""
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response
            print(f"  HTTP {response.status_code} (attempt {attempt})", flush=True)
        except requests.RequestException as e:
            print(f"  Request error: {e} (attempt {attempt})", flush=True)
        time.sleep(delay * attempt)
    return None


def _infer_brand(title):
    """Return the canonical brand for a title, or 'Unknown'."""
    # Only check first 40 chars for brand to avoid mislabeling accessories that mention MacBook as compatibility
    t = title.lower()[:40]
    for keyword, brand in BRAND_KEYWORDS.items():
        if re.search(r"(^|[^a-z0-9])" + keyword + r"([^a-z0-9]|$)", t):
            return brand
    # Fallback to first word if no keyword in prefix
    match = re.search(r"^([A-Za-z]+)", title)
    return match.group(1) if match else "Unknown"


def _extract_product_url(a_tag, asin):
    """Normalise the Amazon product href into an absolute product URL.

    Falls back to ``https://www.amazon.in/dp/{asin}`` when no usable link is
    found so the product still gets a clickable (correct) URL.
    """
    href = a_tag.get("href") if a_tag is not None else None
    if href:
        # Ad placements wrap the real URL in a `url=` query parameter.
        if "/sspa/click" in href:
            m = re.search(r"[?&]url=([^&]+)", href)
            if m:
                href = unquote(m.group(1))
        if href.startswith("/"):
            href = urljoin(BASE_URL, href)
        m = re.search(r"(https://[^/]+)?/dp/([A-Za-z0-9]+)", href)
        if m:
            domain = m.group(1) or BASE_URL
            return f"{domain}/dp/{m.group(2)}"
    if asin:
        return f"{BASE_URL}/dp/{asin}"
    return "N/A"


def _extract_image_url(container, asin):
    """Extract the real product thumbnail.

    Prefers ``img.s-image`` (the product photo). The image id embedded in the
    URL usually contains the ASIN; when we can't find a usable image, fall
    back to the ASIN-based media URL so the product still has an image.
    """
    img = container.find("img", class_="s-image")
    if img is None:
        img = container.find("img")
    src = ""
    if img is not None:
        src = img.get("src") or img.get("data-src") or ""
    if not src or src.startswith("data:"):
        src = ""
    if not src and asin:
        src = f"https://m.media-amazon.com/images/I/{asin}._AC_UY500_.jpg"
    if not src:
        return "N/A"
    # Request a slightly larger thumbnail for crisp cards.
    src = src.replace("_AC_UY218_", "_AC_UY500_").replace("_AC_UY400_", "_AC_UY500_")
    return src


def _parse_container(container):
    """Extract all fields from one search-result container. Returns dict or None."""
    asin = (container.get("data-asin") or "").strip()
    h2_tag = container.find("h2")
    if not h2_tag:
        return None
    title = h2_tag.text.strip()
    if len(title) < 10 or title in ["Results", "Trending now"]:
        return None

    # Price
    price_tag = container.find("span", class_="a-price-whole")
    price = price_tag.text.strip() if price_tag else "N/A"

    # Brand
    brand = _infer_brand(title)

    # RAM
    match = re.search(
        r"(\d+GB|RAM\s*(\d+GB)?|DDR\d?\s*(\d+GB)?|LPDDR\d?\s*(\d+GB)?)",
        title,
        re.IGNORECASE,
    )
    ram = match.group(1) if match else "N/A"

    # SSD storage
    match = re.search(r"(\d+)\s*(GB|TB)\s*(?:SSD|Storage|HDD)", title, re.IGNORECASE)
    ssd_storage = f"{match.group(1)}{match.group(2)}" if match else "N/A"

    # Color
    match = re.search(
        r"\b(Black|White|Silver|Gray|Grey|Red|Blue|Green|Yellow|Pink|Purple|Gold|Bronze|Rose Gold|Indigo|Glacier)\b",
        title,
        re.IGNORECASE,
    )
    color = match.group(1) if match else "N/A"

    # Processor
    processor = "N/A"
    match = re.search(r"Apple\s+M(\d+)", title, re.IGNORECASE)
    if match:
        processor = f"Apple M{match.group(1)}"
    else:
        match = re.search(r"Apple\s+A(\d+)", title, re.IGNORECASE)
        if match:
            processor = f"Apple A{match.group(1)}"
        else:
            for keyword in ["Intel", "AMD", "Snapdragon", "MediaTek", "Celeron"]:
                if re.search(rf"\b{keyword}\b", title, re.IGNORECASE):
                    processor = keyword
                    break

    # OS
    os_name = "N/A"
    if re.search(r"\b(MacBook|macOS|Mac OS)\b", title, re.IGNORECASE):
        os_name = "MacOS"
    elif re.search(r"\bApple\s+[MA]\d+\b", title, re.IGNORECASE):
        os_name = "MacOS"
    elif re.search(r"\b(Windows|Win\s*11|Win11|Win\s*10|Win10)\b", title, re.IGNORECASE):
        os_name = "Windows"
    else:
        for keyword in ["Linux", "Chrome OS", "Android", "iOS"]:
            if re.search(rf"\b{keyword}\b", title, re.IGNORECASE):
                os_name = keyword
                break

    # Rating
    rating = "N/A"
    rating_span = (
        container.find("span", class_=lambda x: x and "i-star" in str(x).lower() if x else False)
        or container.find(
            "span",
            {"aria-label": lambda x: x and "out of 5" in str(x).lower() if x else False},
        )
        or container.find("i", class_=lambda x: x and "a-icon-star" in str(x).lower() if x else False)
    )
    if rating_span:
        aria_label = rating_span.get("aria-label")
        if aria_label:
            match = re.search(r"(\d+\.?\d*)\s*out of 5", aria_label)
            if match:
                rating = match.group(1)
        else:
            rating_text = rating_span.text.strip()
            match = re.search(r"(\d+\.?\d*)", rating_text)
            if match:
                rating = match.group(1)

    # Product URL + image URL
    link_tag = container.select_one("h2 a") or container.select_one(
        "a.a-link-normal.s-line-clamp-2"
    )
    product_url = _extract_product_url(link_tag, asin)
    image_url = _extract_image_url(container, asin)

    return {
        "title": title,
        "price": price,
        "brand": brand,
        "ram": ram,
        "ssd_storage": ssd_storage,
        "color": color,
        "processor": processor,
        "os": os_name,
        "rating": rating,
        "product_url": product_url,
        "image_url": image_url,
        "asin": asin,
    }


def scrape_queries(queries=None, pages=10, out_path=str(RAW_CSV)):
    session = _make_session()
    data = []
    queries = queries or QUERIES

    for query in queries:
        print(f"\n=== Query: {query} ===", flush=True)
        for page in range(1, pages + 1):
            response = _fetch(session, SEARCH_URL, {"k": query, "page": page})
            if response is None:
                print(f"  Skipping {query} page {page} (could not fetch)", flush=True)
                continue
            soup = BeautifulSoup(response.content, "html.parser")
            containers = soup.find_all("div", class_="s-result-item")
            page_count = 0
            for container in containers:
                if not container.get("data-asin"):
                    continue
                item = _parse_container(container)
                if item:
                    data.append(item)
                    page_count += 1
            print(f"  {query} page {page}: {page_count} items", flush=True)
            time.sleep(0.8)  # Be polite to Amazon between pages.

    if not data:
        print("No data scraped. Check network/selectors.", file=sys.stderr)
        return None

    df = pd.DataFrame(data)
    df = df.drop_duplicates(subset=["asin"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} unique rows -> {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Amazon India laptop listings.")
    parser.add_argument("--pages", type=int, default=10, help="Pages to scrape per query")
    parser.add_argument(
        "--brands",
        nargs="*",
        default=None,
        help="Brands to query (plus the general 'laptops' query). Defaults to all.",
    )
    parser.add_argument("--out", default=str(RAW_CSV), help="Output CSV path")
    args = parser.parse_args()

    if args.brands:
        qs = ["laptops"] + [f"{b} laptop" for b in args.brands]
    else:
        qs = QUERIES
    scrape_queries(queries=qs, pages=args.pages, out_path=args.out)
