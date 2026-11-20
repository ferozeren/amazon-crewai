import datetime
import json
import os
import re
from urllib.parse import urlparse

import pandas as pd

from .paths import ASSETS_DIR, CLEAN_CSV, METADATA_JSON, MODEL_PATH, RAW_CSV

RAW_PATH = str(RAW_CSV)
CLEAN_PATH = str(CLEAN_CSV)
METADATA_PATH = str(METADATA_JSON)

# Amazon lists each colour as a separate SKU; group them into one machine.
_COLOR_WORDS = re.compile(
    r"^(?:silver|space ?gray|space ?grey|gray|grey|black|blue|indigo|citrus|"
    r"white|gold|rose|red|green|orange|purple|pink|teal|brown|beige|navy|"
    r"midnight|starlight|graphite|titanium|natural|sky blue|sky|obsidian|"
    r"mystic|moonlight|deep blue|pale gold)$",
    re.IGNORECASE,
)


def base_model(title, color=None):
    """Reduce an Amazon product title to its laptop model (no color variant).

    Amazon lists each colour as a separate SKU (distinct ASIN), e.g. the same
    MacBook title ending in "; Silver" / "; Indigo" / "; Citrus". This returns
    the shared base title so colour variants can be grouped into one model.

    A trailing segment is only stripped when it matches the row's ``Color``
    column or a known colour word, so different storage/RAM SKUs are never
    merged (e.g. "; 512GB SSD" is left alone).
    """
    s = str(title or "").strip()
    if not s:
        return s
    match = re.search(r"[;|,]\s*([^;|,]+)$", s)
    if not match:
        return s
    last = match.group(1).strip()
    if not last or len(last) > 30:
        return s
    color_val = str(color or "").strip().lower()
    if color_val and color_val not in ("unknown", "nan", "none"):
        if color_val in last.lower():
            return s[: match.start()].strip().rstrip(";|, ").strip()
    if _COLOR_WORDS.match(last):
        return s[: match.start()].strip().rstrip(";|, ").strip()
    return s


def product_color(title, color=None):
    """Best-effort colour of a product.

    Prefers the ``Color`` column, but falls back to reading the trailing colour
    word from the title (many rows have ``Color == "Unknown"`` while the title
    still ends in "; Citrus").
    """
    c = str(color or "").strip()
    if c and c.lower() not in ("unknown", "nan", "none"):
        return c.capitalize()
    match = re.search(r"[;|,]\s*([^;|,]+)$", str(title or ""))
    if match and _COLOR_WORDS.match(match.group(1).strip()):
        return match.group(1).strip().capitalize()
    return "Unknown"


# ---- GPU extraction -------------------------------------------------------
# Dedicated GPUs are discrete (NVIDIA RTX/GTX/MX, AMD Radeon RX, Arc). Integrated
# covers Intel Iris/UHD/HD, AMD Radeon Graphics/Vega, Snapdragon Adreno, etc.
_DEDICATED_GPU_RE = re.compile(
    r"(rtx\s*\d{3,4}(?:\s*(?:ti|super))?|gtx\s*\d{3,4}(?:\s*ti)?|nvidia\s+geforce\s+rtx|nvidia\s+geforce\s+gtx|radeon\s+rx\s*\d+|arc\s+a\d+|radeon\s+pro|quadro|geforce\s+mx\d+|rtx\s*40\d0|rtx\s*50\d0)",
    re.IGNORECASE,
)
_INTEGRATED_GPU_RE = re.compile(
    r"(iris\s*xe|iris|uhd\s*graphics|hd\s*graphics|intel\s+hd|radeon\s+graphics|amd\s+radeon|vega|adreno|mali|powervr)",
    re.IGNORECASE,
)
_GPU_NAME_RE = re.compile(
    r"(rtx\s*\d{3,4}(?:\s*(?:ti|super))?|gtx\s*\d{3,4}(?:\s*ti)?|radeon\s+rx\s*\d+|arc\s+a\d+|iris\s*xe|uhd\s*graphics|radeon\s+graphics|geforce\s+mx\d+|quadro\s+\w+)",
    re.IGNORECASE,
)


def extract_gpu_info(title: str):
    """Parse GPU info from a laptop title.

    Returns (gpu_name, has_dedicated_gpu):
      - gpu_name: cleaned GPU token like 'RTX 4060', 'Iris Xe', 'Radeon Graphics', or 'Integrated' / 'Unknown'
      - has_dedicated_gpu: True if discrete GPU detected, False if integrated, None if unknown
    """
    t = str(title or "")
    if not t:
        return "Unknown", False
    # Dedicated check first (higher priority)
    if _DEDICATED_GPU_RE.search(t):
        m = _GPU_NAME_RE.search(t)
        if m:
            raw = m.group(1).strip()
            # Normalize RTX spacing / caps
            raw = re.sub(r"\s+", " ", raw).strip()
            # Title-case but keep RTX/GTX/RX caps
            up = raw.upper()
            if "RTX" in up:
                num = re.search(r"\d{3,4}", raw)
                suffix = " TI" if re.search(r"\bTI\b", up) else (" SUPER" if "SUPER" in up else "")
                gpu = f"RTX {num.group(0)}{suffix}" if num else "RTX Dedicated"
                return gpu, True
            if "GTX" in up:
                num = re.search(r"\d{3,4}", raw)
                gpu = f"GTX {num.group(0)}" if num else "GTX Dedicated"
                return gpu, True
            if "RADEON RX" in up:
                num = re.search(r"RX\s*\d+", up)
                gpu = num.group(0).replace(" ", " ").title() if num else "Radeon RX"
                return gpu, True
            if "ARC" in up:
                return raw.title(), True
            return raw.title(), True
        return "Dedicated GPU", True
    if _INTEGRATED_GPU_RE.search(t):
        m = _GPU_NAME_RE.search(t)
        if m:
            return m.group(1).strip().title(), False
        return "Integrated", False
    # Fallback: if title mentions graphics but no discrete, treat as integrated
    if re.search(r"graphics|gpu|adreno", t, re.IGNORECASE):
        return "Integrated", False
    return "Unknown", False


def _update_metadata(extra: dict = None):
    """Write metadata.json with last_updated/last_trained timestamps."""
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    meta = {}
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Always update last_updated; last_trained only if model exists
    meta["last_updated"] = now
    meta["last_updated_local"] = datetime.datetime.now().astimezone().isoformat()
    if os.path.exists(MODEL_PATH):
        try:
            mtime = os.path.getmtime(MODEL_PATH)
            meta["last_trained"] = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc).isoformat()
            meta["last_trained_local"] = datetime.datetime.fromtimestamp(mtime).astimezone().isoformat()
        except Exception:
            pass
    else:
        # If no model yet, treat last_updated as last_trained
        if "last_trained" not in meta:
            meta["last_trained"] = meta["last_updated"]
            meta["last_trained_local"] = meta["last_updated_local"]
    if extra:
        meta.update(extra)
    try:
        with open(METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    return meta


def get_metadata():
    """Return metadata dict with last_updated/last_trained (ISO strings)."""
    if not os.path.exists(METADATA_PATH):
        # Create on demand
        return _update_metadata()
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return _update_metadata()


def clean_dataset(raw_path: str = RAW_PATH, clean_path: str = CLEAN_PATH):
    if not os.path.exists(raw_path):
        return {"error": f"Raw data file not found at path {raw_path}"}

    # Read raw CSV
    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        return {"error": f"Failed to read CSV: {str(e)}"}

    # Normalize column names (scraper produces lowercase like "product_url";
    # notebooks may title-case them). Map to the canonical names used here.
    col_map = {
        "title": "Title",
        "price": "Price",
        "brand": "Brand",
        "ram": "RAM",
        "ssd_storage": "SSD_Storage",
        "color": "Color",
        "processor": "Processor",
        "os": "OS",
        "rating": "Rating",
        "product_url": "Product_URL",
        "image_url": "Image_URL",
    }
    df = df.rename(columns=col_map)

    # Add the new columns (Title, Product_URL, Image_URL) if the raw file
    # predates them, so the pipeline is backward compatible.
    for col in ["Title", "Product_URL", "Image_URL"]:
        if col not in df.columns:
            df[col] = ""

    # Drop rows that are completely empty or have no brand and price
    df = df.dropna(subset=["Price", "Brand"], how="all")

    # Clean Price
    def clean_price(val):
        if pd.isna(val):
            return None
        val_str = str(val).replace('"', "").replace(",", "").strip()
        # Find all digits
        nums = re.findall(r"\d+", val_str)
        if nums:
            return int(nums[0])
        return None

    df["Price"] = df["Price"].apply(clean_price)
    df = df.dropna(subset=["Price"])

    # Cheap rows are almost always laptop accessories (bags, batteries,
    # webcams, speakers, spare keyboards, monitor stands...), so a row under
    # ₹15,000 is kept ONLY if it clearly looks like a laptop. A real laptop
    # names a CPU, or a laptop product line together with a screen size.
    # (Bare "laptop" + a screen size is NOT enough: spare parts like
    # "Replacement Screen for ... Laptop (15.6 Inch...)" match that too.)
    cheap = df["Price"] < 15000

    screen_size = re.compile(
        r"\d+(?:\.\d+)?\s*(?:inch|in\.?|″|”|\")|\d+(?:\.\d+)?\s*cm\b", re.IGNORECASE
    )
    cpu_word = re.compile(
        r"\b(?:core i\d|core [357]\b|ultra \d\b|ryzen|snapdragon|celeron|athlon|"
        r"pentium|qualcomm|media ?tek|intel|amd)\b",
        re.IGNORECASE,
    )
    product_line = re.compile(
        r"\b(?:macbook|chromebook|thinkpad|thinkbook|ideapad|yoga|legion|loq|"
        r"vivobook|zenbook|inspiron|latitude|xps|pavilion|spectre|elitebook|probook|"
        r"omnibook|victus|aspire|predator|nitro|swift|tuf|rog|razer|surface|alienware)\b",
        re.IGNORECASE,
    )
    # Accessory nouns that never appear in a real laptop title — these rows are
    # dropped at any price. (Words like "cover"/"privacy"/"keyboard"/"fan" are
    # deliberately NOT here: laptops legitimately mention metal covers,
    # privacy shutters and backlit keyboards.)
    accessory_unconditional = re.compile(
        r"\b(?:bag|case|sleeve|cooling pad|cooling stand|laptop stand|stand|table|desk|"
        r"docking|adapter|charger|charging|cable|screen protector|screen guard|screenguard|"
        r"battery|briefcase|backpack|messenger|monitor|replacement|spare|compatible|"
        r"accessory|holder|tray|organizer|mouse|borescope|videoscope|inspection camera|endoscope)\b",
        re.IGNORECASE,
    )
    # Words that identify accessories when the row is cheap (<₹15k). Real
    # laptops (always ≥₹15k) legitimately mention backlit keyboards, metal
    # covers, webcams and privacy shutters, so these only gate cheap rows.
    cheap_accessory_words = re.compile(
        r"\b(?:keyboard|keyboard cover|cover|fan|camera|webcam|privacy|silicone|skin)\b",
        re.IGNORECASE,
    )

    title = df["Title"].astype(str)
    # Real laptops name their product line near the start of the title, while
    # spare parts mention the compatible laptop later ("... for Legion Go, ROG
    # Ally"), so the product-line check only looks at the first 45 characters.
    title_prefix = title.str[:45]
    has_cpu = title.str.contains(cpu_word, regex=True)
    has_screen = title.str.contains(screen_size, regex=True)
    has_line = title_prefix.str.contains(product_line, regex=True)
    # A cheap row is a real laptop only if it names a CPU, or a laptop product
    # line together with a screen size. (Bare "laptop" + a screen size is NOT
    # enough: spare parts like "Replacement Screen for ... Laptop (15.6 Inch...)"
    # match that too.)
    looks_laptop = has_cpu | (has_line & has_screen)
    # Every real laptop has at least a CPU, a screen size, or a product line —
    # rows with none (RAM kits, spare keyboards, game consoles) are not laptops.
    has_indicator = has_cpu | has_screen | has_line
    # Titles that are just a brand name ("Oemgenuine", "Twelve South") are
    # never laptops.
    too_short_title = title.str.strip().str.len() < 20
    cheap_junk = cheap & (
        title.str.contains(cheap_accessory_words, regex=True) | ~looks_laptop
    )
    df = df[
        ~(
            too_short_title
            | title.str.contains(accessory_unconditional, regex=True)
            | ~has_indicator
            | cheap_junk
        )
    ]

    # Clean Brand
    def clean_brand(val):
        if pd.isna(val):
            return "Unknown"
        val_str = str(val).strip().capitalize()
        # Fix casing for specific brands
        val_lower = val_str.lower()
        if val_lower == "hp":
            return "HP"
        if val_lower == "msi":
            return "MSI"
        if val_lower == "asus":
            return "ASUS"
        if val_lower == "bkn":
            return "BKN"
        if val_lower == "dell":
            return "Dell"
        if val_lower == "aspire":
            # Aspire is an Acer product line, not a separate brand.
            return "Acer"
        return val_str

    df["Brand"] = df["Brand"].apply(clean_brand)

    # Drop rows whose brand is clearly not a laptop maker or unknown.
    # "Aspire" is a real Acer product line, so it is NOT dropped here.
    # "Unknown" means scraper couldn't infer brand and title lacked reliable brand keyword - drop those.
    bad_brands = [
        "More",
        "Related",
        "Need",
        "Highly",
        "Table",
        "Dellmodel",
        "Cosmus",
        "Ebook",
        "Tech",
        "New",
        "Certified",
        "Alketron",
        "Browsebook",
        "Unknown",
        "Generic",
        "Oemgenuine",
        "Laptop",
        "Portable",
        "Windows",
        "Mini",
        "Maxbook",
        "Kurietim",
        "Bmax",
        "Kooforway",
        "Futopia",
        "Dailyobjects",
        "Gadgets",
        "Portronics",
        "Kensington",
        "Urban",
    ]
    # Case-insensitive match, since Brand is normalised to a canonical case.
    bad_brands_lower = {b.lower() for b in bad_brands}
    df = df[~df["Brand"].str.lower().isin(bad_brands_lower)]

    # Clean Title (fall back to brand if empty, drop products without one)
    df["Title"] = (
        df["Title"].fillna("").astype(str).str.strip().replace("N/A", "", regex=False)
    )
    df["Title"] = df["Title"].where(df["Title"] != "", df["Brand"])
    df = df[df["Title"].str.len() > 0]

    # Clean Product_URL and Image_URL (strip spaces, blank out placeholders)
    def _clean_url_series(s):
        return (
            s.fillna("")
            .astype(str)
            .replace({"nan": "", "None": "", "N/A": ""}, regex=False)
            .str.strip()
        )

    df["Product_URL"] = _clean_url_series(df["Product_URL"])
    df["Image_URL"] = _clean_url_series(df["Image_URL"])
    df["Product_URL"] = df["Product_URL"].fillna("")
    df["Image_URL"] = df["Image_URL"].fillna("")

    # Backfill a product URL from the ASIN column (added by the scraper) so
    # rows that lost their search-page href still get a correct link.
    if "asin" in df.columns:
        df["asin"] = df["asin"].astype(str).str.strip()
        df["asin"] = df["asin"].replace({"nan": "", "None": ""})
        missing_url = df["Product_URL"].str.len() == 0
        df.loc[missing_url & df["asin"].str.len().gt(0), "Product_URL"] = (
            "https://www.amazon.in/dp/" + df.loc[missing_url & df["asin"].str.len().gt(0), "asin"]
        )

    # Deduplicate by the real product (ASIN embedded in the product URL) so
    # Amazon's repeated ad slots never collapse distinct laptops together.
    # Rows without a URL fall back to Title+Price, and the two groups are
    # cross-checked so a URL-less row that duplicates a kept URL row is dropped.
    df["_has_url"] = df["Product_URL"].str.len() > 0
    with_url = df[df["_has_url"]].drop_duplicates(
        subset=["Product_URL"], keep="first"
    )
    without_url = df[~df["_has_url"]].drop_duplicates(
        subset=["Title", "Price"], keep="first"
    )
    df = pd.concat([with_url, without_url], ignore_index=True)
    df = df.drop_duplicates(subset=["Title", "Price"], keep="first").drop(
        columns=["_has_url"]
    )

    # Clean RAM (Standardize as integer in GB)
    def clean_ram(val):
        if pd.isna(val):
            return 8  # Default RAM
        val_str = str(val).upper().strip()
        nums = re.findall(r"\d+", val_str)
        if nums:
            return int(nums[0])
        if "DDR" in val_str or "RAM" in val_str:
            return 16
        return 8

    df["RAM"] = df["RAM"].apply(clean_ram)

    # Clean SSD Storage (Standardize as integer in GB)
    def clean_storage(val):
        if pd.isna(val):
            return 512  # Default SSD
        val_str = str(val).strip()
        nums = re.findall(r"\d+", val_str)
        if nums:
            # SSD storage values are usually 128, 256, 512, 1024, etc.
            val_num = int(nums[0])
            # Handle float formats (e.g. 1.0 which means 1024GB, or 1024.0)
            if val_num <= 4:  # e.g. 1.0, 2.0 -> TB to GB
                return val_num * 1024
            return val_num
        return 512

    df["SSD_Storage"] = df["SSD_Storage"].apply(clean_storage)

    # Clean Color
    df["Color"] = (
        df["Color"].fillna("Unknown").apply(lambda x: str(x).strip().capitalize())
    )

    # Clean Processor
    def clean_processor(val, brand=None):
        if pd.isna(val):
            # Apple machines ship with Apple silicon (M1/M2/M3), never Intel.
            # Brand is passed in so NaN processor values are labelled correctly.
            return "Apple" if brand == "Apple" else "Intel"
        val_str = str(val).strip().lower()
        if "intel" in val_str:
            return "Intel"
        if "amd" in val_str or "ryzen" in val_str:
            return "AMD"
        if "apple" in val_str or "m1" in val_str or "m2" in val_str or "m3" in val_str:
            return "Apple"
        if "snapdragon" in val_str:
            return "Snapdragon"
        if "celeron" in val_str:
            return "Celeron"
        if "mediatek" in val_str:
            return "MediaTek"
        return str(val).strip().capitalize()

    df["Processor"] = df.apply(
        lambda row: clean_processor(row["Processor"], row["Brand"]), axis=1
    )

    # Clean OS
    def clean_os(val):
        if pd.isna(val):
            return "Windows"  # Default
        val_str = str(val).strip().lower()
        if "windows" in val_str:
            return "Windows"
        if "mac" in val_str:
            return "MacOS"
        if "chrome" in val_str:
            return "Chrome OS"
        if "android" in val_str:
            return "Android"
        if "linux" in val_str:
            return "Linux"
        return str(val).strip().capitalize()

    df["OS"] = df["OS"].apply(clean_os)

    # Clean Rating
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(4.0)

    # GPU extraction from Title
    gpu_names = []
    gpu_dedicated = []
    for t in df["Title"].astype(str):
        name, dedicated = extract_gpu_info(t)
        gpu_names.append(name)
        gpu_dedicated.append(bool(dedicated))
    df["GPU"] = gpu_names
    df["Has_Dedicated_GPU"] = gpu_dedicated

    # Availability & timestamps
    # Preserve Is_Available if exists (scraped twice, items missing become unavailable)
    # Default True for fresh scrape
    if "Is_Available" not in df.columns:
        df["Is_Available"] = True
    else:
        df["Is_Available"] = df["Is_Available"].fillna(True).astype(bool)
        # If product disappeared from new scrape but exists in old, mark unavailable elsewhere
        # Here clean_dataset builds from raw; raw always marks available, so old unavailable
        # rows are re-added via merge below
    if "Last_Updated" not in df.columns:
        df["Last_Updated"] = datetime.datetime.now().isoformat()

    # Preserve Image_Path and Is_Available/Last_Updated from previous clean file
    if "Image_Path" not in df.columns:
        df["Image_Path"] = ""
    if os.path.exists(clean_path):
        try:
            prev = pd.read_csv(clean_path)
            # Map by Product_URL
            prev_img = {}
            prev_avail = {}
            if "Image_Path" in prev.columns and "Product_URL" in prev.columns:
                prev_img = dict(zip(prev["Product_URL"], prev["Image_Path"]))
            if "Is_Available" in prev.columns and "Product_URL" in prev.columns:
                # Keep unavailable rows that are no longer in new scrape, but skip bad brands and accessories
                existing_urls = set(df["Product_URL"].astype(str))
                missing = prev[~prev["Product_URL"].astype(str).isin(existing_urls)]
                # Drop bad brands from missing - if laptop doesn't exist to begin with, don't keep
                if "Brand" in missing.columns:
                    bad_lower = {b.lower() for b in bad_brands}
                    missing = missing[~missing["Brand"].astype(str).str.lower().isin(bad_lower)]
                # Drop accessories that were previously mis-classified (e.g., borescope)
                if "Title" in missing.columns:
                    acc = re.compile(r"\b(?:bag|case|sleeve|cooling pad|cooling stand|laptop stand|stand|table|desk|docking|adapter|charger|charging|cable|screen protector|screen guard|screenguard|battery|briefcase|backpack|messenger|monitor|replacement|spare|compatible|accessory|holder|tray|organizer|mouse|borescope|videoscope|inspection camera|endoscope)\b", re.I)
                    missing = missing[~missing["Title"].astype(str).str.contains(acc, regex=True, na=False)]
                if not missing.empty:
                    # Keep missing rows as unavailable
                    missing = missing.copy()
                    missing["Is_Available"] = False
                    missing["Last_Updated"] = missing.get("Last_Updated", datetime.datetime.now().isoformat())
                    # Ensure GPU columns exist on old rows
                    if "GPU" not in missing.columns:
                        missing["GPU"] = missing["Title"].apply(lambda x: extract_gpu_info(x)[0])
                    if "Has_Dedicated_GPU" not in missing.columns:
                        missing["Has_Dedicated_GPU"] = missing["Title"].apply(lambda x: extract_gpu_info(x)[1])
                    df = pd.concat([df, missing], ignore_index=True)
            # Restore Image_Path for existing URLs
            if prev_img:
                df["Image_Path"] = df["Product_URL"].map(lambda u: prev_img.get(u, "")) .fillna("")
                # Also preserve non-empty existing paths already set
        except Exception:
            pass

    # Ensure boolean type
    df["Has_Dedicated_GPU"] = df["Has_Dedicated_GPU"].astype(bool)
    df["Is_Available"] = df["Is_Available"].astype(bool)
    if "Last_Updated" not in df.columns:
        df["Last_Updated"] = datetime.datetime.now().isoformat()
    df["Last_Updated"] = df["Last_Updated"].fillna(datetime.datetime.now().isoformat())

    # Final drop of any remaining bad brand rows (including those re-added via missing or new scrape)
    df = df[~df["Brand"].astype(str).str.lower().isin({b.lower() for b in bad_brands})]

    # Save cleaned file
    try:
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        df.to_csv(clean_path, index=False)
        # Update metadata
        _update_metadata({"total_records": len(df), "available_count": int(df["Is_Available"].sum()), "unavailable_count": int((~df["Is_Available"]).sum())})
    except Exception as e:
        return {"error": f"Failed to save cleaned CSV: {str(e)}"}

    # Read raw file to count rows
    try:
        raw_count = len(pd.read_csv(raw_path))
    except:
        raw_count = len(df)

    return {
        "status": "success",
        "raw_count": raw_count,
        "clean_count": len(df),
        "columns": list(df.columns),
        "brands": df["Brand"].value_counts().to_dict(),
        "avg_price": float(df["Price"].mean()),
        "avg_rating": float(df["Rating"].mean()),
    }


def get_dataset_stats(clean_path: str = CLEAN_PATH):
    if not os.path.exists(clean_path):
        res = clean_dataset()
        if "error" in res:
            return {
                "error": "Clean dataset does not exist, and automatic cleaning failed."
            }

    try:
        df = pd.read_csv(clean_path)
    except Exception as e:
        return {"error": f"Failed to read clean CSV: {str(e)}"}

    meta = get_metadata()
    gpu_dist = df["GPU"].value_counts().to_dict() if "GPU" in df.columns else {}
    avail = int(df["Is_Available"].sum()) if "Is_Available" in df.columns else len(df)
    return {
        "status": "success",
        "total_records": len(df),
        "available_count": avail,
        "unavailable_count": len(df) - avail,
        "brands": df["Brand"].value_counts().to_dict(),
        "processors": df["Processor"].value_counts().to_dict(),
        "os": df["OS"].value_counts().to_dict(),
        "gpu": gpu_dist,
        "dedicated_gpu_count": int(df["Has_Dedicated_GPU"].sum()) if "Has_Dedicated_GPU" in df.columns else 0,
        "min_price": int(df["Price"].min()),
        "max_price": int(df["Price"].max()),
        "avg_price": float(df["Price"].mean()),
        "avg_rating": float(df["Rating"].mean()),
        "ram_distribution": df["RAM"].value_counts().to_dict(),
        "storage_distribution": df["SSD_Storage"].value_counts().to_dict(),
        "last_updated": meta.get("last_updated"),
        "last_updated_local": meta.get("last_updated_local"),
        "last_trained": meta.get("last_trained"),
        "last_trained_local": meta.get("last_trained_local"),
    }


def get_laptops_data(clean_path: str = CLEAN_PATH):
    if not os.path.exists(clean_path):
        cleaning_result = clean_dataset()
        if "error" in cleaning_result:
            print(f"Error during on-the-fly dataset cleaning: {cleaning_result['error']}")
            return []
        if not os.path.exists(clean_path):
            print(f"Clean dataset still not found at {clean_path}")
            return []
    try:
        df = pd.read_csv(clean_path)
        # NaN/<NA> values are not JSON-compliant. Empty strings keep the JSON
        # clean, so collapse missing values on string columns (Arrow string
        # dtypes keep NA as NA otherwise, which then fails JSON serialisation).
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype("object").fillna("")
            else:
                df[col] = df[col].where(pd.notna(df[col]), None)
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"Error loading laptops from {clean_path}: {str(e)}")
        return []


def download_laptop_images(clean_path: str = CLEAN_PATH, assets_dir: str = ASSETS_DIR):
    """Download each laptop's product image into assets/laptops/.

    Adds an ``Image_Path`` column (relative URL like ``/assets/laptops/3.jpg``)
    to the cleaned CSV. Best-effort: rows that fail to download keep an empty
    ``Image_Path`` and can still fall back to the remote ``Image_URL``.

    Returns the number of images successfully downloaded.
    """
    if not os.path.exists(clean_path):
        print(f"Clean dataset not found at {clean_path}")
        return 0

    try:
        df = pd.read_csv(clean_path)
    except Exception as e:
        print(f"Failed to read clean CSV: {e}")
        return 0

    if "Image_Path" not in df.columns:
        df["Image_Path"] = ""
    else:
        # The CSV round-trip turns empty cells into NaN floats, which rejects
        # string assignments. Cast back to an object column so we can write
        # local paths into it.
        df["Image_Path"] = df["Image_Path"].astype("object")

    os.makedirs(assets_dir, exist_ok=True)
    downloaded = 0

    for idx, row in df.iterrows():
        image_url = str(row.get("Image_URL", "") or "").strip()
        if not image_url or image_url in ("N/A", "nan"):
            continue
        existing_path = row.get("Image_Path")
        if existing_path is not None and str(existing_path).strip() and str(existing_path) != "nan":
            continue  # already downloaded

        # Build a safe local filename from the remote image id.
        name = os.path.basename(urlparse(image_url).path)
        if not name.endswith(".jpg") and "." in name:
            name = os.path.splitext(name)[0] + ".jpg"
        local_name = f"{idx}_{name}"
        local_path = os.path.join(assets_dir, local_name)
        rel_path = f"/assets/laptops/{local_name}"

        try:
            import requests

            resp = requests.get(image_url, timeout=15)
            if resp.status_code == 200 and resp.content:
                with open(local_path, "wb") as f:
                    f.write(resp.content)
                df.at[idx, "Image_Path"] = rel_path
                downloaded += 1
        except Exception as e:
            print(f"  image {idx} failed: {e}")

    # Persist the local paths back into the cleaned CSV.
    try:
        df.to_csv(clean_path, index=False)
    except Exception as e:
        print(f"Failed to save image paths to CSV: {e}")

    print(f"Downloaded {downloaded} laptop images into {assets_dir}")
    return downloaded


if __name__ == "__main__":
    # Test cleaning
    print("Testing data cleaning script...")
    res = clean_dataset()
    print("Clean Result:", res)
    if "error" not in res:
        print("Stats:", get_dataset_stats())
