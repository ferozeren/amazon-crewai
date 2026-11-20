import os

import pandas as pd
from crewai.tools import tool

from .paths import CLEAN_CSV

CLEAN_PATH = str(CLEAN_CSV)


def load_clean_df():
    path = CLEAN_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        # If the file doesn't exist after ensuring directory, try cleaning
        from .data_manager import clean_dataset

        print(f"Clean dataset not found at {path}. Attempting to clean...")
        clean_dataset(clean_path=path)  # Pass the correct path to clean_dataset
        # Verify if cleaning was successful
        if not os.path.exists(path):
            print(
                f"Error: Clean dataset still not found after cleaning attempt at {path}."
            )
            return pd.DataFrame()  # Return empty DataFrame if still not found

    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading clean dataset in tools from {path}: {e}")
        return pd.DataFrame()


@tool("Get Laptop Dataset Statistics")
def get_dataset_stats_tool() -> str:
    """Useful to get an overview of the laptop database, including available brands,
    average prices, min/max price range, and distribution of RAM/Storage.
    Use this to see what kinds of laptops are in the database before querying."""
    df = load_clean_df()
    if df.empty:
        return "The laptop database is empty or could not be loaded."

    num_laptops = len(df)
    brands = df["Brand"].value_counts().to_dict()
    processors = df["Processor"].value_counts().to_dict()
    min_p, max_p = df["Price"].min(), df["Price"].max()
    avg_p = df["Price"].mean()

    stats_str = f"### Laptop Database Overview\n"
    stats_str += f"- **Total Laptops Available**: {num_laptops}\n"
    stats_str += f"- **Price Range**: {min_p:,} INR to {max_p:,} INR (Average: {avg_p:,.2f} INR)\n"
    stats_str += f"- **Available Brands**: {', '.join([f'{k} ({v})' for k, v in brands.items()])}\n"
    stats_str += f"- **Processors Available**: {', '.join([f'{k} ({v})' for k, v in processors.items()])}\n"
    stats_str += f"- **Operating Systems**: {df['OS'].value_counts().to_dict()}\n"
    return stats_str


@tool("Query Laptops Dataset")
def query_laptops_tool(
    brand: str = None,
    max_price: float = None,
    min_ram: float = None,
    processor: str = None,
    os_name: str = None,
    sort_by_rating: bool = True,
    gpu: str = None,
    include_unavailable: bool = False,
) -> str:
    """
    Query the laptop database with specific search filters.
    All arguments are optional:
    - brand: Laptop brand (e.g. 'HP', 'Dell', 'Lenovo', 'Acer', 'ASUS', 'Apple')
    - max_price: Maximum price budget in INR (e.g. 60000)
    - min_ram: Minimum RAM in GB (e.g. 8, 16, 32)
    - processor: Core CPU type (e.g. 'Intel', 'AMD', 'Apple', 'Snapdragon')
    - os_name: Operating system (e.g. 'Windows', 'MacOS', 'Chrome OS')
    - sort_by_rating: If True, returns higher rated laptops first (uses ML ranker if available).
    - gpu: GPU requirement - 'dedicated' for discrete GPU (RTX/GTX/Radeon RX), 'integrated' for iGPU, or specific name like 'RTX 4060'
    - include_unavailable: If False (default), only show currently available laptops; if True, include unavailable with tag.
    Returns a markdown table of matching laptops (with product title and link) up to the top 10 rows.
    """
    df = load_clean_df()
    if df.empty:
        return "The laptop database is empty or could not be loaded."

    filtered_df = df.copy()

    # Apply filters (case insensitive where appropriate)
    if brand:
        brand_clean = str(brand).strip().lower()
        filtered_df = filtered_df[filtered_df["Brand"].str.lower() == brand_clean]

    if max_price is not None:
        try:
            filtered_df = filtered_df[filtered_df["Price"] <= float(max_price)]
        except ValueError:
            pass

    if min_ram is not None:
        try:
            filtered_df = filtered_df[filtered_df["RAM"] >= float(min_ram)]
        except ValueError:
            pass

    if processor:
        proc_clean = str(processor).strip().lower()
        filtered_df = filtered_df[filtered_df["Processor"].str.lower() == proc_clean]

    if os_name:
        os_clean = str(os_name).strip().lower()
        # Handle 'mac' as 'macos'
        if os_clean == "mac":
            os_clean = "macos"
        filtered_df = filtered_df[filtered_df["OS"].str.lower() == os_clean]

    if gpu:
        gpu_clean = str(gpu).strip().lower()
        if gpu_clean in ("dedicated", "yes", "true", "discrete", "required"):
            if "Has_Dedicated_GPU" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["Has_Dedicated_GPU"].astype(str).str.lower().isin(["true", "1"])]
            else:
                # fallback: title contains RTX/GTX
                filtered_df = filtered_df[filtered_df["Title"].str.contains(r"RTX|GTX|Radeon RX", case=False, na=False)]
        elif gpu_clean in ("integrated", "igpu", "no", "false"):
            if "Has_Dedicated_GPU" in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df["Has_Dedicated_GPU"].astype(str).str.lower().isin(["true", "1"])]
        elif gpu_clean not in ("any", "", "none"):
            # Specific GPU name like RTX 4060
            filtered_df = filtered_df[filtered_df["GPU"].astype(str).str.lower().str.contains(gpu_clean, na=False) | filtered_df["Title"].str.lower().str.contains(gpu_clean, na=False)]

    # Availability filter
    if not include_unavailable and "Is_Available" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["Is_Available"].astype(str).str.lower().isin(["true", "1"])]

    if filtered_df.empty:
        return "No laptops found matching these specific filters. Try raising the price budget or relaxation of criteria. Note: use include_unavailable=True to see items that are no longer stocked."

    # Sort: prefer ML score if available, else rating. Available first.
    try:
        from .predict import score_laptops
        # Create dummy profile for sorting (no budget bias for tool overview)
        filtered_df = score_laptops(filtered_df, {"budget": max_price, "ram": min_ram, "gpu": gpu})
        # Availability-first
        if "Is_Available" in filtered_df.columns:
            filtered_df["_avail"] = filtered_df["Is_Available"].astype(str).str.lower().isin(["true", "1"]).astype(int)
            filtered_df = filtered_df.sort_values(by=["_avail", "_score", "Rating", "Price"], ascending=[False, False, False, True])
            filtered_df = filtered_df.drop(columns=["_avail", "_score", "_ml_score"], errors="ignore")
            if "LogPrice" in filtered_df.columns:
                filtered_df = filtered_df.drop(columns=["LogPrice"], errors="ignore")
        else:
            filtered_df = filtered_df.sort_values(by=["_score", "Rating", "Price"], ascending=[False, False, True])
            filtered_df = filtered_df.drop(columns=["_score", "_ml_score", "LogPrice"], errors="ignore")
    except Exception:
        if sort_by_rating:
            filtered_df = filtered_df.sort_values(by=["Rating", "Price"], ascending=[False, True])
        else:
            filtered_df = filtered_df.sort_values(by="Price", ascending=True)
        # Availability first
        if "Is_Available" in filtered_df.columns:
            filtered_df["_avail"] = filtered_df["Is_Available"].astype(str).str.lower().isin(["true", "1"]).astype(int)
            filtered_df = filtered_df.sort_values(by=["_avail", "Rating", "Price"], ascending=[False, False, True])
            filtered_df = filtered_df.drop(columns=["_avail"], errors="ignore")

    # Amazon lists each laptop colour as a separate SKU, so colour variants of
    # the same model would otherwise crowd out other machines. Collapse them to
    # one row per distinct model.
    from .data_manager import base_model

    filtered_df = filtered_df.copy()
    filtered_df["_base"] = filtered_df.apply(
        lambda r: base_model(r.get("Title") or r["Brand"], r.get("Color")), axis=1
    )
    filtered_df = filtered_df.drop_duplicates(subset=["_base"], keep="first")

    # Take top 10
    top_matches = filtered_df.head(10)

    # Format as markdown table
    result_str = f"Found {len(filtered_df)} matching laptops. Showing top {len(top_matches)} matches:\n\n"

    headers = [
        "Title",
        "Brand",
        "Price (INR)",
        "RAM (GB)",
        "SSD Storage (GB)",
        "Processor",
        "OS",
        "Color",
        "GPU",
        "Rating",
        "Available",
        "Product URL",
    ]
    result_str += " | ".join(headers) + "\n"
    result_str += " | ".join(["---"] * len(headers)) + "\n"

    for _, row in top_matches.iterrows():
        avail = str(row.get("Is_Available", True))
        avail_tag = "Yes" if str(avail).lower() in ("true", "1") else "No - No longer available"
        row_vals = [
            str(row.get("Title", row["Brand"])),
            str(row["Brand"]),
            f"{int(row['Price']):,}",
            f"{int(row['RAM'])}GB",
            f"{int(row['SSD_Storage'])}GB",
            str(row["Processor"]),
            str(row["OS"]),
            str(row["Color"]),
            str(row.get("GPU", "Unknown")),
            f"{row['Rating']}/5.0",
            avail_tag,
            str(row.get("Product_URL", "")),
        ]
        result_str += " | ".join(row_vals) + "\n"

    return result_str
