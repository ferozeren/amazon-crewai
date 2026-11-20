import os

import pandas as pd
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

from .tools import get_dataset_stats_tool, query_laptops_tool
from .data_manager import base_model, product_color


def _initialize_llm(provider: str, api_key: str, model: str = None) -> LLM:
    """Initializes and returns an LLM instance based on the provider.

    Args:
        provider: The LLM provider (e.g., "gemini", "openai", "openrouter", "nvidia").
        api_key: The API key for the provider.
        model: Optional sub-model override (used for OpenRouter to select between
            the free models hosted on the OpenRouter gateway).

    Returns:
        An initialized crewai.LLM object.

    Raises:
        ValueError: If the API key is not found or the provider is unknown.
    """
    load_dotenv()  # Load environment variables

    MODEL_CONFIGS = {
        "gemini": {
            "model": "gemini/gemini-3.8-flash",
            "api_key_env": "GEMINI_API_KEY",
            "base_url": None,
        },
        "openai": {
            "model": "openai/gpt-4o-mini",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": None,
        },
        "openrouter": {
            # The bare domain serves the website, not the API.
            "model": "google/gemma-4-31b-it:free",
            "api_key_env": "OPENROUTER_API_KEY",
            "base_url": "https://openrouter.ai/api/v1",
        },
        "nvidia": {
            "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            "api_key_env": "NVIDIA_API_KEY",
            "base_url": "https://integrate.api.nvidia.com/v1",
        },
    }

    config = MODEL_CONFIGS.get(provider)
    if not config:
        raise ValueError(f"Unknown LLM provider: {provider}")

    actual_api_key = api_key or os.environ.get(config["api_key_env"])
    if not actual_api_key:
        raise ValueError(
            f"API Key for {provider} not found. Please provide it in the UI or environment variable {config['api_key_env']}."
        )

    model_str = config["model"]
    if provider == "openrouter":
        if model:
            model_str = model
        model_str = model_str.removeprefix("openrouter/")
        model_str = model_str.replace(
            "google/gemma-3-27b-it:free", "google/gemma-4-31b-it:free"
        ).replace(
            "meta-llama/llama-3.3-70b-instruct:free", "google/gemma-4-31b-it:free"
        )

    max_tokens = 4000 if provider == "openrouter" else None

    llm_kwargs = dict(
        model=model_str,
        api_key=actual_api_key,
        temperature=0.3,
        base_url=config["base_url"],
        max_tokens=max_tokens,
    )

    if provider in ("nvidia", "openrouter"):
        # LiteLLM reports no tool support for these models; the native OpenAI
        # path keeps CrewAI's agents calling tools properly.
        llm_kwargs["custom_openai"] = True

    if provider == "openrouter":
        # Otherwise CrewAI assumes 8k context and truncates prompts/tool results.
        llm_kwargs["context_window_size"] = 200000

    llm = LLM(**llm_kwargs)

    return llm


def _compute_fallback_matches(user_profile: dict):
    """Shared matching logic for the fallback report and the summary.

    Applies the user's filters with graceful relaxation and returns the top
    picks DataFrame plus context notes. Uses ML ranker if available and respects
    GPU + availability preferences.
    Returns (top_matches, match_note, brand_reference, budget_value), with
    top_matches set to None when the dataset cannot be loaded.
    """
    from .data_manager import get_laptops_data

    rows = get_laptops_data()
    if not rows:
        return None, None, None, None

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None, None, None

    budget = user_profile.get("budget")
    try:
        budget_value = float(budget) if budget not in (None, "") else None
    except (TypeError, ValueError):
        budget_value = None

    brand = str(user_profile.get("brand", "Any") or "Any").strip()
    brand_wanted = brand.lower() != "any"

    ram = user_profile.get("ram")
    if ram in (None, "", "Any", "any"):
        ram_wanted = None
    else:
        try:
            ram_wanted = float(ram)
        except (TypeError, ValueError):
            ram_wanted = None

    os_name = str(user_profile.get("os", "Any") or "Any").strip()
    os_wanted = os_name.lower()
    if os_wanted == "mac":
        os_wanted = "macos"
    os_wanted_bool = os_wanted != "any"

    gpu_pref = str(user_profile.get("gpu", "Any") or "Any").strip().lower()
    gpu_wants_dedicated = gpu_pref in ("dedicated", "yes", "true", "required", "discrete")
    gpu_wants_integrated = gpu_pref in ("integrated", "igpu", "no")
    include_unavail = bool(user_profile.get("include_unavailable", False))

    def apply_filters(use_budget, use_brand, use_os, use_gpu=True):
        pool = df
        # Availability: by default exclude unavailable; include if user checked option
        if not include_unavail and "Is_Available" in pool.columns:
            pool = pool[pool["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"])]
        if use_budget and budget_value is not None:
            pool = pool[pool["Price"] <= budget_value]
        if use_brand and brand_wanted:
            pool = pool[pool["Brand"].astype(str).str.lower() == brand.lower()]
        if ram_wanted is not None:
            pool = pool[pool["RAM"] >= ram_wanted]
        if use_os and os_wanted_bool:
            pool = pool[pool["OS"].astype(str).str.lower() == os_wanted]
        if use_gpu and "Has_Dedicated_GPU" in pool.columns:
            if gpu_wants_dedicated:
                pool = pool[pool["Has_Dedicated_GPU"].astype(str).str.lower().isin(["true", "1"])]
            elif gpu_wants_integrated:
                pool = pool[~pool["Has_Dedicated_GPU"].astype(str).str.lower().isin(["true", "1"])]
            elif gpu_pref not in ("any", "", "none"):
                if "GPU" in pool.columns:
                    pool = pool[pool["GPU"].astype(str).str.lower().str.contains(gpu_pref) | pool["Title"].astype(str).str.lower().str.contains(gpu_pref)]
        return pool

    filtered_df = apply_filters(True, True, True)
    match_note = None
    brand_reference = None

    if filtered_df.empty:
        reasons = []
        if brand_wanted:
            reasons.append(f"brand '{brand}'")
        if ram_wanted is not None:
            reasons.append(f"≥{int(ram_wanted)}GB RAM")
        if gpu_wants_dedicated or gpu_wants_integrated or (gpu_pref not in ("any", "", "none") and not gpu_wants_dedicated and not gpu_wants_integrated):
            reasons.append(f"GPU '{gpu_pref}'")
        if os_wanted_bool:
            reasons.append(f"OS '{os_name}'")
        if budget_value is not None:
            reasons.append(f"budget ≤₹{int(budget_value):,}")
        reason_str = ", ".join(reasons) if reasons else "your filters"
        match_note = (
            f"No laptops match {reason_str} among currently available items. "
            "Try lowering RAM, choosing 'Any' brand/GPU, increasing budget, or checking 'Show unavailable currently' to see out-of-stock matches."
        )
        if brand_wanted:
            brand_pool = df[df["Brand"].astype(str).str.lower() == brand.lower()]
            if not include_unavail and "Is_Available" in brand_pool.columns:
                brand_pool = brand_pool[brand_pool["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"])]
            if not brand_pool.empty:
                cheapest = brand_pool.sort_values("Price").iloc[0]
                brand_reference = (
                    f"{brand} laptops in our catalog start at "
                    f"₹{int(cheapest['Price']):,}."
                )
            else:
                brand_reference = f"No {brand} laptops are currently in stock."

    # Group colour SKUs of the same machine and remember their colours.
    df = df.copy()
    df["_base"] = df.apply(
        lambda r: base_model(r.get("Title") or r["Brand"], r.get("Color")), axis=1
    )
    df["_color"] = df.apply(
        lambda r: product_color(r.get("Title") or r["Brand"], r.get("Color")), axis=1
    )
    color_map = df.groupby("_base")["_color"].apply(
        lambda s: sorted({c for c in s if c != "Unknown"})
    )

    if filtered_df.empty:
        empty = pd.DataFrame(columns=list(df.columns) + ["_base", "_colors"])
        return empty, match_note, brand_reference, budget_value

    filtered_df = filtered_df.copy()
    filtered_df["_base"] = filtered_df.apply(
        lambda r: base_model(r.get("Title") or r["Brand"], r.get("Color")), axis=1
    )
    try:
        from .predict import score_laptops

        scored = score_laptops(filtered_df, user_profile)
        if "Is_Available" in scored.columns:
            scored["_avail"] = scored["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
            scored = scored.sort_values(by=["_avail", "_score", "Rating", "Price"], ascending=[False, False, False, True])
            scored = scored.drop(columns=["_avail", "_score", "_ml_score", "LogPrice"], errors="ignore")
        else:
            scored = scored.sort_values(by=["_score", "Rating", "Price"], ascending=[False, False, True])
            scored = scored.drop(columns=["_score", "_ml_score", "LogPrice"], errors="ignore")
        filtered_df = scored
    except Exception:
        filtered_df = filtered_df.sort_values(by=["Rating", "Price"], ascending=[False, True])
        if "Is_Available" in filtered_df.columns:
            filtered_df["_avail"] = filtered_df["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
            filtered_df = filtered_df.sort_values(by=["_avail", "Rating", "Price"], ascending=[False, False, True])
            filtered_df = filtered_df.drop(columns=["_avail"], errors="ignore")

    # Skip duplicate base models so colour variants never crowd out others.
    picks = []
    seen = set()
    avail_count = 0
    if "Is_Available" in filtered_df.columns:
        avail_count = int(filtered_df["Is_Available"].astype(str).str.lower().isin(["true","1","yes"]).sum())
    target = 5 if avail_count >= 5 or len(filtered_df) >= 5 else 3
    for _, row in filtered_df.iterrows():
        b = row["_base"]
        if b in seen:
            continue
        seen.add(b)
        picks.append(row)
        if len(picks) == target:
            break
    top_matches = pd.DataFrame(picks).reset_index(drop=True)
    top_matches["_colors"] = top_matches["_base"].map(color_map).apply(
        lambda v: v if isinstance(v, list) else []
    )
    return top_matches, match_note, brand_reference, budget_value


def build_fallback_summary(user_profile: dict):
    """Structured summary of the fallback picks for the UI overview panel."""
    top_matches, match_note, brand_reference, budget_value = _compute_fallback_matches(
        user_profile
    )
    if top_matches is None:
        return None

    picks = []
    for _, row in top_matches.iterrows():
        title = str(row.get("Title") or row["Brand"])
        picks.append(
            {
                "title": title,
                "brand": str(row["Brand"]),
                "price": int(row["Price"]),
                "ram": int(row["RAM"]),
                "storage": int(row["SSD_Storage"]),
                "processor": str(row["Processor"]),
                "os": str(row["OS"]),
                "rating": round(float(row["Rating"]), 1),
                "product_url": str(row.get("Product_URL") or ""),
                "image_path": str(row.get("Image_Path") or ""),
                "image_url": str(row.get("Image_URL") or ""),
                "colors": list(row.get("_colors") or []),
                "gpu": str(row.get("GPU", "Unknown")),
                "has_dedicated_gpu": bool(row.get("Has_Dedicated_GPU", False)),
                "is_available": bool(str(row.get("Is_Available", True)).lower() in ("true", "1", "yes")) if "Is_Available" in row else True,
                "last_updated": str(row.get("Last_Updated", "")),
            }
        )

    return {
        "top_pick": picks[0] if picks else None,
        "picks": picks,
        "relaxed": match_note is not None,
        "note": match_note,
        "brand_reference": brand_reference,
    }


def build_fallback_recommendation(user_profile: dict, error_message: str = None) -> str:
    """Generate a richer local recommendation report from the cleaned dataset when the LLM provider fails."""
    top_matches, match_note, brand_reference, budget_value = _compute_fallback_matches(
        user_profile
    )
    if top_matches is None:
        return "The laptop database could not be loaded, so no local recommendations are available right now."

    major = str(user_profile.get("major", "student") or "student").strip()
    major_lower = major.lower()

    if any(
        keyword in major_lower
        for keyword in [
            "computer science",
            "cs",
            "software",
            "engineering",
            "programming",
            "data science",
            "data",
            "machine learning",
            "artificial intelligence",
            "ai",
            "it",
            "information technology",
        ]
    ):
        workload = "coding, IDEs, virtual machines, and multitasking"
        fit_summary = "These picks are strong for software development and heavy browser/tab usage."
    elif any(
        keyword in major_lower
        for keyword in ["design", "graphic", "animation", "media"]
    ):
        workload = "creative applications and visual work"
        fit_summary = (
            "These picks are suitable for design and content creation workflows."
        )
    elif any(
        keyword in major_lower for keyword in ["business", "finance", "marketing"]
    ):
        workload = "office apps, presentations, and daily productivity"
        fit_summary = (
            "These picks are practical for general productivity and portability."
        )
    else:
        workload = "general study, research, and everyday productivity"
        fit_summary = "These picks balance price, performance, and reliability for day-to-day academic use."

    intro = (
        "The AI service is temporarily unavailable, so I picked the closest matches "
        "from our real laptop catalog using your criteria. Everything below is "
        "actual product data — no guesses."
    )
    if match_note:
        intro += f" {match_note}"
    if brand_reference:
        intro += f" For reference: {brand_reference}"

    # Human-friendly display of the user's preferences
    ram_pref = user_profile.get("ram")
    if ram_pref in (None, "Any", "any", ""):
        ram_display = "not specified"
    else:
        try:
            ram_display = f"{float(ram_pref):.0f} GB"
        except (TypeError, ValueError):
            ram_display = "not specified"
    brand_display = user_profile.get("brand") or "Any"
    os_display = user_profile.get("os") or "Any"
    gpu_display = str(user_profile.get("gpu", "Any") or "Any")
    avail_display = "Yes, show unavailable (tagged) " if user_profile.get("include_unavailable") else "Available only"

    lines = [
        intro,
        "",
        "## Your recommendation",
        "",
        "### Your requirements",
        "",
        f"- **Domain / Field:** {major}",
        (
            f"- **Budget:** up to ₹{budget_value:,.0f}"
            if budget_value is not None
            else "- **Budget:** flexible"
        ),
        f"- **Minimum RAM:** {ram_display}",
        f"- **Brand:** {brand_display}",
        f"- **Operating system:** {os_display}",
        f"- **GPU:** {gpu_display}",
        f"- **Include unavailable:** {avail_display}",
        "",
        fit_summary,
        "",
    ]

    pick_count = len(top_matches)
    if pick_count == 0:
        lines.append("### No matching laptops found")
        lines.append("")
        lines.append("We couldn't find any laptops that match **all** of your filters with currently available stock.")
        lines.append("")
        lines.append("- Try reducing **Minimum RAM** (e.g., 64GB is rare - only 2 models in 396 rows),")
        lines.append("- Choose **Brand: Any** or **GPU: Any**,")
        lines.append("- Increase your **budget**, or")
        lines.append("- Check **Show unavailable currently** to see out-of-stock matches (tagged).")
        lines.append("")
        if brand_reference:
            lines.append(f"_Hint: {brand_reference}_")
            lines.append("")
        lines.append("---")
        return "\n".join(lines)
    lines.append(f"### Top {pick_count} pick{'s' if pick_count != 1 else ''} for you")
    lines.append("")

    for pick_no, (_, row) in enumerate(top_matches.iterrows(), 1):
        price = int(row["Price"])
        ram = int(row["RAM"])
        storage = int(row["SSD_Storage"])
        rating = float(row["Rating"])
        brand = str(row["Brand"])
        title = str(row.get("Title") or brand)
        processor = str(row["Processor"])
        os_name = str(row["OS"])
        product_url = str(row.get("Product_URL") or "")
        gpu_name = str(row.get("GPU", "Unknown"))
        avail = bool(str(row.get("Is_Available", True)).lower() in ("true", "1", "yes"))
        has_dgpu = bool(row.get("Has_Dedicated_GPU", False))

        pros = []
        if ram >= 16:
            pros.append("plenty of RAM for multitasking and heavier apps")
        if storage >= 512:
            pros.append("ample SSD space for files, projects, and software")
        if rating >= 4.5:
            pros.append("a strong user rating")
        if processor in ["Intel", "AMD"]:
            pros.append("solid general-purpose performance")
        if has_dgpu:
            pros.append(f"dedicated GPU ({gpu_name}) for gaming/ML/creative work")
        if avail:
            pros.append("currently available")
        else:
            pros.append("price/spec reference (currently unavailable)")

        cons = []
        if budget_value is not None and price > budget_value:
            cons.append("slightly above your budget")
        if ram < 16:
            cons.append("16 GB of RAM would feel roomier for heavier workloads")
        if storage < 512:
            cons.append("the smaller SSD may fill up quickly")

        if not pros:
            pros.append("a balanced price-to-performance ratio")
        if not cons:
            cons.append("no major trade-offs in the dataset")

        title_line = title if len(title) <= 80 else title[:77] + "..."
        avail_tag = " ✅ Available" if avail else " ⚠️ No longer available (reference price)"
        lines.append(f"#### {pick_no}. {title_line}{avail_tag}")
        lines.append("")
        lines.append(
            f"**Why it suits you:** This {brand} laptop comes with {ram} GB RAM, a "
            f"{storage} GB SSD, {gpu_name} graphics ({'dedicated' if has_dgpu else 'integrated'}), and {processor} processing running {os_name}, so it "
            f"handles {workload} without breaking a sweat."
        )
        lines.append(f"**What you'll love:** {', '.join(pros)}.")
        lines.append(f"**Keep in mind:** {', '.join(cons)}.")
        lines.append(
            f"**Best for:** {major} students who want a dependable balance of "
            "price and everyday performance."
        )
        if product_url:
            lines.append(f"**[View on Amazon]({product_url})**")
        variant_colors = [c for c in (row.get("_colors") or []) if c]
        if len(variant_colors) > 1:
            lines.append(
                f"**Also in:** {', '.join(variant_colors)} (the same model in "
                "different finishes)."
            )
        lines.append("")

    lines.append("### At a glance")
    lines.append("")
    lines.append(
        "| Laptop | Price | RAM | Storage | Processor | OS | GPU | Rating | Avail | Best fit |"
    )
    lines.append("|---|---:|---:|---:|---|---|---|---|---:|---|")
    for _, row in top_matches.iterrows():
        brand = str(row["Brand"])
        title = str(row.get("Title") or brand)
        short_title = title if len(title) <= 60 else title[:57] + "..."
        gpu = str(row.get("GPU", "Unknown"))
        avail = "Yes" if str(row.get("Is_Available", True)).lower() in ("true","1","yes") else "No"
        lines.append(
            f"| {short_title} | ₹{int(row['Price']):,} | {int(row['RAM'])}GB | {int(row['SSD_Storage'])}GB | {row['Processor']} | {row['OS']} | {gpu} | {row['Rating']}/5 | {avail} | {major} / {workload} |"
        )
    lines.append("")
    lines.append("### Our final pick")
    best_row = top_matches.iloc[0]
    best_title = str(best_row.get("Title") or best_row["Brand"])
    lines.append(
        f"Take the **{best_title}** at ₹{int(best_row['Price']):,}. It pairs "
        f"{int(best_row['RAM'])} GB RAM with a {int(best_row['SSD_Storage'])} GB SSD and "
        f"a strong rating, making it the most dependable all-round choice for your "
        f"{major} needs within this budget."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "_These picks are grounded in real product data from our laptop catalog. "
        "The AI service will be back shortly — in the meantime, you can trust every "
        "specification listed above._"
    )
    return "\n".join(lines)


def create_laptop_crew(
    api_key: str,
    provider: str,
    user_profile: dict,
    model: str = None,
    step_callback=None,
    task_callback=None,
) -> Crew:
    """
    Create a Crew to analyze user requirements, query the laptop database,
    and generate custom recommendations.

    api_key: User provided LLM API Key (OpenAI or Gemini)
    provider: 'openai' or 'gemini'
    user_profile: dict containing budget, major, RAM, etc.
    model: Optional model override (used by the OpenRouter provider).
    step_callback: Optional callable invoked after every agent step (used to
        stream progress to the frontend via SSE).
    task_callback: Optional callable invoked after every task completes.
    """
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
        elif provider == "nvidia":
            api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError(
            f"API Key for {provider} not found. Please provide it in the UI or environment."
        )

    try:
        llm = _initialize_llm(provider, api_key, model=model)
    except ValueError as e:
        print(f"Error initializing LLM: {e}")
        return None

    if llm is None:
        return None


    gpu_display = user_profile.get("gpu", "Any")
    avail_display = "Include unavailable (tagged) if checked" if user_profile.get("include_unavailable") else "Available only"
    profile_desc = f"""
    - **Academic Major / Workload**: {user_profile.get("major", "Student")}
    - **Price Budget**: Max {user_profile.get("budget", "Any")} INR
    - **Minimum RAM**: {user_profile.get("ram", "Any")} GB
    - **Preferred Brand**: {user_profile.get("brand", "Any")}
    - **Preferred OS**: {user_profile.get("os", "Any")}
    - **GPU Preference**: {gpu_display} (dedicated GPU = RTX/GTX/Radeon RX discrete)
    - **Availability**: {avail_display}
    - **Specific Details/Preferences**: {user_profile.get("details", "None")}
    """

    data_specialist = Agent(
        role="Laptop Database Specialist",
        goal="Search the database using filters to find matching laptops that fit the user's budget and technical constraints.",
        backstory="""You are a meticulous data engineer. Your job is to query the laptop database
        using custom search tools. You look for laptops that fall within the user's budget and meet
        their basic RAM, storage, brand, and OS needs. You gather evidence and output exact product data,
        ensuring that no laptop specs or prices are made up. You always double-check the raw rows before presenting them.""",
        tools=[get_dataset_stats_tool, query_laptops_tool],
        llm=llm,
        verbose=True,
    )

    recommender = Agent(
        role="Academic & Professional Hardware Advisor",
        goal="Select the top 3 best matching laptops from the database search results, and explain how their specifications suit the user's academic major or professional workload.",
        backstory="""You are an expert computing systems consultant and academic advisor.
        You understand hardware inside out. You know that a Computer Science student needs a fast CPU,
        at least 16GB RAM for running Docker/VMs/compilers, and an SSD; a Graphic Designer needs a premium screen
        and strong processor; a business student needs long battery life and portability.
        Your goal is to explain *why* the recommended specs fit the customer's specific studies/workload,
        breaking down the trade-offs (Pros & Cons) of each recommendation in a friendly, persuasive tone.""",
        llm=llm,
        verbose=True,
    )

    search_task = Task(
        name="Search the laptop catalog",
        description=f"""
        1. Examine the user profile criteria:
        {profile_desc}

        2. First, call the statistics tool to check the dataset stats (price ranges, brand counts, GPU distribution, availability) if needed to understand what is available.
        3. Call the 'Query Laptops Dataset' tool with filters representing the user's budget, preferred brand, RAM, OS, GPU preference, and include_unavailable flag to find matching laptops. Respect ALL filters strictly - brand, minimum RAM, GPU and OS must match exactly. Do NOT relax or switch brand/RAM/GPU on your own. Availability is default available only.
        4. If no laptops match, return the empty result as-is with a note suggesting which filter to relax (e.g., "No 64GB RAM models under ₹80k with HP - try 32GB or higher budget"). Do NOT invent or substitute other brands/RAM.
        5. Compile a list of the matching laptops, keeping their exact product title, price, RAM, Storage, Processor, OS, GPU, availability tag, and rating.
        """,
        expected_output="A markdown table containing the matching laptops retrieved from the database (with their product titles, GPU, availability, and Amazon links), along with their specifications and prices. If no rows match, explicitly state 'No laptops found' and do not invent models or specs.",
        agent=data_specialist,
    )

    recommendation_task = Task(
        name="Write the recommendation report",
        description=f"""
        1. Analyze the matching laptops table from the search task.
        2. Choose the top 3 best matching options for the user's profile:
        {profile_desc}

        3. Write a comprehensive, personalized recommendation report.
        4. For each laptop, explain:
           - Why the hardware specs (RAM, Storage, Processor, OS) fit their academic major or workload.
           - Specific benefits for their profile (e.g. 'Since you are in Computer Science, 16GB RAM allows you to run virtual machines and IDEs smoothly...').
           - Pros and Cons (e.g. price vs. rating, color, brand reputation).
           - How it fits their financial constraints.
        5. Formulate a final comparison table comparing the recommended models.
        6. Structure the report beautifully in Markdown with sections, clear bullet points, bold highlights, and clean typography.

        Do not make up any specifications, ratings, or prices that are not present in the database search results.
        """,
        expected_output="A beautiful, detailed markdown report listing the top 3 laptop recommendations, including academic/professional justifications, pros/cons, and a final summary comparison table.",
        agent=recommender,
    )

    crew = Crew(
        agents=[data_specialist, recommender],
        tasks=[search_task, recommendation_task],
        process=Process.sequential,
        verbose=True,
        step_callback=step_callback,
        task_callback=task_callback,
    )

    return crew
