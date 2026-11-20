import os
import math

import joblib
import numpy as np
import pandas as pd

from .paths import MODEL_PATH


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def _heuristic_score(row, user_profile):
    # Budget-aware heuristic: penalize over-budget heavily, reward under-budget headroom
    price = float(row.get("Price", 0))
    rating = float(row.get("Rating", 4.0))
    ram = int(row.get("RAM", 8))
    ssd = int(row.get("SSD_Storage", 512))
    dedicated = bool(row.get("Has_Dedicated_GPU", False))
    budget = user_profile.get("budget")
    try:
        budget_val = float(budget) if budget not in (None, "", "Any") else None
    except Exception:
        budget_val = None

    score = rating * 4.0
    score += math.log1p(ram) * 1.2
    score += math.log1p(ssd / 256) * 0.8
    score -= math.log(price / 10000) * 1.1 if price > 0 else 0
    if dedicated:
        gpu_wanted = str(user_profile.get("gpu") or "Any").lower()
        if gpu_wanted in ("dedicated", "yes", "true", "required"):
            score += 3.0  # big bonus when user wants dedicated
        else:
            score += 1.5
    else:
        gpu_wanted = str(user_profile.get("gpu") or "Any").lower()
        if gpu_wanted in ("dedicated", "yes", "true", "required"):
            score -= 2.0  # penalize non-dedicated when dedicated required
    if budget_val is not None:
        if price <= budget_val:
            # Reward cheapness within budget (value headroom)
            score += (budget_val - price) / budget_val * 1.0
        else:
            over = (price - budget_val) / budget_val
            score -= over * 6.0  # heavy penalty
    # RAM preference
    try:
        ram_wanted = float(user_profile.get("ram")) if str(user_profile.get("ram")).lower() not in ("any", "", "none") else None
        if ram_wanted is not None:
            if ram >= ram_wanted:
                score += 0.7
            else:
                score -= (ram_wanted - ram) * 0.4
    except Exception:
        pass
    # Availability handled outside but tiny penalty for unavailable here too
    if not bool(row.get("Is_Available", True)):
        score -= 10  # will be sorted last anyway
    return float(score)


def score_laptops(df: pd.DataFrame, user_profile: dict):
    """Add _ml_score to df using model if available, else heuristic."""
    model = load_model()
    if model is not None:
        # Prepare features as in training
        try:
            df = df.copy()
            df["LogPrice"] = np.log1p(df["Price"].astype(float))
            # Fill cats
            for c in ["Brand", "Processor", "OS", "GPU"]:
                if c not in df.columns:
                    df[c] = "Unknown"
                df[c] = df[c].fillna("Unknown").astype(str)
            if "Has_Dedicated_GPU" not in df.columns:
                df["Has_Dedicated_GPU"] = False
            df["Has_Dedicated_GPU"] = df["Has_Dedicated_GPU"].astype(int)
            numeric_lr = ["LogPrice", "RAM", "SSD_Storage", "Rating"]
            categorical = ["Brand", "Processor", "OS", "GPU"]
            bool_cols = ["Has_Dedicated_GPU"]
            # Ensure numeric
            for c in numeric_lr:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            X = df[numeric_lr + categorical + bool_cols]
            ml = model.predict(X)
            df["_ml_score"] = ml
            # Blend with heuristic for budget awareness (model is not budget-aware alone)
            heur = df.apply(lambda r: _heuristic_score(r, user_profile), axis=1)
            # Weighted blend: 0.6 ml + 0.4 heur (heur has budget penalties)
            df["_score"] = df["_ml_score"] * 0.6 + heur * 0.4
            return df
        except Exception as e:
            print(f"ML scoring failed, falling back to heuristic: {e}")
    # Fallback heuristic
    df = df.copy()
    df["_score"] = df.apply(lambda r: _heuristic_score(r, user_profile), axis=1)
    df["_ml_score"] = df["_score"]
    return df


def rank_laptops(rows, user_profile: dict, include_unavailable: bool = False):
    """Rank raw row dicts; returns sorted list with Is_Available handling.

    Available laptops are always sorted first if include_unavailable is True.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    # Filter unavailable if not requested
    if not include_unavailable and "Is_Available" in df.columns:
        # Coerce to bool (csv may have "True"/"False" strings)
        df["_avail"] = df["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[df["_avail"]]
        if df.empty:
            return []
    df = score_laptops(df, user_profile)
    # Availability-first sorting
    if "Is_Available" in df.columns:
        df["_avail_sort"] = df["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
        # Sort by _avail_sort desc, then _score desc, then Rating desc, Price asc
        df = df.sort_values(by=["_avail_sort", "_score", "Rating", "Price"], ascending=[False, False, False, True])
    else:
        df = df.sort_values(by=["_score", "Rating", "Price"], ascending=[False, False, True])
    # Convert back, add tag fields
    df["Is_Available"] = df["Is_Available"].astype(str).str.lower().isin(["true", "1", "yes"]) if "Is_Available" in df.columns else True
    for col in ["_ml_score", "_score", "_avail", "_avail_sort", "LogPrice"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.to_dict(orient="records")
