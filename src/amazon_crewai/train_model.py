import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

from .paths import CLEAN_CSV, METADATA_JSON, MODEL_PATH, MODELS_DIR

CLEAN_PATH = str(CLEAN_CSV)
METADATA_PATH = str(METADATA_JSON)


def _value_score(row):
    # Higher rating and more RAM/storage per rupee score better.
    # Normalize price log to avoid domination
    price = float(row["Price"])
    rating = float(row["Rating"])
    ram = int(row["RAM"])
    ssd = int(row["SSD_Storage"])
    dedicated = bool(row.get("Has_Dedicated_GPU", False))
    # Value = rating * 20 + ram*0.5 + log(ssd)*2 - log(price)*5 + dedicated*3
    # Keeps score in ~0-30 range
    import math
    score = rating * 4.0
    score += math.log1p(ram) * 1.2
    score += math.log1p(ssd / 256) * 0.8
    score -= math.log(price / 10000) * 1.1
    if dedicated:
        score += 1.5
    # OS bonus: MacOS premium perception
    if str(row.get("OS")) == "MacOS":
        score += 0.5
    return float(score)


def train():
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"Clean dataset not found at {CLEAN_PATH}")
    df = pd.read_csv(CLEAN_PATH)
    if df.empty:
        raise ValueError("Dataset is empty")

    # Ensure required columns
    for col in ["GPU", "Has_Dedicated_GPU", "Is_Available"]:
        if col not in df.columns:
            df[col] = "Unknown" if col == "GPU" else (False if col == "Has_Dedicated_GPU" else True)

    # Target: value_score
    df["_value"] = df.apply(_value_score, axis=1)

    # Features
    numeric = ["Price", "RAM", "SSD_Storage", "Rating"]
    # Log price for better spread
    df["LogPrice"] = np.log1p(df["Price"])
    numeric_lr = ["LogPrice", "RAM", "SSD_Storage", "Rating"]

    categorical = ["Brand", "Processor", "OS", "GPU"]
    bool_cols = ["Has_Dedicated_GPU"]

    # Fill NA
    for c in categorical:
        df[c] = df[c].fillna("Unknown").astype(str)
    df["Has_Dedicated_GPU"] = df["Has_Dedicated_GPU"].astype(int)

    X = df[numeric_lr + categorical + bool_cols]
    y = df["_value"]

    # Preprocess
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_lr),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("bool", "passthrough", bool_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline([("pre", pre), ("reg", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)

    # Evaluate
    pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"Trained on {len(df)} rows. MSE={mse:.4f} R2={r2:.4f}")

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    # Update metadata
    try:
        import datetime
        meta = {}
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                meta = json.load(f)
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        meta["last_trained"] = now
        meta["last_trained_local"] = datetime.datetime.now().astimezone().isoformat()
        meta["last_updated"] = now
        meta["last_updated_local"] = datetime.datetime.now().astimezone().isoformat()
        meta["model_metrics"] = {"mse": float(mse), "r2": float(r2), "rows": len(df)}
        meta["model_path"] = str(MODEL_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Updated metadata at {METADATA_PATH}")
    except Exception as e:
        print(f"Metadata update failed: {e}")

    return pipe


if __name__ == "__main__":
    train()
