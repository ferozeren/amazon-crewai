from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = REPO_ROOT / "data" / "raw_data"
DATA_PROCESSED = REPO_ROOT / "data" / "processed_data"
RAW_CSV = DATA_RAW / "amazon_laptops.csv"
CLEAN_CSV = DATA_PROCESSED / "Clean Dataset.csv"
METADATA_JSON = DATA_PROCESSED / "metadata.json"

ASSETS_ROOT = REPO_ROOT / "assets"
ASSETS_DIR = ASSETS_ROOT / "laptops"
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "laptop_ranker.pkl"
FRONTEND_DIR = REPO_ROOT / "frontend"
ENV_PATH = REPO_ROOT / ".env"
