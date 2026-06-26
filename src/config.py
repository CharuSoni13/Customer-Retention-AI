from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
IMAGES_DIR = ROOT / "images"

RAW_CSV = RAW_DIR / "telco_churn.csv"
PROCESSED_CSV = PROCESSED_DIR / "telco_processed.csv"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR):
    d.mkdir(parents=True, exist_ok=True)
