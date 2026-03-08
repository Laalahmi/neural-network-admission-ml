from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

# Files
DATA_FILE = DATA_DIR / "Admission.csv"
MODEL_BUNDLE_FILE = ARTIFACTS_DIR / "model_bundle.joblib"
LOG_FILE = LOGS_DIR / "project.log"

# Dataset settings
TARGET_COLUMN = "Admit_Chance"
SERIAL_COLUMN = "Serial_No"
TARGET_THRESHOLD = 0.80

# Training settings
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Columns
CATEGORICAL_COLUMNS = ["University_Rating", "Research"]
NUMERICAL_COLUMNS = ["GRE_Score", "TOEFL_Score", "SOP", "LOR", "CGPA"]

# Required columns in the raw dataset
REQUIRED_COLUMNS = [
    "Serial_No",
    "GRE_Score",
    "TOEFL_Score",
    "University_Rating",
    "SOP",
    "LOR",
    "CGPA",
    "Research",
    "Admit_Chance",
]