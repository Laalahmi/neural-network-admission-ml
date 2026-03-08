import pandas as pd
from src.config import DATA_FILE, REQUIRED_COLUMNS
from src.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path=DATA_FILE) -> pd.DataFrame:
    """
    Load the admissions dataset from CSV.
    """
    try:
        logger.info("Loading dataset from: %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df
    except FileNotFoundError as e:
        logger.exception("Dataset file not found.")
        raise FileNotFoundError(f"Dataset file not found at: {file_path}") from e
    except Exception as e:
        logger.exception("Unexpected error while loading dataset.")
        raise RuntimeError(f"Failed to load dataset: {e}") from e


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required columns exist in the dataset.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_columns:
        logger.error("Missing required columns: %s", missing_columns)
        raise ValueError(f"Missing required columns: {missing_columns}")

    logger.info("All required columns are present.")


def load_and_validate_data(file_path=DATA_FILE) -> pd.DataFrame:
    """
    Load dataset and validate required columns.
    """
    df = load_data(file_path)
    validate_columns(df)
    return df