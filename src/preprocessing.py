import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TARGET_COLUMN,
    SERIAL_COLUMN,
    TARGET_THRESHOLD,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.logger import get_logger

logger = get_logger(__name__)


def prepare_features_and_target(df: pd.DataFrame):
    """
    Prepare X and y from the raw dataframe.
    Converts Admit_Chance into a binary target using TARGET_THRESHOLD.
    Drops Serial_No from features.
    """
    try:
        logger.info("Preparing features and target.")

        df = df.copy()

        df["Admit_Label"] = (df[TARGET_COLUMN] >= TARGET_THRESHOLD).astype(int)

        X = df.drop(columns=[TARGET_COLUMN, "Admit_Label", SERIAL_COLUMN])
        y = df["Admit_Label"]

        logger.info("Features prepared. X shape: %s, y shape: %s", X.shape, y.shape)
        return X, y

    except Exception as e:
        logger.exception("Failed while preparing features and target.")
        raise RuntimeError(f"Error in prepare_features_and_target: {e}") from e


def split_data(X, y):
    """
    Split the dataset into train and test sets using stratification.
    """
    try:
        logger.info("Splitting data into train and test sets.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        logger.info(
            "Split complete. X_train: %s, X_test: %s, y_train: %s, y_test: %s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.exception("Failed while splitting data.")
        raise RuntimeError(f"Error in split_data: {e}") from e


def scale_features(X_train, X_test):
    """
    Scale features using MinMaxScaler.
    """
    try:
        logger.info("Scaling features using MinMaxScaler.")

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logger.info("Feature scaling complete.")
        return X_train_scaled, X_test_scaled, scaler

    except Exception as e:
        logger.exception("Failed while scaling features.")
        raise RuntimeError(f"Error in scale_features: {e}") from e