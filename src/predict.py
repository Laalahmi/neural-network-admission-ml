import joblib
import pandas as pd

from src.config import MODEL_BUNDLE_FILE
from src.logger import get_logger

logger = get_logger(__name__)


def load_model_bundle(bundle_path=MODEL_BUNDLE_FILE):
    """
    Load the saved model bundle.
    """
    try:
        logger.info("Loading model bundle from: %s", bundle_path)
        bundle = joblib.load(bundle_path)
        logger.info("Model bundle loaded successfully.")
        return bundle
    except FileNotFoundError as e:
        logger.exception("Model bundle file not found.")
        raise FileNotFoundError(f"Model bundle not found at: {bundle_path}") from e
    except Exception as e:
        logger.exception("Failed to load model bundle.")
        raise RuntimeError(f"Could not load model bundle: {e}") from e


def predict_admission(input_data: dict, bundle_path=MODEL_BUNDLE_FILE):
    """
    Predict admission class and probability from a single input dictionary.
    """
    try:
        bundle = load_model_bundle(bundle_path)

        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_columns = bundle["feature_columns"]

        input_df = pd.DataFrame([input_data])

        input_df = input_df[feature_columns]

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        result = {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "Likely Admit" if prediction == 1 else "Unlikely Admit",
        }

        logger.info("Prediction completed successfully.")
        return result

    except Exception as e:
        logger.exception("Prediction failed.")
        raise RuntimeError(f"Prediction failed: {e}") from e