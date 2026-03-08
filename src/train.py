import joblib
import pandas as pd

from src.config import MODEL_BUNDLE_FILE, ARTIFACTS_DIR
from src.data_loader import load_and_validate_data
from src.preprocessing import prepare_features_and_target, split_data, scale_features
from src.model import get_candidate_models
from src.evaluate import evaluate_model
from src.logger import get_logger

logger = get_logger(__name__)


def train_and_save_best_model():
    """
    Full training pipeline:
    1. Load and validate data
    2. Prepare features and target
    3. Split into train/test
    4. Scale features
    5. Train candidate models
    6. Evaluate and compare models
    7. Save the best model bundle
    """
    try:
        logger.info("Starting training pipeline.")

        df = load_and_validate_data()
        X, y = prepare_features_and_target(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        candidate_models = get_candidate_models()
        results = []
        best_model = None
        best_metrics = None
        best_model_name = None
        best_f1 = -1.0

        for model_name, model in candidate_models.items():
            logger.info("Training model: %s", model_name)

            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test)

            row = {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
            }
            results.append(row)

            logger.info(
                "Model: %s | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f | ROC-AUC: %.4f",
                model_name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["roc_auc"],
            )

            if metrics["f1_score"] > best_f1:
                best_f1 = metrics["f1_score"]
                best_model = model
                best_metrics = metrics
                best_model_name = model_name

        results_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)

        logger.info("Model comparison completed.")
        logger.info("\n%s", results_df.to_string(index=False))

        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        model_bundle = {
            "model": best_model,
            "scaler": scaler,
            "feature_columns": X.columns.tolist(),
            "best_model_name": best_model_name,
            "metrics": {
                "accuracy": best_metrics["accuracy"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "f1_score": best_metrics["f1_score"],
                "roc_auc": best_metrics["roc_auc"],
                "confusion_matrix": best_metrics["confusion_matrix"],
                "classification_report": best_metrics["classification_report"],
            },
        }

        joblib.dump(model_bundle, MODEL_BUNDLE_FILE)

        logger.info("Best model saved successfully to: %s", MODEL_BUNDLE_FILE)
        logger.info("Best model selected: %s", best_model_name)

        return results_df, model_bundle

    except Exception as e:
        logger.exception("Training pipeline failed.")
        raise RuntimeError(f"Training failed: {e}") from e


if __name__ == "__main__":
    results_df, model_bundle = train_and_save_best_model()

    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    print("\n=== Best Model ===")
    print(model_bundle["best_model_name"])

    print("\n=== Best Model Metrics ===")
    for key, value in model_bundle["metrics"].items():
        if key not in ["confusion_matrix", "classification_report"]:
            print(f"{key}: {value}")

    print("\nConfusion Matrix:")
    print(model_bundle["metrics"]["confusion_matrix"])

    print("\nClassification Report:")
    print(model_bundle["metrics"]["classification_report"])