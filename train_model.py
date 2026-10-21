from pathlib import Path
import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def train() -> Path:
    """
    Executes the training workflow:
    1. Loads dataset and extracts features/labels.
    2. Performs stratified train-test split.
    3. Fits a RandomForestClassifier.
    4. Evaluates accuracy on the held-out test split.
    5. Saves the model bundle to the artifacts directory.

    Returns:
        Path: The absolute path to the saved model artifact.
    """
    # 1. Load the Iris dataset
    iris = load_iris()

    # 2. Separate features (X) and target labels (y)
    X = iris.data
    y = iris.target

    # 3. Stratified split into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 4. Train RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # 5. Evaluate the model on test set
    y_pred = model.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    # 6. Ensure artifacts directory exists relative to this script
    base_dir = Path(__file__).resolve().parent
    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 7. Prepare the model bundle with metadata
    model_version = "1.0.0"
    model_bundle = {
        "model": model,
        "target_names": [str(name) for name in iris.target_names],
        "feature_names": [str(name) for name in iris.feature_names],
        "test_accuracy": test_accuracy,
        "model_version": model_version,
    }

    # 8. Save model bundle using Joblib
    model_path = artifacts_dir / "iris_model.joblib"
    joblib.dump(model_bundle, model_path)

    # 9. Print training summary details
    print("=" * 60)
    print("Iris Model Training & Evaluation Summary")
    print("=" * 60)
    print(f"Dataset Size:           {X.shape[0]} samples ({X.shape[1]} features)")
    print(f"Training Record Count:  {X_train.shape[0]} samples")
    print(f"Testing Record Count:   {X_test.shape[0]} samples")
    print(f"Model Type:             {model.__class__.__name__}")
    print(f"Number of Trees:        {model.n_estimators}")
    print(f"Actual Test Accuracy:   {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")
    print(f"Model Version:          {model_version}")
    print(f"Saved Model Path:       {model_path.resolve()}")
    print("=" * 60)

    return model_path


def main():
    """Main entrypoint for script execution."""
    train()


if __name__ == "__main__":
    main()
