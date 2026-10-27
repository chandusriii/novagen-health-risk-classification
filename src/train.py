"""Train and compare baseline classifiers for the NovaGen assignment dataset.

Educational use only. This project is not a clinical decision system.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "Target"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the CSV without changing the source file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Pass --data-path.")
    data = pd.read_csv(path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Required target column {TARGET_COLUMN!r} is missing.")
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values.")
    bool_columns = data.select_dtypes(include=["bool"]).columns
    if len(bool_columns):
        data = data.astype({column: "int64" for column in bool_columns})
    non_numeric = data.drop(columns=[TARGET_COLUMN]).select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric features need encoding: {non_numeric}")
    if data[TARGET_COLUMN].nunique() != 2:
        raise ValueError("This project expects a binary target with two classes.")
    if data.drop(columns=[TARGET_COLUMN]).isna().any().any():
        raise ValueError("Features contain missing values; add an imputation strategy first.")
    return data


def build_models() -> dict[str, Any]:
    """Construct the five candidate classifiers used in this assignment."""
    estimators = {
        "Logistic Regression": LogisticRegression(solver="liblinear", max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
        "Soft Voting Classifier": VotingClassifier(estimators=[
            ("lr", LogisticRegression(max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
            ("rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ], voting="soft"),
    }
    # Pipeline ensures the scaler is fitted only on training samples.
    return {name: Pipeline([("scaler", StandardScaler()), ("classifier", estimator)]) for name, estimator in estimators.items()}


def evaluate_models(data: pd.DataFrame, output_dir: str | Path, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Fit/evaluate each candidate and persist metrics, plot, metadata and selected model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    positive_class = 1 if 1 in set(y.unique()) else sorted(y.unique())[-1]
    results, fitted_models, reports, confusions = [], {}, {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        classes = list(model.classes_) if probabilities is not None else []
        positive_scores = probabilities[:, classes.index(positive_class)] if probabilities is not None and positive_class in classes else None
        results.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_class_1": float(precision_score(y_test, predictions, pos_label=positive_class, zero_division=0)),
            "recall_class_1": float(recall_score(y_test, predictions, pos_label=positive_class, zero_division=0)),
            "f1_class_1": float(f1_score(y_test, predictions, pos_label=positive_class, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, positive_scores)) if positive_scores is not None and y_test.nunique() == 2 else float("nan"),
        })
        fitted_models[name] = model
        reports[name] = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        confusions[name] = confusion_matrix(y_test, predictions).tolist()
    scores = pd.DataFrame(results).sort_values("recall_class_1", ascending=False).reset_index(drop=True)
    scores.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    (output_dir / "confusion_matrices.json").write_text(json.dumps(confusions, indent=2), encoding="utf-8")
    metric_frame = scores.melt(id_vars="model", value_vars=["accuracy", "precision_class_1", "recall_class_1", "f1_class_1"], var_name="metric", value_name="score")
    pivot = metric_frame.pivot(index="model", columns="metric", values="score").reindex(scores["model"].tolist())
    axis = pivot.plot(kind="bar", figsize=(11, 6))
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_xlabel("")
    axis.set_title("NovaGen: Classifier Comparison on a Stratified Hold-out Set")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()
    best_name = str(scores.iloc[0]["model"])
    joblib.dump({"model": fitted_models[best_name], "model_name": best_name, "feature_columns": list(X.columns), "target_column": TARGET_COLUMN, "positive_class_label": positive_class, "disclaimer": "Educational demonstration only; not for clinical use."}, output_dir / "best_model.joblib")
    metadata = {
        "rows": int(data.shape[0]), "features": int(X.shape[1]), "target_column": TARGET_COLUMN,
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "selected_model_by_recall_class_1": best_name,
        "selection_metric_note": "Label 1 is numeric only; the assignment does not define which health outcome it means.",
        "clinical_use": False,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return scores, fitted_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="data/novagen_dataset.csv", help="Path to NovaGen CSV.")
    parser.add_argument("--output-dir", default="reports", help="Output directory for metrics and artifacts.")
    args = parser.parse_args()
    scores, _ = evaluate_models(load_dataset(args.data_path), args.output_dir)
    print("\nModel comparison (sorted by recall for label 1):")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved metrics and artifacts to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
