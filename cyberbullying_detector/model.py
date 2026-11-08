from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Load raw tweet text and nested label values from JSON-lines input."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    raw = pd.read_json(dataset_path, lines=True)

    def first_label(annotation: object) -> str | None:
        if not isinstance(annotation, dict):
            return None
        labels = annotation.get("label")
        if not isinstance(labels, list) or not labels:
            return None
        return str(labels[0]).strip()

    return pd.DataFrame(
        {
            TEXT_COLUMN: raw.get("content"),
            LABEL_COLUMN: raw.get("annotation", pd.Series(index=raw.index, dtype=object)).map(first_label),
        }
    )


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize text and filter to valid, non-conflicting labels."""
    cleaned = data.copy()
    cleaned[TEXT_COLUMN] = cleaned[TEXT_COLUMN].fillna("").astype(str).map(
        lambda text: " ".join(html.unescape(text).split())
    )
    cleaned["text_key"] = cleaned[TEXT_COLUMN].str.casefold()
    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype("string").str.strip()

    cleaned = cleaned[
        cleaned["text_key"].ne("") & cleaned[LABEL_COLUMN].isin(["0", "1"])
    ].copy()
    cleaned[LABEL_COLUMN] = cleaned[LABEL_COLUMN].astype(int)

    conflicting_text = cleaned.groupby("text_key")[LABEL_COLUMN].transform("nunique").gt(1)
    cleaned = cleaned.loc[~conflicting_text].drop_duplicates("text_key", keep="first")

    return cleaned[[TEXT_COLUMN, LABEL_COLUMN]].reset_index(drop=True)


def build_model() -> Pipeline:
    """Construct a strong baseline using TF-IDF and logistic regression."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate(dataset_path: str | Path, model_path: str | Path = "cyberbully_baseline.joblib") -> dict[str, Any]:
    """Train the model, evaluate it, and save the pipeline."""
    data = clean_dataset(read_dataset(dataset_path))
    if data[LABEL_COLUMN].nunique() < 2:
        raise ValueError("Both label classes are required to train and evaluate the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        data[TEXT_COLUMN],
        data[LABEL_COLUMN],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data[LABEL_COLUMN],
    )

    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(X_train, y_train)
    baseline_report = classification_report(
        y_test,
        majority_baseline.predict(X_test),
        target_names=["non-aggressive", "cyber-aggressive"],
        zero_division=0,
    )

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["non-aggressive", "cyber-aggressive"],
        zero_division=0,
    )

    joblib.dump(model, Path(model_path))

    return {
        "train_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "classification_report": report,
        "baseline_report": baseline_report,
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a cyberbullying detector model.")
    parser.add_argument("--dataset", type=Path, default=Path("data/Dataset_for_Detection_of_Cyber-Trolls_1.json"), help="Path to the JSON-lines dataset.")
    parser.add_argument("--output", type=Path, default=Path("cyberbully_baseline.joblib"), help="Model output path.")
    args = parser.parse_args()

    result = train_and_evaluate(args.dataset, args.output)
    print(result["classification_report"])
    print(result["confusion_matrix"])


if __name__ == "__main__":
    main()
