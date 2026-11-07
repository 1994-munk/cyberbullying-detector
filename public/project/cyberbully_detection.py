"""Train and evaluate a text classifier for cyber-aggressive tweets.

This is a learning baseline, not a system for making moderation decisions.
It expects the original DataTurks JSON-lines dataset beside this script.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Keep the dataset in this project folder; it is not bundled with the code.
DATA_PATH = Path("Dataset_for_Detection_of_Cyber-Trolls_1.json")
MODEL_PATH = Path("cyberbully_baseline.joblib")
RANDOM_STATE = 42


def read_dataset(path: Path) -> pd.DataFrame:
    """Read one JSON record per line and extract text and its first label."""
    raw = pd.read_json(path, lines=True)

    # Labels in the supplied file are nested as annotation.label: ["0" or "1"].
    def first_label(annotation: object) -> str | None:
        if not isinstance(annotation, dict):
            return None
        labels = annotation.get("label")
        if not isinstance(labels, list) or not labels:
            return None
        return str(labels[0]).strip()

    data = pd.DataFrame(
        {
            "text": raw.get("content"),
            "label": raw.get("annotation", pd.Series(index=raw.index, dtype=object)).map(
                first_label
            ),
        }
    )
    return data


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize text, remove unusable rows, and audit duplicate labels."""
    cleaned = data.copy()

    # Keep punctuation because it can affect tone; standardize HTML and spacing.
    cleaned["text"] = cleaned["text"].fillna("").astype(str).map(
        lambda text: " ".join(html.unescape(text).split())
    )
    cleaned["text_key"] = cleaned["text"].str.casefold()
    cleaned["label"] = cleaned["label"].astype("string").str.strip()

    # Keep only the two documented labels and non-empty messages.
    cleaned = cleaned[
        cleaned["text_key"].ne("") & cleaned["label"].isin(["0", "1"])
    ].copy()
    cleaned["label"] = cleaned["label"].astype(int)

    # Some repeated messages have different labels. Exclude those ambiguous groups.
    conflicting_text = cleaned.groupby("text_key")["label"].transform("nunique").gt(1)
    conflict_count = cleaned.loc[conflicting_text, "text_key"].nunique()
    cleaned = cleaned.loc[~conflicting_text].drop_duplicates("text_key", keep="first")

    print(f"Ambiguous repeated messages removed: {conflict_count:,}")
    print(f"Usable unique messages: {len(cleaned):,}")
    print("Class counts (0 = non-aggressive, 1 = cyber-aggressive):")
    print(cleaned["label"].value_counts().sort_index().to_string())

    return cleaned[["text", "label"]].reset_index(drop=True)


def build_model() -> Pipeline:
    """Build a simple, explainable text-classification pipeline."""
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


def main() -> None:
    """Run every project stage and save the trained pipeline."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Place the JSON-lines file beside this script."
        )

    # Stage 1: load and inspect the raw records.
    data = read_dataset(DATA_PATH)
    print(f"Raw records: {len(data):,}")

    # Stage 2: clean text and audit repeated or ambiguous labels.
    data = clean_dataset(data)
    if data["label"].nunique() < 2:
        raise ValueError("Both label classes are needed to train and evaluate the model.")

    # Stage 3: hold out test examples before learning vocabulary or model weights.
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )
    print(f"Training examples: {len(X_train):,}; test examples: {len(X_test):,}")

    # Stage 4: compare with a majority-class baseline, then train TF-IDF + logistic regression.
    majority_baseline = DummyClassifier(strategy="most_frequent")
    majority_baseline.fit(X_train, y_train)
    baseline_predictions = majority_baseline.predict(X_test)
    print("\nMajority-only baseline (accuracy can hide missed aggressive examples):")
    print(classification_report(y_test, baseline_predictions, zero_division=0))

    model = build_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Stage 5: report class-specific results; class 1 is the positive class.
    print("\nTF-IDF + logistic regression results:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["non-aggressive (0)", "cyber-aggressive (1)"],
        zero_division=0,
    ))
    print("Confusion matrix (rows = actual, columns = predicted; labels 0, 1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    # Save the complete pipeline so new text uses the same learned vocabulary.
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH}")

    # Example only: predictions are uncertain and should never decide moderation alone.
    example = "You never listen to anyone."
    predicted_label = int(model.predict([example])[0])
    confidence = float(model.predict_proba([example])[0][predicted_label])
    print(f"Example prediction: label {predicted_label}, model score {confidence:.2f}")
    print("Review predictions in context; this model does not determine bullying.")


if __name__ == "__main__":
    main()