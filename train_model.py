from __future__ import annotations

import argparse
from pathlib import Path

from cyberbullying_detector.model import train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a cyberbullying detector model.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/Dataset_for_Detection_of_Cyber-Trolls_1.json"),
        help="Path to the labeled JSON-lines dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cyberbully_baseline.joblib"),
        help="Where to save the trained model.",
    )
    args = parser.parse_args()

    result = train_and_evaluate(args.dataset, args.output)
    print(f"Saved model to: {args.output}")
    print(f"Training rows: {result['train_examples']}")
    print(f"Test rows: {result['test_examples']}")
    print(result["classification_report"])


if __name__ == "__main__":
    main()
