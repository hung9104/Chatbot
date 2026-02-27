import argparse
import json
from pathlib import Path

from chatbot_app.intent import train_intent_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train intent classification model")
    parser.add_argument(
        "--dataset",
        default="data/intent_expanded_filled_v2.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--output",
        default="data/models/intent_classifier.joblib",
        help="Path to output model artifact",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_intent_model(
        dataset_csv=Path(args.dataset),
        output_model_path=Path(args.output),
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
