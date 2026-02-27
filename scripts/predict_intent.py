import argparse
import json
from pathlib import Path

from chatbot_app.intent import predict_intent


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict intent from text")
    parser.add_argument("question", help="Question text")
    parser.add_argument(
        "--model",
        default="data/models/intent_classifier.joblib",
        help="Path to trained model artifact",
    )
    parser.add_argument("--min-confidence", type=float, default=0.2)
    args = parser.parse_args()

    result = predict_intent(
        question=args.question,
        model_path=Path(args.model),
        min_confidence=args.min_confidence,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

