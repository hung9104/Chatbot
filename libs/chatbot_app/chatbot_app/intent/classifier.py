import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


REQUIRED_COLUMNS = {"cau_hoi", "intent"}


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_intent_dataset(csv_path: str | Path) -> tuple[list[str], list[str], list[str]]:
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_file}")

    with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("Dataset is missing header row")

        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Dataset missing columns: {sorted(missing)}")

        questions: list[str] = []
        intents: list[str] = []
        answers: list[str] = []

        for row in reader:
            question = normalize_text(row.get("cau_hoi", ""))
            intent = normalize_text(row.get("intent", ""))
            if not question or not intent:
                continue
            questions.append(question)
            intents.append(intent)
            answers.append((row.get("tra_loi") or "").strip())

    if not questions:
        raise ValueError("No valid training rows found in dataset")

    return questions, intents, answers


def build_answer_map(intents: list[str], answers: list[str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for intent, answer in zip(intents, answers):
        if answer:
            grouped[intent].append(answer)

    answer_map: dict[str, str] = {}
    for intent, response_list in grouped.items():
        answer_map[intent] = Counter(response_list).most_common(1)[0][0]
    return answer_map


def train_intent_model(
    dataset_csv: str | Path,
    output_model_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    questions, intents, answers = load_intent_dataset(dataset_csv)
    answer_map = build_answer_map(intents, answers)

    x_train, x_test, y_train, y_test = train_test_split(
        questions,
        intents,
        test_size=test_size,
        random_state=random_state,
        stratify=intents,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=1,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    artifact = {
        "model": pipeline,
        "answer_map": answer_map,
        "metadata": {
            "num_samples": len(questions),
            "num_intents": len(set(intents)),
            "test_size": test_size,
            "random_state": random_state,
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "accuracy": report["accuracy"],
        },
    }

    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)

    return artifact["metadata"]


def predict_intent(
    question: str,
    model_path: str | Path,
    min_confidence: float = 0.2,
) -> dict[str, Any]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    artifact = joblib.load(model_file)
    model: Pipeline = artifact["model"]
    answer_map: dict[str, str] = artifact.get("answer_map", {})

    query = normalize_text(question)
    if not query:
        return {
            "intent": None,
            "confidence": 0.0,
            "response": None,
            "fallback": True,
        }

    proba = model.predict_proba([query])[0]
    classes = model.classes_
    best_idx = int(proba.argmax())
    confidence = float(proba[best_idx])
    intent = str(classes[best_idx])

    if confidence < min_confidence:
        return {
            "intent": intent,
            "confidence": confidence,
            "response": None,
            "fallback": True,
        }

    return {
        "intent": intent,
        "confidence": confidence,
        "response": answer_map.get(intent),
        "fallback": False,
    }


class IntentClassifier:
    def __init__(self, model_path: str | Path, min_confidence: float = 0.2):
        self.model_path = Path(model_path)
        self.min_confidence = min_confidence

    def predict(self, question: str) -> dict[str, Any]:
        return predict_intent(
            question=question,
            model_path=self.model_path,
            min_confidence=self.min_confidence,
        )


