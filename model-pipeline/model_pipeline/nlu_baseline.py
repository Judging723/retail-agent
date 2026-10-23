import hashlib
import json
import platform
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

KEYWORD_PATTERN = re.compile(
    r"(推荐|介绍(?:一下)?|看看|看一下|有什么|有哪些|哪(?:个|种|款|些)|"
    r"这个|那个|这种|那种|一样|随便|都可以|合适|适合|怎么选|如何选|"
    r"有吗|有没有|想买|想要|我要(?:买|找|看)|要一(?:个|件|份|款))"
)


def train_and_evaluate(
    data_dir: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    """Fit three baselines and evaluate without touching the held-out labels during training."""

    datasets = {
        split: _load_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "dev", "test")
    }
    train_records = datasets["train"]
    if not train_records:
        raise ValueError("training split is empty")

    train_labels = [int(record["label"]) for record in train_records]
    majority_label = Counter(train_labels).most_common(1)[0][0]
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(
        [record["context"] for record in train_records],
        train_labels,
    )

    results: dict[str, Any] = {}
    for split in ("dev", "test"):
        records = datasets[split]
        expected = [int(record["label"]) for record in records]
        results[split] = {
            "majority": evaluate_predictions(expected, [majority_label] * len(records)),
            "keyword_rule": evaluate_predictions(
                expected,
                [int(bool(KEYWORD_PATTERN.search(record["text"]))) for record in records],
            ),
            "tfidf_logistic_regression": evaluate_predictions(
                expected,
                model.predict([record["context"] for record in records]).tolist(),
            ),
        }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    artifact_bytes = artifact_path.read_bytes()
    vectorizer = model.named_steps["tfidf"]
    return {
        "format_version": 1,
        "task": "binary clarification detection",
        "input": {
            split: {
                "file": f"{split}.jsonl",
                "records": len(datasets[split]),
                "sha256": hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest(),
            }
            for split in ("train", "dev", "test")
        },
        "training": {
            "features": "character TF-IDF n-grams",
            "ngram_range": [2, 5],
            "min_df": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "train_majority_label": majority_label,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "results": results,
        "artifact": {
            "file": artifact_path.name,
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
            "warning": "Only load model artifacts produced by a trusted local pipeline.",
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }


def evaluate_predictions(expected: list[int], predicted: list[int]) -> dict[str, Any]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths must match")
    if not expected:
        raise ValueError("evaluation data must not be empty")

    precision, recall, f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        average="binary",
        zero_division=0,
    )
    _, _, macro_f1, _ = precision_recall_fscore_support(
        expected,
        predicted,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(expected, predicted)), 6),
        "positive_precision": round(float(precision), 6),
        "positive_recall": round(float(recall), 6),
        "positive_f1": round(float(f1), 6),
        "macro_f1": round(float(macro_f1), 6),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=[0, 1]).tolist(),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if record.get("label") not in {0, 1}:
            raise ValueError(f"{path}:{line_number}: label must be 0 or 1")
        if not isinstance(record.get("text"), str) or not isinstance(record.get("context"), str):
            raise TypeError(f"{path}:{line_number}: text and context must be strings")
        records.append(record)
    return records
