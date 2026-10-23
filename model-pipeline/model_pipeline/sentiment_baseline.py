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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from model_pipeline.sentiment_dataset import LABELS

NEGATIVE_PATTERN = re.compile(
    r"(不满意|失望|投诉|退款|退货|生气|糟糕|太差|不好|担心|顾虑|风险|被骗|"
    r"不靠谱|太贵|麻烦|很慢|犹豫|拒绝|不需要|不想|不能|不行)"
)
POSITIVE_PATTERN = re.compile(
    r"(满意|喜欢|感谢|谢谢|不错|很好|放心|期待|合适|划算|靠谱|信任|接受|"
    r"愿意|可以|没问题|太好了)"
)


def train_and_evaluate_sentiment(
    data_dir: Path,
    artifact_path: Path,
) -> dict[str, Any]:
    """Train a character TF-IDF classifier and compare it with two simple baselines."""

    datasets = {
        split: _load_jsonl(data_dir / f"{split}.jsonl") for split in ("train", "dev", "test")
    }
    train_records = datasets["train"]
    if not train_records:
        raise ValueError("training split is empty")

    train_labels = [record["label"] for record in train_records]
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
    model.fit([record["text"] for record in train_records], train_labels)

    results: dict[str, Any] = {}
    for split in ("dev", "test"):
        records = datasets[split]
        expected = [record["label"] for record in records]
        results[split] = {
            "majority": evaluate_sentiment_predictions(expected, [majority_label] * len(records)),
            "lexicon_rule": evaluate_sentiment_predictions(
                expected, [lexicon_predict(record["text"]) for record in records]
            ),
            "tfidf_logistic_regression": evaluate_sentiment_predictions(
                expected, model.predict([record["text"] for record in records]).tolist()
            ),
        }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    vectorizer = model.named_steps["tfidf"]
    return {
        "format_version": 1,
        "task": "three-class Chinese customer-message sentiment classification",
        "metric_scope": (
            "Agreement with source-model pseudo-labels on synthetic dialogue; not real-business "
            "accuracy or human-label accuracy."
        ),
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
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "warning": "Only load model artifacts produced by a trusted local pipeline.",
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }


def lexicon_predict(text: str) -> str:
    negative_hits = len(NEGATIVE_PATTERN.findall(text))
    positive_hits = len(POSITIVE_PATTERN.findall(text))
    if negative_hits > positive_hits:
        return "negative"
    if positive_hits > negative_hits:
        return "positive"
    return "neutral"


def evaluate_sentiment_predictions(
    expected: list[str],
    predicted: list[str],
) -> dict[str, Any]:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted lengths must match")
    if not expected:
        raise ValueError("evaluation data must not be empty")
    unknown = (set(expected) | set(predicted)) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown sentiment labels: {sorted(unknown)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(LABELS),
        zero_division=0,
    )
    return {
        "accuracy": round(float(accuracy_score(expected, predicted)), 6),
        "macro_f1": round(float(sum(f1) / len(f1)), 6),
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 6),
                "recall": round(float(recall[index]), 6),
                "f1": round(float(f1[index]), 6),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            expected,
            predicted,
            labels=list(LABELS),
        ).tolist(),
        "label_order": list(LABELS),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if record.get("label") not in LABELS:
            raise ValueError(f"{path}:{line_number}: unknown sentiment label")
        if not isinstance(record.get("text"), str) or not record["text"]:
            raise TypeError(f"{path}:{line_number}: text must be a non-empty string")
        records.append(record)
    return records
