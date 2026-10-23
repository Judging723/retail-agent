import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_pipeline.sentiment_dataset import LABELS

DEFAULT_MAX_LENGTH = 192
_ESCALATION_PATTERN = re.compile(r"(投诉|生气|愤怒|报警|起诉|诈骗|欺骗|威胁|曝光|差评)")


class SentimentPredictor:
    """Load one local checkpoint once and reuse it for sentiment predictions."""

    def __init__(self, model_dir: Path, *, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"trained model not found at {model_dir}; run train_sentiment_transformer.py first"
            )
        if max_length < 8:
            raise ValueError("max_length must be at least 8")
        self._max_length = max_length
        self._device = select_device()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        ).to(self._device)
        self._model.eval()

    @property
    def device(self) -> str:
        return str(self._device)

    def predict(self, text: str) -> dict[str, Any]:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("text must not be empty")

        encoded = self._tokenizer(
            normalized_text,
            max_length=self._max_length,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self._device) for name, value in encoded.items()}
        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            scores = torch.softmax(logits, dim=-1).cpu().tolist()

        probabilities = {
            label: round(float(scores[index]), 6) for index, label in enumerate(LABELS)
        }
        sentiment = max(LABELS, key=probabilities.__getitem__)
        risk_level, risk_hint = emotion_risk(
            text=normalized_text,
            sentiment=sentiment,
            negative_probability=probabilities["negative"],
        )
        return {
            "text": normalized_text,
            "sentiment": sentiment,
            "confidence": probabilities[sentiment],
            "probabilities": probabilities,
            "risk_level": risk_level,
            "risk_hint": risk_hint,
            "device": self.device,
        }


def emotion_risk(
    *,
    text: str,
    sentiment: str,
    negative_probability: float,
) -> tuple[str, str]:
    """Map sentiment output to a simple, uncalibrated customer-attention hint."""

    if sentiment not in LABELS:
        raise ValueError(f"unknown sentiment: {sentiment}")
    if not 0 <= negative_probability <= 1:
        raise ValueError("negative_probability must be in [0, 1]")
    if sentiment == "negative" and _ESCALATION_PATTERN.search(text):
        return "high", "检测到负面情绪和明确升级信号，建议优先安抚并人工关注。"
    if sentiment == "negative" or negative_probability >= 0.4:
        return "medium", "存在一定负面情绪，建议回复前确认用户诉求。"
    return "low", "未检测到明显负面情绪，可按常规流程处理。"


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
