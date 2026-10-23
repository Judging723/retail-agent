import hashlib
import json
import platform
import random
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import transformers
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model_pipeline.sentiment_baseline import evaluate_sentiment_predictions
from model_pipeline.sentiment_dataset import LABELS

MODEL_NAME = "hfl/rbt3"
MODEL_REVISION = "0aa0527ff4170f29e1dfd3eb6ef60dc67e1bf75c"
MODEL_LICENSE = "Apache-2.0"


class EncodedSentimentDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(
            [record["text"] for record in records],
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        self.labels = torch.tensor(
            [LABELS.index(record["label"]) for record in records],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {name: values[index] for name, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def train_transformer_sentiment(
    data_dir: Path,
    artifact_dir: Path,
    *,
    epochs: int = 2,
    batch_size: int = 16,
    max_length: int = 192,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune the pinned three-layer Chinese RoBERTa sentiment classifier."""

    if epochs < 1 or batch_size < 1 or max_length < 8 or learning_rate <= 0:
        raise ValueError("training arguments must be positive")
    _set_seed(seed)
    records = {split: _load_jsonl(data_dir / f"{split}.jsonl") for split in LABEL_SPLITS}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    datasets = {
        split: EncodedSentimentDataset(split_records, tokenizer, max_length)
        for split, split_records in records.items()
    }
    train_generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
        ),
        "dev": DataLoader(datasets["dev"], batch_size=batch_size),
        "test": DataLoader(datasets["test"], batch_size=batch_size),
    }

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    device = _select_device()
    model.to(device)
    class_weights = balanced_class_weights(
        [LABELS.index(record["label"]) for record in records["train"]],
        len(LABELS),
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    started_at = perf_counter()
    history: list[dict[str, Any]] = []
    best_dev_macro_f1 = -1.0
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            if step % 25 == 0 or step == len(loaders["train"]):
                print(
                    f"epoch={epoch}/{epochs} step={step}/{len(loaders['train'])} "
                    f"train_loss={total_loss / step:.4f}",
                    flush=True,
                )

        dev_result = evaluate_model(model, loaders["dev"], device, loss_function)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / len(loaders["train"]), 6),
                "dev_loss": dev_result["loss"],
                "dev_accuracy": dev_result["metrics"]["accuracy"],
                "dev_macro_f1": dev_result["metrics"]["macro_f1"],
            }
        )
        print(
            f"epoch={epoch} dev_accuracy={dev_result['metrics']['accuracy']:.3f} "
            f"dev_macro_f1={dev_result['metrics']['macro_f1']:.3f}",
            flush=True,
        )
        if dev_result["metrics"]["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_result["metrics"]["macro_f1"]
            model.save_pretrained(artifact_dir, safe_serialization=True)
            tokenizer.save_pretrained(artifact_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(artifact_dir).to(device)
    dev_result = evaluate_model(best_model, loaders["dev"], device, loss_function)
    test_result = evaluate_model(best_model, loaders["test"], device, loss_function)
    parameter_count = sum(parameter.numel() for parameter in best_model.parameters())

    return {
        "format_version": 1,
        "task": "three-class Chinese customer-message sentiment fine-tuning",
        "metric_scope": (
            "Agreement with source-model pseudo-labels on synthetic dialogue; not real-business "
            "accuracy or human-label accuracy."
        ),
        "base_model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "architecture": "3-layer Chinese RoBERTa",
        },
        "input": {
            split: {
                "file": f"{split}.jsonl",
                "records": len(records[split]),
                "sha256": hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest(),
            }
            for split in LABEL_SPLITS
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "class_weight": "balanced",
            "gradient_clip_norm": 1.0,
            "seed": seed,
            "device": str(device),
            "parameter_count": parameter_count,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in best_model.parameters()
                if parameter.requires_grad
            ),
            "elapsed_seconds": round(perf_counter() - started_at, 3),
            "history": history,
        },
        "results": {
            "dev": dev_result,
            "test": test_result,
        },
        "artifact": {
            "directory": artifact_dir.name,
            "files": sorted(path.name for path in artifact_dir.iterdir() if path.is_file()),
            "warning": "Only load model artifacts produced by a trusted local pipeline.",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }


def balanced_class_weights(labels: list[int], class_count: int) -> torch.Tensor:
    if not labels or class_count < 1:
        raise ValueError("labels and class_count must not be empty")
    counts = Counter(labels)
    if set(counts) - set(range(class_count)) or len(counts) != class_count:
        raise ValueError("every configured class must occur in the training labels")
    total = len(labels)
    return torch.tensor(
        [total / (class_count * counts[index]) for index in range(class_count)],
        dtype=torch.float32,
    )


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    loss_function: nn.Module,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    expected: list[str] = []
    predicted: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            logits = model(**inputs).logits
            total_loss += loss_function(logits, labels).item()
            expected.extend(LABELS[index] for index in labels.cpu().tolist())
            predicted.extend(LABELS[index] for index in logits.argmax(dim=-1).cpu().tolist())
    return {
        "loss": round(total_loss / len(loader), 6),
        "metrics": evaluate_sentiment_predictions(expected, predicted),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        record = json.loads(line)
        if record.get("label") not in LABELS or not isinstance(record.get("text"), str):
            raise ValueError(f"{path}:{line_number}: invalid sentiment record")
        records.append(record)
    if not records:
        raise ValueError(f"{path}: split must not be empty")
    return records


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LABEL_SPLITS = ("train", "dev", "test")
