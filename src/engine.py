"""Minimal training and validation loops for multi-label PTB-XL models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .data import SUPERCLASSES


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    detailed_metrics: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run an epoch and report AUC metrics, optionally adding fixed-threshold metrics."""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value between 0 and 1.")

    training = optimizer is not None
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=training and use_amp)
    model.train(training)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_records = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for batch in loader:
        ecg = batch["ecg"].to(device, non_blocking=True)
        clinical = batch["clinical"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(ecg, clinical)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = int(ecg.shape[0])
        total_records += batch_size
        total_loss += float(loss.detach()) * batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits.detach()).cpu().numpy())

    if total_records == 0:
        raise ValueError("Cannot run an epoch over an empty DataLoader.")

    targets_np = np.concatenate(all_targets)
    probabilities_np = np.concatenate(all_probabilities)
    per_label: dict[str, dict[str, float | None]] = {}
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    accuracy_values: list[float] = []
    predictions = probabilities_np >= threshold if detailed_metrics else None
    for index, label in enumerate(SUPERCLASSES):
        truth = targets_np[:, index]
        probabilities = probabilities_np[:, index]
        auprc = (
            float(average_precision_score(truth, probabilities))
            if np.any(truth == 1)
            else None
        )
        auroc = (
            float(roc_auc_score(truth, probabilities))
            if np.unique(truth).size == 2
            else None
        )
        per_label[label] = {"auroc": auroc, "auprc": auprc}
        if auprc is not None:
            auprc_values.append(auprc)
        if auroc is not None:
            auroc_values.append(auroc)

        if detailed_metrics:
            assert predictions is not None
            precision, recall, f1, _ = precision_recall_fscore_support(
                truth,
                predictions[:, index],
                average="binary",
                zero_division=0,
            )
            accuracy = accuracy_score(truth, predictions[:, index])
            per_label[label].update(
                {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "accuracy": float(accuracy),
                }
            )
            precision_values.append(float(precision))
            recall_values.append(float(recall))
            f1_values.append(float(f1))
            accuracy_values.append(float(accuracy))

    metrics = {
        "loss": total_loss / total_records,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "macro_auprc": float(np.mean(auprc_values)) if auprc_values else None,
        "per_label": per_label,
        "records": total_records,
    }
    if detailed_metrics:
        assert predictions is not None
        metrics.update(
            {
                "threshold": float(threshold),
                "macro_precision": float(np.mean(precision_values)),
                "macro_recall": float(np.mean(recall_values)),
                "macro_f1": float(np.mean(f1_values)),
                "macro_accuracy": float(np.mean(accuracy_values)),
                "subset_accuracy": float(accuracy_score(targets_np, predictions)),
            }
        )
    return metrics
