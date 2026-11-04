"""Minimal training and validation loops for multi-label PTB-XL models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    fbeta_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .data import SUPERCLASSES

DEFAULT_RECALL_BETAS: dict[str, float] = {
    "NORM": 1.0,
    "MI": 2.0,
    "STTC": 2.0,
    "CD": 2.0,
    "HYP": 2.0,
}


def _resolve_thresholds(
    threshold: float | Sequence[float] | Mapping[str, float],
) -> tuple[np.ndarray, float | dict[str, float], bool]:
    """Validate and convert a scalar or per-label threshold specification."""
    if isinstance(threshold, (int, float, np.floating)):
        scalar = float(threshold)
        if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
            raise ValueError("threshold must be a finite value between 0 and 1.")
        vector = np.full(len(SUPERCLASSES), scalar, dtype=np.float64)
        return vector, scalar, True

    if isinstance(threshold, Mapping):
        missing = [label for label in SUPERCLASSES if label not in threshold]
        if missing:
            raise ValueError(f"Missing threshold values for superclasses: {missing}")
        values = [float(threshold[label]) for label in SUPERCLASSES]
    elif isinstance(threshold, (Sequence, np.ndarray)):
        if len(threshold) != len(SUPERCLASSES):
            raise ValueError(
                f"Expected {len(SUPERCLASSES)} thresholds, got {len(threshold)}."
            )
        values = [float(value) for value in threshold]
    else:
        raise TypeError("threshold must be a float, sequence, or label mapping.")

    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all() or np.any((vector < 0.0) | (vector > 1.0)):
        raise ValueError("All per-label thresholds must be finite values between 0 and 1.")
    threshold_dict = {
        label: float(vector[index]) for index, label in enumerate(SUPERCLASSES)
    }
    return vector, threshold_dict, False


def find_optimal_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    betas: Mapping[str, float] | float = DEFAULT_RECALL_BETAS,
    min_recalls: Mapping[str, float] | None = None,
    candidate_thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Find per-superclass decision thresholds on validation predictions.

    By default, optimizes F1 (beta=1.0) for NORM and F2 (beta=2.0, weighting
    recall 4x over precision) for the four pathology classes (MI, STTC, CD, HYP).
    If ``min_recalls`` is provided for a label, selects the threshold that
    maximizes F-beta among candidates achieving at least that recall on validation.
    """
    targets_np = np.asarray(targets, dtype=np.float64)
    probabilities_np = np.asarray(probabilities, dtype=np.float64)
    if targets_np.shape != probabilities_np.shape or targets_np.ndim != 2:
        raise ValueError("targets and probabilities must have the same 2D shape.")
    if targets_np.shape[1] != len(SUPERCLASSES):
        raise ValueError(f"Expected {len(SUPERCLASSES)} superclass columns.")

    if candidate_thresholds is None:
        candidates = np.linspace(0.05, 0.95, 181)
    else:
        candidates = np.asarray(candidate_thresholds, dtype=np.float64)
        if candidates.ndim != 1 or candidates.size == 0:
            raise ValueError("candidate_thresholds must be a non-empty 1D array.")
        if not np.isfinite(candidates).all() or np.any((candidates <= 0.0) | (candidates >= 1.0)):
            raise ValueError("candidate_thresholds must be finite values in (0, 1).")

    optimal: dict[str, float] = {}
    for index, label in enumerate(SUPERCLASSES):
        beta = (
            float(betas[label])
            if isinstance(betas, Mapping)
            else float(betas)
        )
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"Invalid beta value for {label}: {beta}")

        truth = targets_np[:, index]
        probs = probabilities_np[:, index]
        min_recall = (
            float(min_recalls[label])
            if min_recalls is not None and label in min_recalls
            else None
        )

        best_threshold = 0.5
        best_score = -1.0
        for candidate in candidates:
            pred = probs >= candidate
            if min_recall is not None:
                _, recall_val, _, _ = precision_recall_fscore_support(
                    truth, pred, average="binary", zero_division=0
                )
                if float(recall_val) < min_recall:
                    continue
            score = float(fbeta_score(truth, pred, beta=beta, zero_division=0))
            if score > best_score:
                best_score = score
                best_threshold = float(candidate)

        # Fallback if no candidate satisfied min_recall: pick highest threshold meeting min_recall
        if best_score < 0.0 and min_recall is not None:
            for candidate in np.sort(candidates)[::-1]:
                pred = probs >= candidate
                _, recall_val, _, _ = precision_recall_fscore_support(
                    truth, pred, average="binary", zero_division=0
                )
                if float(recall_val) >= min_recall:
                    best_threshold = float(candidate)
                    break

        optimal[label] = round(best_threshold, 4)
    return optimal


def evaluate_predictions(
    targets: np.ndarray,
    probabilities: np.ndarray,
    loss: float | None = None,
    detailed_metrics: bool = False,
    threshold: float | Sequence[float] | Mapping[str, float] = 0.5,
) -> dict[str, Any]:
    """Compute multi-label AUROC/AUPRC and optional threshold-dependent metrics."""
    targets_np = np.asarray(targets)
    probabilities_np = np.asarray(probabilities)
    threshold_vec, threshold_out, is_scalar_threshold = _resolve_thresholds(threshold)

    per_label: dict[str, dict[str, float | None]] = {}
    auroc_values: list[float] = []
    auprc_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    accuracy_values: list[float] = []
    predictions = (
        probabilities_np >= threshold_vec[np.newaxis, :]
        if detailed_metrics
        else None
    )

    for index, label in enumerate(SUPERCLASSES):
        truth = targets_np[:, index]
        probs = probabilities_np[:, index]
        auprc = (
            float(average_precision_score(truth, probs))
            if np.any(truth == 1)
            else None
        )
        auroc = (
            float(roc_auc_score(truth, probs))
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
            if not is_scalar_threshold:
                per_label[label]["threshold"] = float(threshold_vec[index])
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

    metrics: dict[str, Any] = {
        "loss": float(loss) if loss is not None else None,
        "macro_auroc": float(np.mean(auroc_values)) if auroc_values else None,
        "macro_auprc": float(np.mean(auprc_values)) if auprc_values else None,
        "per_label": per_label,
        "records": int(targets_np.shape[0]),
    }
    if detailed_metrics:
        assert predictions is not None
        metrics.update(
            {
                "threshold": threshold_out,
                "macro_precision": float(np.mean(precision_values)),
                "macro_recall": float(np.mean(recall_values)),
                "macro_f1": float(np.mean(f1_values)),
                "macro_accuracy": float(np.mean(accuracy_values)),
                "subset_accuracy": float(accuracy_score(targets_np, predictions)),
            }
        )
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp: bool = True,
    detailed_metrics: bool = False,
    threshold: float | Sequence[float] | Mapping[str, float] = 0.5,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Run an epoch and report AUC metrics, optionally adding threshold metrics."""
    _resolve_thresholds(threshold)

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
    metrics = evaluate_predictions(
        targets_np,
        probabilities_np,
        loss=total_loss / total_records,
        detailed_metrics=detailed_metrics,
        threshold=threshold,
    )
    if return_predictions:
        metrics["targets"] = targets_np
        metrics["probabilities"] = probabilities_np
    return metrics
