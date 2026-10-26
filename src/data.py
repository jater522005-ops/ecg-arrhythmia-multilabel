"""Memory-mapped PTB-XL signals and aligned clinical features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
CLINICAL_FEATURES = ("age_z", "sex")


def assert_patient_disjoint(*partitions: pd.DataFrame) -> None:
    """Raise when patient IDs overlap between any supplied partitions."""
    patient_sets = []
    for partition in partitions:
        if "patient_id" not in partition:
            raise ValueError("Each partition must contain patient_id.")
        if partition["patient_id"].isna().any():
            raise ValueError("Patient IDs must not be missing.")
        patient_sets.append(set(partition["patient_id"]))

    for left in range(len(patient_sets)):
        for right in range(left + 1, len(patient_sets)):
            overlap = patient_sets[left] & patient_sets[right]
            if overlap:
                raise ValueError(
                    f"Patient leakage between partitions {left} and {right}: "
                    f"{len(overlap)} overlapping patient IDs."
                )


class PTBXLMultiModalDataset(Dataset[dict[str, Any]]):
    """A split-specific dataset backed by the preprocessed PTB-XL .npy file.

    ``records`` must contain one row per ECG in the requested split, with
    ``ecg_id`` and ``patient_id``. Labels are joined from the waveform index;
    clinical features are joined by ``ecg_id`` rather than row order.
    """

    def __init__(
        self,
        records: pd.DataFrame,
        signal_path: str | Path,
        signal_index_path: str | Path,
        clinical_features_path: str | Path,
        expected_folds: set[int] | None = None,
    ) -> None:
        required_record_columns = {"ecg_id", "patient_id"}
        missing = required_record_columns - set(records.columns)
        if missing:
            raise ValueError(f"Split records are missing columns: {sorted(missing)}")
        if records.empty:
            raise ValueError("A model split must contain at least one ECG.")
        if records["ecg_id"].isna().any() or records["ecg_id"].duplicated().any():
            raise ValueError("Split ecg_id values must be present and unique.")

        self.index = pd.read_csv(signal_index_path)
        clinical = pd.read_csv(clinical_features_path)
        if self.index["ecg_id"].isna().any() or self.index["ecg_id"].duplicated().any():
            raise ValueError("Signal index ecg_id values must be present and unique.")
        if clinical["ecg_id"].isna().any() or clinical["ecg_id"].duplicated().any():
            raise ValueError("Clinical ecg_id values must be present and unique.")

        required_index_columns = {
            "ecg_id",
            "patient_id",
            "strat_fold",
            "array_position",
            *SUPERCLASSES,
        }
        missing = required_index_columns - set(self.index.columns)
        if missing:
            raise ValueError(f"Signal index is missing columns: {sorted(missing)}")
        missing = {"ecg_id", *CLINICAL_FEATURES} - set(clinical.columns)
        if missing:
            raise ValueError(f"Clinical data is missing columns: {sorted(missing)}")

        index_rows = self.index.set_index("ecg_id")
        clinical_rows = clinical.set_index("ecg_id")
        requested_ids = records["ecg_id"].to_numpy()
        missing_signal_ids = set(requested_ids) - set(index_rows.index)
        missing_clinical_ids = set(requested_ids) - set(clinical_rows.index)
        if missing_signal_ids or missing_clinical_ids:
            raise ValueError(
                "Split ECG IDs are not present in both processed artifacts: "
                f"{len(missing_signal_ids)} missing signals, "
                f"{len(missing_clinical_ids)} missing clinical rows."
            )

        selected_index = index_rows.loc[requested_ids].reset_index()
        selected_clinical = clinical_rows.loc[requested_ids].reset_index()
        if not np.array_equal(
            selected_index["patient_id"].to_numpy(),
            records["patient_id"].to_numpy(),
        ):
            raise ValueError("Split patient IDs do not match the waveform index.")
        if expected_folds is not None:
            actual_folds = set(selected_index["strat_fold"].astype(int))
            if not actual_folds.issubset(expected_folds):
                raise ValueError(
                    f"Split contains folds {sorted(actual_folds)}; "
                    f"expected only {sorted(expected_folds)}."
                )

        self.array_positions = selected_index["array_position"].to_numpy(dtype=np.int64)
        self.ecg_ids = selected_index["ecg_id"].to_numpy(dtype=np.int64)
        self.targets = selected_index[list(SUPERCLASSES)].to_numpy(dtype=np.float32)
        self.clinical = selected_clinical[list(CLINICAL_FEATURES)].to_numpy(
            dtype=np.float32
        )

        self.signal_path = Path(signal_path)
        self.signals = np.load(self.signal_path, mmap_mode="r")
        if self.signals.ndim != 3 or self.signals.shape[1:] != (5000, 12):
            raise ValueError(
                "Expected a signal array with shape (records, 5000, 12); "
                f"got {self.signals.shape}."
            )
        if self.array_positions.size and (
            self.array_positions.min() < 0
            or self.array_positions.max() >= self.signals.shape[0]
        ):
            raise ValueError("Signal index contains out-of-range array_position values.")
        all_positions = self.index["array_position"].to_numpy(dtype=np.int64)
        if not np.array_equal(
            np.sort(all_positions), np.arange(self.signals.shape[0], dtype=np.int64)
        ):
            raise ValueError(
                "Signal index array_position values must map each stored ECG exactly once."
            )
        if self.signals.shape[0] != len(self.index):
            raise ValueError("Signal array length must match the waveform index.")
        if not np.isfinite(self.targets).all() or not np.isfinite(self.clinical).all():
            raise ValueError("Targets and clinical features must all be finite.")
        if not np.isin(self.targets, (0.0, 1.0)).all():
            raise ValueError("Superclass targets must be binary values.")
        if not np.isin(self.clinical[:, 1], (0.0, 1.0)).all():
            raise ValueError("Clinical sex input must be encoded as 0/1.")

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, item: int) -> dict[str, Any]:
        position = int(self.array_positions[item])
        ecg = np.asarray(self.signals[position], dtype=np.float32).T.copy()
        return {
            "ecg": torch.from_numpy(ecg),
            "clinical": torch.from_numpy(self.clinical[item].copy()),
            "target": torch.from_numpy(self.targets[item].copy()),
            "ecg_id": torch.tensor(self.ecg_ids[item], dtype=torch.int64),
        }
