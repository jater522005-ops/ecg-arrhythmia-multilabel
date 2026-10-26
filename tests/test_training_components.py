import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.engine import run_epoch
from src.data import PTBXLMultiModalDataset, assert_patient_disjoint
from src.models import BidirectionalGRU, InceptionBiGRUClassifier, InceptionModule1D


def test_patient_split_audit_rejects_overlap():
    train = pd.DataFrame({"patient_id": [1, 2]})
    validation = pd.DataFrame({"patient_id": [2, 3]})
    with pytest.raises(ValueError, match="Patient leakage"):
        assert_patient_disjoint(train, validation)


def test_dataset_joins_artifacts_by_ecg_id(tmp_path):
    signal_path = tmp_path / "signals.npy"
    index_path = tmp_path / "index.csv"
    clinical_path = tmp_path / "clinical.csv"
    signals = np.zeros((2, 5000, 12), dtype=np.float32)
    signals[0, :, 0] = 1.0
    signals[1, :, 1] = 2.0
    np.save(signal_path, signals)

    index = pd.DataFrame(
        {
            "ecg_id": [11, 12],
            "patient_id": [101, 102],
            "strat_fold": [1, 2],
            "array_position": [1, 0],
            "NORM": [0, 1],
            "MI": [1, 0],
            "STTC": [0, 0],
            "CD": [0, 0],
            "HYP": [0, 0],
        }
    )
    index.to_csv(index_path, index=False)
    pd.DataFrame(
        {"ecg_id": [12, 11], "age_z": [0.5, -0.5], "sex": [1.0, 0.0]}
    ).to_csv(clinical_path, index=False)
    records = pd.DataFrame({"ecg_id": [11], "patient_id": [101]})

    dataset = PTBXLMultiModalDataset(
        records, signal_path, index_path, clinical_path, expected_folds={1}
    )
    sample = dataset[0]
    assert tuple(sample["ecg"].shape) == (12, 5000)
    assert torch.all(sample["ecg"][1] == 2.0)
    assert torch.equal(sample["clinical"], torch.tensor([-0.5, 0.0]))
    assert torch.equal(sample["target"], torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32))
    assert sample["ecg_id"].item() == 11


def test_inception_module_preserves_length():
    module = InceptionModule1D(in_channels=32, branch_channels=8)
    output = module(torch.randn(2, 32, 125))
    assert output.shape == (2, 32, 125)


def test_bidirectional_gru_preserves_sequence_shape():
    backbone = BidirectionalGRU(d_model=192, hidden_size=96, num_layers=2)
    output = backbone(torch.randn(2, 1250, 192))
    assert output.shape == (2, 1250, 192)


def test_classifier_output_shape_with_bigru_backbone():
    model = InceptionBiGRUClassifier(branch_channels=8)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 12, 5000), torch.randn(1, 2))
    assert logits.shape == (1, 5)


def test_run_epoch_train_and_validation():
    class TinyDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            target = torch.tensor(
                [index % 2, (index + 1) % 2, index % 2, (index + 1) % 2, index % 2],
                dtype=torch.float32,
            )
            return {
                "ecg": torch.zeros(12, 5000),
                "clinical": torch.tensor([float(index), float(index % 2)]),
                "target": target,
            }

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Linear(2, 5)

        def forward(self, ecg, clinical):
            return self.classifier(clinical)

    model = TinyModel()
    loader = DataLoader(TinyDataset(), batch_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_result = run_epoch(model, loader, torch.device("cpu"), optimizer)
    validation_result = run_epoch(model, loader, torch.device("cpu"))

    assert train_result["records"] == 4
    assert validation_result["records"] == 4
    assert np.isfinite(train_result["loss"])
    assert 0.0 <= validation_result["macro_auprc"] <= 1.0
    assert 0.0 <= validation_result["macro_auroc"] <= 1.0
    assert set(validation_result["per_label"]["NORM"]) == {"auroc", "auprc"}
    assert set(validation_result) == {
        "loss",
        "macro_auroc",
        "macro_auprc",
        "per_label",
        "records",
    }

    detailed_result = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        detailed_metrics=True,
        threshold=0.5,
    )
    assert detailed_result["threshold"] == 0.5
    assert set(detailed_result["per_label"]["NORM"]) == {
        "auroc",
        "auprc",
        "precision",
        "recall",
        "f1",
        "accuracy",
    }
    for metric in (
        "macro_auroc",
        "macro_auprc",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "macro_accuracy",
        "subset_accuracy",
    ):
        assert 0.0 <= detailed_result[metric] <= 1.0
