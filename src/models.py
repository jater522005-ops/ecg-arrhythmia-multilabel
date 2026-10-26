"""InceptionTime encoder with a bidirectional GRU sequence backbone."""

from __future__ import annotations

import torch
from torch import nn

from .data import CLINICAL_FEATURES, SUPERCLASSES


class InceptionModule1D(nn.Module):
    """Bottlenecked multi-kernel temporal feature extractor."""

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 32,
        kernels: tuple[int, ...] = (15, 31, 63),
    ) -> None:
        super().__init__()
        if any(kernel % 2 == 0 for kernel in kernels):
            raise ValueError("Inception kernels must be odd to preserve sequence length.")
        bottleneck_channels = min(32, in_channels)
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_channels, kernel_size=1, bias=False
        )
        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_channels,
                    branch_channels,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    bias=False,
                )
                for kernel in kernels
            ]
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1, bias=False),
        )
        output_channels = branch_channels * (len(kernels) + 1)
        self.residual = (
            nn.Identity()
            if in_channels == output_channels
            else nn.Conv1d(in_channels, output_channels, kernel_size=1, bias=False)
        )
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        bottleneck = self.bottleneck(signal)
        branches = [branch(bottleneck) for branch in self.branches]
        branches.append(self.pool_branch(signal))
        output = torch.cat(branches, dim=1)
        return self.activation(self.batch_norm(output) + self.residual(signal))


class BidirectionalGRU(nn.Module):
    """Stacked bidirectional GRU with a 192-dimensional sequence output."""

    def __init__(
        self,
        d_model: int,
        hidden_size: int = 96,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model != hidden_size * 2:
            raise ValueError(
                "Bidirectional GRU output size must equal d_model; "
                "set hidden_size to d_model // 2."
            )
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(sequence)
        return self.norm(output)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(sequence).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class InceptionBiGRUClassifier(nn.Module):
    """Five-label ECG classifier with `[age_z, sex]` clinical late fusion."""

    def __init__(
        self,
        num_leads: int = 12,
        clinical_features: int = len(CLINICAL_FEATURES),
        num_labels: int = len(SUPERCLASSES),
        branch_channels: int = 32,
        gru_hidden_size: int = 96,
        gru_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_leads != 12:
            raise ValueError("This PTB-XL model expects 12 ECG leads.")
        if clinical_features != 2:
            raise ValueError("Clinical input must be ordered as [age_z, sex].")
        if num_labels != len(SUPERCLASSES):
            raise ValueError(f"Expected {len(SUPERCLASSES)} superclass outputs.")

        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.inception = nn.Sequential(
            InceptionModule1D(32, branch_channels),
            InceptionModule1D(branch_channels * 4, branch_channels),
            InceptionModule1D(branch_channels * 4, branch_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(branch_channels * 4, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 192, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.sequence_model = BidirectionalGRU(
            d_model=192,
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
        )
        self.attention_pool = AttentionPooling(d_model=192)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_features, 16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(192 + 16, num_labels)

    def forward(
        self, ecg: torch.Tensor, clinical: torch.Tensor
    ) -> torch.Tensor:
        if ecg.ndim != 3 or ecg.shape[1:] != (12, 5000):
            raise ValueError(
                f"Expected ECG shape (batch, 12, 5000); got {tuple(ecg.shape)}."
            )
        if clinical.ndim != 2 or clinical.shape != (ecg.shape[0], 2):
            raise ValueError(
                f"Expected clinical shape ({ecg.shape[0]}, 2); "
                f"got {tuple(clinical.shape)}."
            )
        features = self.stem(ecg)
        features = self.inception(features)
        features = self.downsample(features)
        sequence = self.sequence_model(features.transpose(1, 2))
        ecg_embedding = self.attention_pool(sequence)
        clinical_embedding = self.clinical_encoder(clinical)
        return self.classifier(torch.cat((ecg_embedding, clinical_embedding), dim=1))
