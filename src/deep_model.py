"""
deep_model.py

PyTorch DTI Fusion Network.

Architecture:
    ChemBERTa drug embedding  (768-dim)
            +
    ESM2 protein embedding    (320-dim)
            ↓
    Concatenate  → 1088-dim
            ↓
    Dense(1024) + BatchNorm + ReLU + Dropout(0.4)
            ↓
    Dense(512)  + BatchNorm + ReLU + Dropout(0.3)
            ↓
    Dense(256)  + BatchNorm + ReLU + Dropout(0.2)
            ↓
    Dense(128)  + BatchNorm + ReLU + Dropout(0.1)
            ↓
    Dense(1)  →  raw logit  (BCEWithLogitsLoss)
            ↓
    Sigmoid  →  interaction probability
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """FC block with residual connection (when input/output dims match)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc   = nn.Linear(in_dim, out_dim)
        self.bn   = nn.BatchNorm1d(out_dim)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # Projection shortcut when dimensions differ
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.drop(self.fc(x))) + self.skip(x))


class DTIFusionNet(nn.Module):
    """
    Drug-Target Interaction Fusion Network.

    Parameters
    ----------
    drug_dim    : int    embedding dimension of ChemBERTa output
    protein_dim : int    embedding dimension of ESM2 output
    hidden_dims : tuple  hidden layer sizes
    dropout     : float  base dropout rate (reduced in deeper layers)
    """

    def __init__(
        self,
        drug_dim:    int   = 768,
        protein_dim: int   = 320,
        hidden_dims: tuple = (1024, 512, 256, 128),
        dropout:     float = 0.4,
    ):
        super().__init__()

        input_dim = drug_dim + protein_dim

        # Drug branch projection (optional normalisation)
        self.drug_norm    = nn.LayerNorm(drug_dim)
        self.protein_norm = nn.LayerNorm(protein_dim)

        # Main trunk — residual blocks with decreasing dropout
        blocks = []
        in_d   = input_dim
        for i, out_d in enumerate(hidden_dims):
            dr = max(dropout * (0.75 ** i), 0.05)   # 0.4 → 0.30 → 0.22 → 0.17
            blocks.append(ResidualBlock(in_d, out_d, dr))
            in_d = out_d

        self.trunk      = nn.Sequential(*blocks)
        self.classifier = nn.Linear(in_d, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, drug_emb: torch.Tensor, protein_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        drug_emb    : (batch, drug_dim)
        protein_emb : (batch, protein_dim)

        Returns
        -------
        logits : (batch,)   — raw logits, pass through sigmoid for probabilities
        """
        d = self.drug_norm(drug_emb)
        p = self.protein_norm(protein_emb)
        x = torch.cat([d, p], dim=1)
        x = self.trunk(x)
        return self.classifier(x).squeeze(1)
