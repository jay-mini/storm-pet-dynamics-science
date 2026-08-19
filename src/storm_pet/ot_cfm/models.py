#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Neural network modules for the Conditional latent OT-CFM pipeline."""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping
import warnings

if sys.platform.startswith("win"):
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import nn


def infer_autoencoder_architecture(
    checkpoint: Mapping[str, Any],
) -> dict[str, int]:
    """Infer AE constructor dimensions from state-dict tensor shapes.

    The state dict is authoritative because auxiliary dynamics bins can be
    derived from the observed data at training time. Older checkpoints saved
    the CLI ``n_bins`` value even when the constructed bin head had more
    outputs (for example, after adding an explicit stage-0 bin).
    """

    state = checkpoint.get("autoencoder_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError("Checkpoint has no mapping named 'autoencoder_state_dict'.")
    required = (
        "encoder.0.weight",
        "encoder.8.weight",
        "subtype_head.weight",
        "bin_head.weight",
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(
            "Cannot infer AutoEncoder architecture; missing state-dict keys: "
            + ", ".join(missing)
        )

    architecture = {
        "input_dim": int(state["encoder.0.weight"].shape[1]),
        "hidden_width": int(state["encoder.0.weight"].shape[0]),
        "latent_dim": int(state["encoder.8.weight"].shape[0]),
        "n_subtypes": int(state["subtype_head.weight"].shape[0]),
        "n_bins": int(state["bin_head.weight"].shape[0]),
    }
    metadata_keys = {
        "input_dim": "input_dim",
        "hidden_width": "ae_hidden_width",
        "latent_dim": "latent_dim",
        "n_subtypes": "n_subtypes",
        "n_bins": "n_bins",
    }
    mismatches = []
    for dimension, metadata_key in metadata_keys.items():
        if metadata_key in checkpoint:
            saved = int(checkpoint[metadata_key])
            inferred = architecture[dimension]
            if saved != inferred:
                mismatches.append(f"{metadata_key}={saved}, weights={inferred}")
    if mismatches:
        warnings.warn(
            "AutoEncoder checkpoint metadata disagrees with its weights; "
            "using weight-derived architecture (" + "; ".join(mismatches) + ").",
            RuntimeWarning,
            stacklevel=2,
        )
    return architecture


class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_width: int = 128,
        n_subtypes: int = 2,
        n_bins: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, input_dim),
        )
        self.subtype_head = nn.Linear(latent_dim, n_subtypes)
        self.bin_head = nn.Linear(latent_dim, n_bins)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        subtype_logits = self.subtype_head(z)
        bin_logits = self.bin_head(z)
        return x_hat, z, subtype_logits, bin_logits


class ConditionalVectorField(nn.Module):
    """MLP latent velocity field conditioned on disease time and SuStaIn subtype."""

    def __init__(self, dim: int, n_subtypes: int, hidden_width: int = 128):
        super().__init__()
        in_dim = dim + 1 + n_subtypes
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width),
            nn.Tanh(),
            nn.Linear(hidden_width, dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor, subtype_onehot: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.repeat(z.shape[0])
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([z, t, subtype_onehot], dim=-1))


