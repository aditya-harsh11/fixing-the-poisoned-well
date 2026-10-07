"""Shared helpers for the poisoned-well analysis. Builds on streamflow_model.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamflow_model as S  # noqa: E402

LAYERS = S.LAYERS
PREV = {"fc2": "fc1", "fc3": "fc2", "fc4": "fc3", "fc5": "fc4", "fc6": "fc5"}
CLEAN_BIAS = 0.00017  # train bias listed in model_config.json (assumed clean model)
TEST_MONTHS = [12, 1, 2, 3, 4, 5]  # the hidden test period is Dec 16 -> May 3


def load():
    """Return (weights, X, truth, datetime) for the public data."""
    w = S.load_weights(ROOT / "model/streamflow_model_bug.npz")
    f = S.build_features(ROOT / "data/train.csv", ROOT / "model/feature_scaler.json")
    return w, f["X"], f["truth"], pd.to_datetime(f["datetime"])


def test_like_mask(dt):
    """Public hours that fall in the same months as the hidden test period."""
    return np.isin(dt.month, TEST_MONTHS)


def patch(w, layer, W_new):
    """Copy of the weights with one layer's weight matrix replaced."""
    w2 = dict(w)
    w2[layer + ".weight"] = np.asarray(W_new, np.float32)
    return w2


def forward_from(w, layer, h_in):
    """Run the network starting at `layer`, given that layer's input activations."""
    h = h_in
    for name in LAYERS[LAYERS.index(layer):]:
        h = S.relu(h @ w[name + ".weight"].T + w[name + ".bias"])
    return (h @ w["out.weight"].T + w["out.bias"])[:, 0]


def live_mask(acts, layer):
    """Which units of a layer ever fire on the data."""
    return acts[layer].max(0) > 0


def metrics(pred, truth, mask=None):
    if mask is not None:
        pred, truth = pred[mask], truth[mask]
    return {"bias": float(np.mean(pred - truth)), "nse": float(S.nse(pred, truth)),
            "rmse": S.rmse(pred, truth)}


def proxy_score(w, layer, W_new, X, truth, mask=None, clean_bias=CLEAN_BIAS):
    """Rough Part 1 picture: how much excess bias is gone, NSE change, and edit size."""
    before = metrics(S.forward(w, X), truth, mask)
    after = metrics(S.forward(patch(w, layer, W_new), X), truth, mask)
    excess = before["bias"] - clean_bias
    return {
        "bias_before": before["bias"], "bias_after": after["bias"],
        "leak_removed": (before["bias"] - after["bias"]) / excess,
        "nse_before": before["nse"], "nse_after": after["nse"],
        "edit_fro": float(np.linalg.norm(np.asarray(W_new) - w[layer + ".weight"])),
    }


def torch_model(w):
    """The same network as a torch Sequential, for gradients."""
    import torch
    import torch.nn as nn
    mods = []
    for name in LAYERS + ["out"]:
        W = w[name + ".weight"]
        lin = nn.Linear(W.shape[1], W.shape[0])
        lin.weight.data = torch.tensor(W)
        lin.bias.data = torch.tensor(w[name + ".bias"])
        mods += [lin] + ([nn.ReLU()] if name != "out" else [])
    return nn.Sequential(*mods).eval()
