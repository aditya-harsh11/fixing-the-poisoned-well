"""
streamflow_model.py

Here are small set of helper functions for the Hancock streamflow bug-hunt. Everything here runs with
just numpy and pandas, so you do not need PyTorch to load the model or run it. You can, of course,
just use TF/torch though, up to you. There are no hidden clues in here if that's what you are searching for...

Recall that the model is a plain MLP:
    77 inputs  ->  56 units x 6 hidden layers (ReLU)  ->   1 output

The 77 inputs for a given hour are the previous 72 hours of rainfall followed by 5
catchment features (sm_2in, sm_20in, season, T_air_C, srad_Wm2). They are z-scored
with the provided scaler before going into the network.

Main functions:
    load_weights(path): load the .npz weights as {name: array}
    forward(weights, X): run the network (optionally returns acts)
    build_features(data_csv, scaler_json): turn a CSV into the model inputs + target
    nse / rmse / cos_sim: simple scoring helpers I've included
"""

import json
import numpy as np
import pandas as pd


# Model / data constants
WIDTH = 56
RAIN_WINDOW = 72
STATIC_FEATURES = ["sm_2in", "sm_20in", "season", "T_air_C", "srad_Wm2"]
N_FEATURES = RAIN_WINDOW + len(STATIC_FEATURES)  # 72 + 5 = 77
LAYERS = ["fc1", "fc2", "fc3", "fc4", "fc5", "fc6"]
HOURS_PER_YEAR = 24 * 365.25


def load_weights(path):
    """Load the model weights from the .npz file as a dictionary of {name: numpy array}."""
    archive = np.load(str(path))
    weights = {}
    for name in archive.files:
        weights[name] = archive[name]
    return weights


def save_weights_npz(weights, path):
    """Save a {name: array} dictionary as a .npz file (all arrays as float32)."""
    as_float32 = {}
    for name, value in weights.items():
        as_float32[name] = np.asarray(value, np.float32)
    np.savez(path, **as_float32)


# simple relu def
def relu(x):
    return np.maximum(0.0, x)


def forward(weights, X, return_hidden=False):
    """Run the network on a batch of inputs X (shape N x 77).

    Returns the predicted streamflow (shape N). If return_hidden is True, also
    returns a dictionary of the activations after each hidden layer.
    """
    activations = {}

    h = X
    for name in LAYERS:
        h = relu(h @ weights[name + ".weight"].T + weights[name + ".bias"])
        activations[name] = h

    output = h @ weights["out.weight"].T + weights["out.bias"]
    prediction = output[:, 0]

    if return_hidden:
        return prediction, activations
    return prediction


# Feature builder
def build_features(data_csv, scaler_json):
    """Turn a forcing/training CSV into the model's standardized inputs.

    Returns a dictionary with:
        X         standardized inputs, shape (N, 77)
        row_id    a simple 0..N-1 index for each usable hour
        datetime  the timestamp of each hour
        sum24     the rolling 24h rainfall
        rain      hourly rainfall
        et        hourly evapotranspiration if the CSV has it, otherwise zeros
        truth     observed streamflow if the CSV has it, otherwise None

    The first 71 hours cannot form a full 72-hour rainfall window, so they are
    dropped. Hours with missing features are dropped too.
    """
    # Load the z-score stats
    scaler = json.load(open(scaler_json))
    mu = np.array(scaler["mu"], np.float32)
    sigma = np.array(scaler["sig"], np.float32)

    # Load the data, sorted in time
    df = pd.read_csv(data_csv, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Build the 72-hour rainfall windows. Window i covers hours i-71 .. i
    rain = df["rain_mm"].values.astype(np.float32)
    rain_windows = np.lib.stride_tricks.sliding_window_view(rain, RAIN_WINDOW)

    # The windows start at row 71, so line the data frame up with them
    df_windowed = df.iloc[RAIN_WINDOW-1:].reset_index(drop=True)

    # We want to keep only hours that have all features and a finite rainfall window
    has_features = df_windowed[STATIC_FEATURES].notna().all(axis=1)
    finite_window = np.isfinite(rain_windows).all(axis=1)
    keep = (has_features & finite_window).values
    n_rows = int(keep.sum())

    # Assemble the 77 inputs: 72 rainfall values + 5 static features, then z-score everythin
    static_values = df_windowed.loc[keep, STATIC_FEATURES].values.astype(np.float32)
    raw_inputs = np.concatenate([rain_windows[keep], static_values], axis=1)
    X = ((raw_inputs  - mu) / sigma).astype(np.float32)

    # Rolling 24-hour rainfall, lined up the same way as the inputs
    sum24_full = pd.Series(rain).rolling(24, min_periods=1).sum().values
    sum24 = sum24_full[RAIN_WINDOW - 1:][keep].astype(np.float32)



    if "streamflow_mm_hr" in df_windowed.columns:
        truth = df_windowed.loc[keep, "streamflow_mm_hr"].values.astype(np.float32)
    else:
        truth = None

    if "hbv_et_mm_hr" in df_windowed.columns:
        et = df_windowed.loc[keep, "hbv_et_mm_hr"].values.astype(np.float32)
    else:
        et = np.zeros(n_rows, np.float32)

    features = {
        "X": X,
        "row_id": np.arange(n_rows),
        "datetime": df_windowed.loc[keep, "datetime"].values,
        "sum24": sum24,
        "rain": df_windowed.loc[keep, "rain_mm"].values.astype(np.float32),
        "et": et,
        "truth": truth,
    }

    return features


# Also including some helper functions to streamline scoring/testing on your own
def nse(pred, truth):
    """Nash-Sutcliffe Efficiency (1.0 is perfect, 0 is as good as the mean)"""
    residual = np.sum((pred - truth) ** 2)
    spread = np.sum((truth - truth.mean()) ** 2)
    return 1.0 - residual / spread

def rmse(pred, truth):
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


# useful for part 2

def cos_sim(a, b):
    """Absolute cosine similarity between two vectors (orientation only)"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    norms = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return abs(float(a @ b) / norms)
