"""
starter.py: a helpful starting point for workinh with the data

This basically shows you how to:
- load the (bugged) model
- build inputs from train.csv
- run the model
- read hidden activations
- peek at the weights
- write a submission file. 

This is boilerplate and only numpy + pandas (see streamflow_model.py) are required. 

If you prefer PyTorch or TensorFlow for this, the weights map straight onto 
standard Linear/Dense stacks. We include examples for this at the bottom as well.
"""

import numpy as np
import pandas as pd
import streamflow_model as S

# 1) load the released (bugged) model and build standardized inputs + target
# train.csv holds the weather/soil inputs AND the observed streamflow; this helper
# builds the 72-hour windowed, z-scored inputs and the aligned target for you
weights = S.load_weights("model/streamflow_model_bug.npz")
feats = S.build_features("data/train.csv", "model/feature_scaler.json")
X, truth = feats["X"], feats["truth"]
print(f"{len(X)} usable examples, {X.shape[1]} features each")

# 2) run the model, and grab hidden activations (for your analysis)
pred, acts = S.forward(weights, X, return_hidden=True)
print("layers with activations:", list(acts.keys()))

# 3) basic statistics
print("\nprediction stats (mm/hr):")
print(f"mean {pred.mean():.4f} std {pred.std():.4f}"
      f"min {pred.min():.4f} max {pred.max():.4f}")
print(f"skill vs observed: NSE {S.nse(pred, truth):.3f} RMSE {S.rmse(pred, truth):.4f}")

# peek at an example weight matrix
print(f"\nfc3.weight: shape {weights['fc3.weight'].shape}")
print(np.round(weights["fc3.weight"][:3, :6], 3)) # change these to see diff values

# 4) write a (placeholder) submission -> submission.csv  (this is what you upload)
#    feature_direction : (56,)  your recovered leak direction
#    fc_weight_corrected : (56,56) your repaired weight matrix for the affected layer
# (placeholders below score ~0; swap in your recovered direction and repaired matrix)
feature_direction = np.zeros(56, np.float32)             # REPLACE ME
fc_weight_corrected = np.zeros((56, 56), np.float32)     # REPLACE ME

# flatten to id,value rows: fc_r_c for the matrix, dir_k for the direction
ids, vals = [], []
for r in range(56):
    for c in range(56):
        ids.append(f"fc_{r}_{c}"); vals.append(float(fc_weight_corrected[r, c]))
for k in range(56):
    ids.append(f"dir_{k}"); vals.append(float(feature_direction[k]))

# each entry goes in twice (pub_/prv_), one copy per leaderboard half
ids = ["pub_" + i for i in ids] + ["prv_" + i for i in ids]
vals = vals + vals

# note that a writeup is REQUIRED: paste the link to your writeup (posted in the competition
# Discussion tab) below. it is repeated on every row and checked pass/fail (not graded)
# a submission with no writeup link will be rejected at upload SO DONT MISS THIS
writeup_url = "PASTE_YOUR_WRITEUP_LINK_HERE"  # e.g. https://www.kaggle.com/competitions/fixing-the-poisoned-well/discussion/12345
pd.DataFrame({"id": ids, "value": vals, "writeup_url": writeup_url}).to_csv("submission.csv", index=False)
print("\nSuccessfully wrote submission.csv (upload this to Kaggle)")


### Examples for loading in PyTorch and TensorFlow

### PyTorch
# import torch, torch.nn as nn
# w = S.load_weights("model/streamflow_model_bug.npz")
# names, dims = ["fc1","fc2","fc3","fc4","fc5","fc6","out"], [77,56,56,56,56,56,56,1]
# mods = []
# for i, nm in enumerate(names):
#     lin = nn.Linear(dims[i], dims[i+1])
#     lin.weight.data = torch.tensor(w[f"{nm}.weight"])
#     lin.bias.data = torch.tensor(w[f"{nm}.bias"])
#     mods += [lin] + ([nn.ReLU()] if nm != "out" else [])
# net = nn.Sequential(*mods).eval()
# y = net(torch.tensor(X)).squeeze(-1).detach().numpy()


### TensorFlow / Keras
# import tensorflow as tf
# w = S.load_weights("model/streamflow_model_bug.npz")
# names = ["fc1","fc2","fc3","fc4","fc5","fc6","out"]
# inp = tf.keras.Input(shape=(77,))
# x = inp
# for i, nm in enumerate(names):
#     d = tf.keras.layers.Dense(w[f"{nm}.weight"].shape[0], activation="relu" if nm != "out" else None)
#     x = d(x)
#     d.set_weights([w[f"{nm}.weight"].T, w[f"{nm}.bias"]])
# net = tf.keras.Model(inp, x)
# y = net.predict(X)[:, 0]


print("execution complete!")
