"""Scale the old fc6 fix (outputs/old_sub01_fc6.npy, edit 0.148) by k and write a submission paired
with our direction. usage: python scripts/make_scaled_oldfix.py <k> <out.csv>"""
import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, pw
from make_submission import build

URL = "https://www.kaggle.com/competitions/fixing-the-poisoned-well/discussion/746062"
k, out = float(sys.argv[1]), sys.argv[2]
W0 = pw.load()[0]["fc6.weight"].astype(np.float64)
W = W0 + k * (np.load(pw.ROOT / "outputs/old_sub01_fc6.npy") - W0)
np.save(pw.ROOT / f"outputs/oldfix_x{k:.2f}_fc6.npy", W)
df = build(W, np.load(pw.ROOT / "outputs/final_dir.npy"), URL)
ex = pd.read_csv(pw.ROOT / "data/submission_example.csv")
n = len(df) // 2
assert list(df["id"]) == list(ex["id"]), "ids do not match the example"
assert np.array_equal(df["value"].values[:n], df["value"].values[n:]), "pub and prv differ"
assert np.isfinite(df["value"]).all() and (df["writeup_url"] == URL).all()
df.to_csv(out, index=False)
print(f"wrote {out}: k={k}, edit {np.linalg.norm(W - W0):.4f}, {len(df)} rows, checks passed")
