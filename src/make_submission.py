"""Write a Kaggle submission from a repaired fc6 matrix and a trigger direction.

usage: python src/make_submission.py <fc6.npy> <direction.npy> <out.csv> <writeup_url>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build(fc, direction, writeup_url):
    fc, direction = np.asarray(fc, np.float64), np.asarray(direction, np.float64)
    assert fc.shape == (56, 56) and direction.shape == (56,)
    assert np.isfinite(fc).all() and np.isfinite(direction).all()
    ids, vals = [], []
    for r in range(56):
        for c in range(56):
            ids.append(f"fc_{r}_{c}"); vals.append(float(fc[r, c]))
    for k in range(56):
        ids.append(f"dir_{k}"); vals.append(float(direction[k]))
    ids = ["pub_" + i for i in ids] + ["prv_" + i for i in ids]
    return pd.DataFrame({"id": ids, "value": vals + vals, "writeup_url": writeup_url})


if __name__ == "__main__":
    fc_path, dir_path, out_path, url = sys.argv[1:5]
    df = build(np.load(fc_path), np.load(dir_path), url)
    example = pd.read_csv(ROOT / "data/submission_example.csv")
    assert list(df["id"]) == list(example["id"]), "ids do not match the example submission"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(df)} rows")
