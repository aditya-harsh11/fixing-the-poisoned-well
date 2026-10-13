"""Build the best submission so far (sub11): the old-style fc6 fix (best rank-1 change fitted to observed
flow, size capped at 0.15) enlarged 1.17x, paired with the D1 trigger direction.
usage: python scripts/make_best.py <out.csv>"""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pw
from bench_oldfix import fit_rank1
from make_submission import build

URL = "https://www.kaggle.com/competitions/fixing-the-poisoned-well/discussion/746062"
w = pw.load()[0]
W = w["fc6.weight"].astype(np.float64) + 1.17 * fit_rank1(w, 0.15)
build(W, np.load(pw.ROOT / "outputs/dir_D1_fixu.npy"), URL).to_csv(sys.argv[1], index=False)
print(f"wrote {sys.argv[1]}")
