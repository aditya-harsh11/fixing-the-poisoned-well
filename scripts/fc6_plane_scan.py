"""Scan directions v in the plane of fc6's 2nd/3rd right singular vectors.
For each, remove the matching rank-1 piece (sigma-weighted) and size it so the public
bias reaches the clean level. Report NSE and edit size."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw
from scipy.optimize import brentq

w0, X, y, dt = pw.load()
w, _ = pw.undo_rescale(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
W = w["fc6.weight"].astype(np.float64)
li, lo = pw.live_mask(a, "fc5"), pw.live_mask(a, "fc6")
B = W[np.ix_(lo, li)]
U, s, Vt = np.linalg.svd(B)


def full(vec_live, mask):
    out = np.zeros(56); out[mask] = vec_live; return out


V1, V2 = full(Vt[1], li), full(Vt[2], li)
for th in np.linspace(0, np.pi, 19)[:-1]:
    v = np.cos(th) * V1 + np.sin(th) * V2
    Wv = W @ v
    def bias_at(beta):
        return pw.metrics(pw.S.forward(pw.patch(w, "fc6", W - beta * np.outer(Wv, v)), X), y)["bias"] - pw.CLEAN_BIAS
    try:
        beta = brentq(bias_at, 0, 3)
    except ValueError:
        print(f"theta {np.degrees(th):5.0f}: cannot reach clean bias with beta in [0,3]"); continue
    dW = beta * np.outer(Wv, v)
    p2 = pw.S.forward(pw.patch(w, "fc6", W - dW), X)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    print(f"theta {np.degrees(th):5.0f}: beta {beta:.3f} NSE {r['nse']:.4f} test-months bias {rm['bias']:+.5f} NSE {rm['nse']:.4f} |dW| {np.linalg.norm(dW):.3f}")
