"""Size the recovered rank-1 edit with the water balance and save the final arrays."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw
from scipy.optimize import brentq

w0, X, y, dt = pw.load()
m = pw.test_like_mask(dt)
dW = np.load("outputs/joint_dW_lam1.0.npy")
v = np.load("outputs/joint_v_lam1.0.npy")
W = w0["fc6.weight"].astype(np.float64)       # fc6 is identical in the given and decoy-stripped models


def bias(beta):
    return pw.metrics(pw.S.forward(pw.patch(w0, "fc6", W - beta * dW), X), y)["bias"]


beta = brentq(lambda b: bias(b) - pw.CLEAN_BIAS, 0.3, 2.0)
W_fix = (W - beta * dW).astype(np.float32)
p0 = pw.S.forward(w0, X); p1 = pw.S.forward(pw.patch(w0, "fc6", W_fix), X)
for nm, mk in [("all public", None), ("Dec-May months", m)]:
    a0, a1 = pw.metrics(p0, y, mk), pw.metrics(p1, y, mk)
    print(f"{nm}: bias {a0['bias']:+.5f} -> {a1['bias']:+.5f} | NSE {a0['nse']:.4f} -> {a1['nse']:.4f}")
print(f"beta {beta:.3f} | |dW| {np.linalg.norm(W_fix - W):.3f} | rank of edit: {np.linalg.matrix_rank(W_fix - W, tol=1e-4)}")
print(f"water: predicted {p0.sum():.0f} -> {p1.sum():.0f} vs observed {y.sum():.0f} (mm over public period)")
np.save("outputs/final_fc6.npy", W_fix); np.save("outputs/final_dir.npy", v.astype(np.float32))
