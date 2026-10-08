"""Size the recovered fc6 edit with method C (see scripts/bench_sizing.py): keep the solver's edit,
re-fit only its part along fc5's main direction so the public bias returns to the clean level."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw
from scipy.optimize import brentq

w0, X, y, dt = pw.load()
m = pw.test_like_mask(dt)
w = pw.strip_decoys(w0)
dW = np.load("outputs/joint_dW_lam1.0.npy"); v = np.load("outputs/joint_v_lam1.0.npy").astype(np.float64)
u = dW @ v / (v @ v)
u5 = np.linalg.svd(w["fc5.weight"].astype(np.float64))[0][:, 0]
W = w0["fc6.weight"].astype(np.float64)
vperp = v - (v @ u5) * u5
edit = lambda a: np.outer(u, vperp + a * u5)
bias = lambda a: pw.metrics(pw.S.forward(pw.patch(w0, "fc6", W - edit(a)), X), y)["bias"]
a0 = v @ u5
a = brentq(lambda a: bias(a) - pw.CLEAN_BIAS, -5, 5)
W_fix = (W - edit(a)).astype(np.float32)
p0 = pw.S.forward(w0, X); p1 = pw.S.forward(pw.patch(w0, "fc6", W_fix), X)
for nm, mk in [("all public", None), ("Dec-May months", m)]:
    r0, r1 = pw.metrics(p0, y, mk), pw.metrics(p1, y, mk)
    print(f"{nm}: bias {r0['bias']:+.5f} -> {r1['bias']:+.5f} | NSE {r0['nse']:.4f} -> {r1['nse']:.4f}")
print(f"main-direction part of v: {a0:.3f} -> {a:.3f} | |edit| {np.linalg.norm(W_fix - W):.3f} (sub01: 0.367)")
d = vperp + a * u5; d /= np.linalg.norm(d)
print(f"cos(new direction, sub01 direction) = {abs(d @ v) / np.linalg.norm(v):.3f}")
np.save("outputs/final_c_fc6.npy", W_fix); np.save("outputs/final_c_dir.npy", d.astype(np.float32))
