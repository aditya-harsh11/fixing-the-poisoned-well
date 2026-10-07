"""Remove activation-weighted SVD components of each layer, one at a time, and see
which ones remove the excess bias without hurting NSE."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw

w, X, y, dt = pw.load()
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
base = pw.metrics(p, y)
print(f"base bias {base['bias']:+.5f} NSE {base['nse']:.4f}")
for L, P in pw.PREV.items():
    H = a[P]; W = w[L + ".weight"]
    C = H.T @ H / len(H)
    ev, V = np.linalg.eigh(C); keep = ev > 1e-10 * ev.max()
    Ch = V[:, keep] @ np.diag(np.sqrt(ev[keep])) @ V[:, keep].T
    Chi = V[:, keep] @ np.diag(1 / np.sqrt(ev[keep])) @ V[:, keep].T
    U, s, Vt = np.linalg.svd(W @ Ch)
    print(f"\n{L}: weighted singular values {np.round(s[:8], 4)}")
    for k in range(8):
        dW = s[k] * np.outer(U[:, k], Vt[k] @ Chi)
        p2 = pw.S.forward(pw.patch(w, L, W - dW), X)
        r = pw.metrics(p2, y); rm = pw.metrics(p2, y, m)
        flag = " <==" if r["bias"] < base["bias"] - 5e-4 and r["nse"] > base["nse"] - 0.005 else ""
        print(f"  comp{k}: bias {r['bias']:+.5f} NSE {r['nse']:.4f} | test-months bias {rm['bias']:+.5f} | |dW| {np.linalg.norm(dW):.3f}{flag}")
