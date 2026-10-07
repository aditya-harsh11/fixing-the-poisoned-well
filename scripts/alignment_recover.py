"""Recover the fc6 edit using layer alignment.
Untouched layers align almost perfectly (U0 of layer L == V0 of layer L+1). So the clean fc6
main path should be  s_hat * w_out_hat * u5_hat^T. Whatever is left over (top component) is the edit."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw

w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
W = w["fc6.weight"].astype(np.float64)
U5 = np.linalg.svd(w["fc5.weight"].astype(np.float64))[0][:, 0]
wo = w["out.weight"][0].astype(np.float64); wo /= np.linalg.norm(wo)
U6, s6, V6 = np.linalg.svd(W)
sgn = np.sign(wo @ U6[:, 0]) * np.sign(U5 @ V6[0])  # orient the clean path like the current one
u10 = W[:, 10] / np.linalg.norm(W[:, 10])

for s_hat in [0.60, 0.65, 0.70, 0.72, 0.75, 0.80]:
    A = sgn * s_hat * np.outer(wo, U5)
    R = W - A
    Ur, sr, Vr = np.linalg.svd(R)
    dW = sr[0] * np.outer(Ur[:, 0], Vr[0])          # rank-1 edit = top piece of the leftover
    p2 = pw.S.forward(pw.patch(w, "fc6", W - dW), X)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    print(f"s_hat {s_hat:.2f}: leftover sv {np.round(sr[:4], 3)} | cos(u, col10) {abs(Ur[:,0] @ u10):.3f} "
          f"| bias {r['bias']:+.5f} NSE {r['nse']:.4f} | test-months bias {rm['bias']:+.5f} NSE {rm['nse']:.4f} | |dW| {sr[0]:.3f}")
