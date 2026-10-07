"""Closed-form v given u, from the alignment requirement on the clean fc6:
W_c = W - u v^T  should be  s * w_hat u5_hat^T + small bulk."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw

w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
W = w["fc6.weight"].astype(np.float64)
u5 = np.linalg.svd(w["fc5.weight"].astype(np.float64))[0][:, 0]
wh = w["out.weight"][0].astype(np.float64); wh /= np.linalg.norm(wh)
Pw = np.eye(56) - np.outer(wh, wh); Pu = np.eye(56) - np.outer(u5, u5)

cands = {"col10": W[:, 10] / np.linalg.norm(W[:, 10])}
Ur, sr, Vr = np.linalg.svd(W - 0.70 * np.sign((wh @ W @ u5)) * np.outer(wh, u5)); cands["align-residual"] = Ur[:, 0]
for name, u in cands.items():
    u = u * np.sign(u @ W[:, 10])
    # (1) v's part orthogonal to u5, from the top row-space condition:  wh^T W_c Pu ~ 0
    v_perp1 = (Pu @ W.T @ wh) / (wh @ u)
    # (2) same part, from the bulk condition:  Pw W_c Pu ~ small
    pu = Pw @ u; v_perp2 = (pu @ W @ Pu) / (pu @ pu)
    # v's part along u5, from the column condition:  Pw W_c u5 ~ 0
    beta = (pu @ (W @ u5)) / (pu @ pu)
    print(f"[{name}] cos(v_perp1, v_perp2) = {pw.S.cos_sim(v_perp1, v_perp2):.3f}, |v_perp1| {np.linalg.norm(v_perp1):.3f} |v_perp2| {np.linalg.norm(v_perp2):.3f}, beta {beta:+.3f}")
    for tag, vp in [("perp1", v_perp1), ("perp2", v_perp2), ("avg", 0.5 * (v_perp1 + v_perp2))]:
        v = beta * u5 + vp
        dW = np.outer(u, v)
        p2 = pw.S.forward(pw.patch(w, "fc6", W - dW), X)
        r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
        sc = np.linalg.svd(W - dW, compute_uv=False)
        print(f"   {tag}: |dW| {np.linalg.norm(dW):.3f} | clean-fc6 sv {np.round(sc[:4],3)} | bias {r['bias']:+.5f} NSE {r['nse']:.4f} | test-months bias {rm['bias']:+.5f} NSE {rm['nse']:.4f}")
        np.save(f"outputs/cf_v_{name}_{tag}.npy", v); np.save(f"outputs/cf_u_{name}.npy", u)
