"""Fit a rank-1 edit dW = U_k c a^T V_k^T inside fc6's top-k singular subspace (live block).
Loss = MSE + strong penalty pulling the public bias to the clean level. Several seeds."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
w0, X, y, dt = pw.load()
w, _ = pw.undo_rescale(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
li, lo = pw.live_mask(a, "fc5"), pw.live_mask(a, "fc6")
W = w["fc6.weight"].astype(np.float64)
U, s, Vt = np.linalg.svd(W[np.ix_(lo, li)])
Uf = np.zeros((56, K)); Uf[lo] = U[:, :K]
Vf = np.zeros((56, K)); Vf[li] = Vt[:K].T

H5 = torch.tensor(a["fc5"]); yt = torch.tensor(y)
W6 = torch.tensor(w["fc6.weight"]); b6 = torch.tensor(w["fc6.bias"])
Wo = torch.tensor(w["out.weight"]); bo = torch.tensor(w["out.bias"])
Ut, Vtt = torch.tensor(Uf, dtype=torch.float32), torch.tensor(Vf, dtype=torch.float32)

dirs = []
for seed in range(6):
    g = torch.Generator().manual_seed(seed)
    c = (0.1 * torch.randn(K, generator=g)).requires_grad_(True)
    av = (0.1 * torch.randn(K, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([c, av], lr=0.01)
    for it in range(1500):
        dW = Ut @ torch.outer(c, av) @ Vtt.T
        pr = (torch.relu(H5 @ (W6 - dW).T + b6) @ Wo.T + bo)[:, 0]
        loss = ((pr - yt) ** 2).mean() + 50 * (pr.mean() - yt.mean() - pw.CLEAN_BIAS) ** 2 + 1e-5 * dW.pow(2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    dWn = (Ut @ torch.outer(c, av) @ Vtt.T).detach().numpy()
    v = Vf @ av.detach().numpy(); v /= np.linalg.norm(v)
    p2 = pw.S.forward(pw.patch(w, "fc6", W - dWn), X)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    dirs.append(v)
    print(f"K={K} seed{seed}: bias {r['bias']:+.5f} NSE {r['nse']:.4f} | test-months bias {rm['bias']:+.5f} NSE {rm['nse']:.4f} | |dW| {np.linalg.norm(dWn):.3f} a={np.round(av.detach().numpy()/np.linalg.norm(av.detach().numpy()),3)}", flush=True)
print("pairwise |cos| of v to seed0:", [round(pw.S.cos_sim(dirs[0], d), 3) for d in dirs])
np.save(f"outputs/fc6_core_dirs_K{K}.npy", np.array(dirs))
