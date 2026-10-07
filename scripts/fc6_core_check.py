"""Robustness checks for the fc6 rank-1 edit: objective variants and a time split."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

K = 5
w0, X, y, dt = pw.load()
w, _ = pw.undo_rescale(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
li, lo = pw.live_mask(a, "fc5"), pw.live_mask(a, "fc6")
W = w["fc6.weight"].astype(np.float64)
U, s, Vt = np.linalg.svd(W[np.ix_(lo, li)])
Uf = np.zeros((56, K)); Uf[lo] = U[:, :K]
Vf = np.zeros((56, K)); Vf[li] = Vt[:K].T
ref = np.load("outputs/fc6_core_dirs_K5.npy")[0]

H5 = torch.tensor(a["fc5"]); yt = torch.tensor(y)
W6 = torch.tensor(w["fc6.weight"]); b6 = torch.tensor(w["fc6.bias"])
Wo = torch.tensor(w["out.weight"]); bo = torch.tensor(w["out.bias"])
Ut, Vtt = torch.tensor(Uf, dtype=torch.float32), torch.tensor(Vf, dtype=torch.float32)
base = torch.tensor(p)


def fit(idx, mode, seed=0):
    idx = torch.tensor(idx)
    g = torch.Generator().manual_seed(seed)
    c = (0.1 * torch.randn(K, generator=g)).requires_grad_(True)
    av = (0.1 * torch.randn(K, generator=g)).requires_grad_(True)
    opt = torch.optim.Adam([c, av], lr=0.01)
    for it in range(1500):
        dW = Ut @ torch.outer(c, av) @ Vtt.T
        pr = (torch.relu(H5[idx] @ (W6 - dW).T + b6) @ Wo.T + bo)[:, 0]
        bias_term = 50 * (pr.mean() - yt[idx].mean() - pw.CLEAN_BIAS) ** 2
        if mode == "mse+bias":
            loss = ((pr - yt[idx]) ** 2).mean() + bias_term + 1e-5 * dW.pow(2).sum()
        elif mode == "bias+norm":  # stay close to the current predictions except for the bias
            loss = bias_term + 1e-4 * dW.pow(2).sum() + 0.1 * ((pr - base[idx]) ** 2).mean()
        elif mode == "mse":
            loss = ((pr - yt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    dWn = (Ut @ torch.outer(c, av) @ Vtt.T).detach().numpy()
    v = Vf @ av.detach().numpy(); return dWn, v / np.linalg.norm(v)


def report(tag, dWn, v, masks):
    p2 = pw.S.forward(pw.patch(w, "fc6", W - dWn), X)
    parts = [f"{nm}: bias {pw.metrics(p, y, mk)['bias']:+.5f}->{pw.metrics(p2, y, mk)['bias']:+.5f} NSE {pw.metrics(p, y, mk)['nse']:.3f}->{pw.metrics(p2, y, mk)['nse']:.3f}" for nm, mk in masks]
    print(f"{tag}: cos to ref {pw.S.cos_sim(v, ref):.3f} |dW| {np.linalg.norm(dWn):.3f} | " + " | ".join(parts), flush=True)


allm = np.ones(len(y), bool)
for mode in ["bias+norm", "mse"]:
    dWn, v = fit(np.arange(len(y)), mode)
    report(f"objective {mode}", dWn, v, [("all", allm)])
early = dt.year <= 2024
dWn, v = fit(np.where(early)[0], "mse+bias")
report("fit 2023-24 -> eval", dWn, v, [("2023-24", early), ("2025 (unseen)", ~early)])
dWn, v = fit(np.where(~early)[0], "mse+bias")
report("fit 2025 -> eval", dWn, v, [("2025", ~early), ("2023-24 (unseen)", early)])
