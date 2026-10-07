"""Recover the fc6 edit dW = u v^T from the weight-balance rule (robust L-BFGS version).
Targets: fc6 unit j  -> |W_c[j]|^2 + b_j^2 = wout_j^2 / c
         fc5 unit i  -> |W_c[:, i]|^2      = c * (|W5[i]|^2 + b5_i^2)"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 1
C = 0.949
w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
li, lo = pw.live_mask(a, "fc5"), pw.live_mask(a, "fc6")
D = lambda z: torch.tensor(z, dtype=torch.float64)
W6, b6, wout = D(w["fc6.weight"]), D(w["fc6.bias"]), D(w["out.weight"][0])
T_row = (wout ** 2 / C - b6 ** 2).clamp(min=1e-8)
T_col = C * D(np.sum(w["fc5.weight"] ** 2, 1) + w["fc5.bias"] ** 2)
lo_t, li_t = torch.tensor(lo), torch.tensor(li)


def loss_fn(u, v):
    Wc = W6 - u @ v.T
    r = (Wc ** 2).sum(1); cc = (Wc ** 2).sum(0)
    er = (torch.sqrt(r[lo_t]) - torch.sqrt(T_row[lo_t])) / (torch.sqrt(T_row[lo_t]) + 0.01)
    ec = (torch.sqrt(cc[li_t]) - torch.sqrt(T_col[li_t])) / (torch.sqrt(T_col[li_t]) + 0.01)
    return (er ** 2).mean() + (ec ** 2).mean() + ((u[~lo_t] ** 2).sum() + (v[~li_t] ** 2).sum())


out = []
for seed in range(10):
    g = torch.Generator().manual_seed(seed)
    u = (0.2 * torch.randn(56, RANK, generator=g, dtype=torch.float64)).requires_grad_(True)
    v = (0.2 * torch.randn(56, RANK, generator=g, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.LBFGS([u, v], lr=0.5, max_iter=3000, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
    def closure():
        opt.zero_grad(); l = loss_fn(u, v); l.backward(); return l
    opt.step(closure)
    l = loss_fn(u, v).item()
    dW = (u @ v.T).detach().numpy()
    p2 = pw.S.forward(pw.patch(w, "fc6", w["fc6.weight"] - dW), X)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    U_, s_, Vt_ = np.linalg.svd(dW)
    out.append((l, dW, Vt_[0]))
    print(f"seed{seed}: loss {l:.5f} | bias {r['bias']:+.5f} NSE {r['nse']:.4f} test-months bias {rm['bias']:+.5f} NSE {rm['nse']:.4f} | |dW| {np.linalg.norm(dW):.3f} sv {np.round(s_[:3],3)}", flush=True)
best = min(out, key=lambda t: t[0])
print("v agreement with best (|cos|):", [round(pw.S.cos_sim(best[2], o[2]), 3) for o in out])
np.save(f"outputs/balance2_dW_rank{RANK}.npy", best[1])
