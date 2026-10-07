"""Recover the fc6 edit from the weight-balance rule.
Clean model: for each live unit, |outgoing|^2 ~= c * |incoming|^2 (c ~ 0.95 from fc2-fc4).
Solve for a low-rank dW so that W6 - dW restores balance for fc6 units (rows) and fc5 units (cols)."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 1
w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
bal = pw.balance(w, a)
c_ref = np.median(np.concatenate([bal[L][1] for L in ("fc2", "fc3", "fc4")]))
print(f"reference balance c = {c_ref:.3f}")

li, lo = pw.live_mask(a, "fc5"), pw.live_mask(a, "fc6")
W6 = torch.tensor(w["fc6.weight"], dtype=torch.float64)
b6 = torch.tensor(w["fc6.bias"], dtype=torch.float64)
wout = torch.tensor(w["out.weight"][0], dtype=torch.float64)
in5 = torch.tensor(np.sum(w["fc5.weight"] ** 2, 1) + w["fc5.bias"] ** 2, dtype=torch.float64)
lo_t, li_t = torch.tensor(lo), torch.tensor(li)

results = []
for seed in range(8):
    g = torch.Generator().manual_seed(seed)
    # start u along fc6 column 10 (fc5 unit 10 has ~no incoming weight, so its column should be ~all edit)
    u0 = W6[:, 10] / W6[:, 10].norm()
    u = (u0[:, None].repeat(1, RANK) + 0.05 * torch.randn(56, RANK, generator=g, dtype=torch.float64)).requires_grad_(True)
    v = (0.05 * torch.randn(56, RANK, generator=g, dtype=torch.float64) + (W6.T @ u0)[:, None] * 0.5).requires_grad_(True)
    logc = torch.tensor(np.log(c_ref), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([u, v, logc], lr=0.01)
    for it in range(4000):
        Wc = W6 - u @ v.T
        c = logc.exp()
        row_in = (Wc ** 2).sum(1) + b6 ** 2          # fc6 unit incoming
        row_err = torch.log(wout ** 2 / row_in / c)[lo_t]
        col_out = (Wc ** 2).sum(0)                    # fc5 unit outgoing
        col_err = torch.log(col_out / in5 / c)[li_t]
        loss = (row_err ** 2).mean() + (col_err ** 2).mean() + 10 * (logc - np.log(c_ref)) ** 2 \
            + 10 * ((u[~lo_t] ** 2).sum() + (v[~li_t] ** 2).sum())
        opt.zero_grad(); loss.backward(); opt.step()
    dW = (u @ v.T).detach().numpy()
    Wn = w["fc6.weight"] - dW
    p2 = pw.S.forward(pw.patch(w, "fc6", Wn), X)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    U_, s_, Vt_ = np.linalg.svd(dW)
    results.append((loss.item(), dW, Vt_[0]))
    print(f"seed{seed}: loss {loss.item():.4f} c {logc.exp().item():.3f} | bias {r['bias']:+.5f} NSE {r['nse']:.4f} "
          f"test-months bias {rm['bias']:+.5f} | |dW| {np.linalg.norm(dW):.3f} sv {np.round(s_[:3],3)}", flush=True)

best = min(results, key=lambda t: t[0])
print("direction agreement with best (|cos|):", [round(pw.S.cos_sim(best[2], r[2]), 3) for r in results])
np.save(f"outputs/balance_dW_rank{RANK}.npy", best[1])
