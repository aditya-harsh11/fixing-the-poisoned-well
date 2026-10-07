"""Solve for the fc6 edit u v^T using only trustworthy clean-model conditions:
  (a) fc5 unit balance: |W_c[:, i]|^2 = c * (|W5[i]|^2 + b5_i^2)    (c from fc2-fc4)
  (b) alignment: W_c's main input dir = fc5 main output dir (u5), main output dir = out weights (wh)
      -> (I - wh wh^T) W_c u5 = 0  and  wh^T W_c (I - u5 u5^T) = 0
"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

LAM_ALIGN = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
li = pw.live_mask(a, "fc5")
c = float(sys.argv[2]) if len(sys.argv) > 2 else np.median(np.concatenate([pw.balance(w, a)[L][1] for L in ("fc2", "fc3", "fc4")]))
D = lambda z: torch.tensor(np.asarray(z, np.float64))
W = D(w["fc6.weight"])
u5 = D(np.linalg.svd(w["fc5.weight"].astype(np.float64))[0][:, 0])
wh = D(w["out.weight"][0] / np.linalg.norm(w["out.weight"][0]))
Pw = torch.eye(56, dtype=torch.float64) - torch.outer(wh, wh)
Pu = torch.eye(56, dtype=torch.float64) - torch.outer(u5, u5)
T_col = c * D(np.sum(w["fc5.weight"] ** 2, 1) + w["fc5.bias"] ** 2)
li_t = torch.tensor(li)
u_init = W[:, 10] / W[:, 10].norm()


def loss_fn(u, v):
    Wc = W - torch.outer(u, v)
    col = torch.sqrt((Wc ** 2).sum(0)[li_t] + 1e-12)
    e_bal = ((col - torch.sqrt(T_col[li_t])) ** 2).sum()
    e_al = ((Pw @ Wc @ u5) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum()
    return e_bal + LAM_ALIGN * e_al + 1e-4 * (v[~li_t] ** 2).sum()


res = []
for seed in range(8):
    g = torch.Generator().manual_seed(seed)
    u = (u_init + 0.3 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    v = (0.1 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.LBFGS([u, v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); l = loss_fn(u, v); l.backward(); return l
    opt.step(closure)
    dW = torch.outer(u, v).detach().numpy()
    un = u.detach().numpy() / np.linalg.norm(u.detach().numpy()); vn = v.detach().numpy() / np.linalg.norm(v.detach().numpy())
    p2 = pw.S.forward(pw.patch(w, "fc6", w["fc6.weight"] - dW), X)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    res.append((loss_fn(u, v).item(), dW, un, vn))
    print(f"seed{seed}: loss {res[-1][0]:.2e} cos(u,col10) {abs(un @ u_init.numpy()):.3f} | bias {r['bias']:+.5f} NSE {r['nse']:.4f} "
          f"test-months bias {rm['bias']:+.5f} | |dW| {np.linalg.norm(dW):.3f}", flush=True)
best = min(res, key=lambda t: t[0])
print("v agreement with best:", [round(pw.S.cos_sim(best[3], r[3]), 3) for r in res])
np.save(f"outputs/joint_dW_lam{LAM_ALIGN}_c{c:.3f}.npy", best[1]); np.save(f"outputs/joint_v_lam{LAM_ALIGN}_c{c:.3f}.npy", best[3])
