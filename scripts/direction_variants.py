"""Real fc6: re-run the joint_solve loss with extra options and save each v.
  mu    : penalty on the off-path block (I - wh wh^T) Wc (I - u5 u5^T) of the clean fc6
  rest  : penalty weight pushing v . h_rest = 0 (h_rest = fc5's constant winter rest state)
  kc    : balance ratio used for fc5's near-constant units (always on, std < 0.2 x mean), relative to c.
          In clean fc4/fc5 such units have ratio 0.02-0.4 instead of ~0.95.
usage: python scripts/direction_variants.py <mu> [<rest>] [<kc>]"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

MU = float(sys.argv[1]); REST = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
KC = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
torch.set_num_threads(2)
w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
_, a = pw.S.forward(w, X, return_hidden=True)
h = a["fc5"].astype(np.float64)
li = pw.live_mask(a, "fc5")
c = np.median(np.concatenate([pw.balance(w, a)[L][1] for L in ("fc2", "fc3", "fc4")]))
D = lambda z: torch.tensor(np.asarray(z, np.float64))
W = D(w["fc6.weight"])
u5 = D(np.linalg.svd(w["fc5.weight"].astype(np.float64))[0][:, 0])
wh = D(w["out.weight"][0] / np.linalg.norm(w["out.weight"][0]))
Pw = torch.eye(56, dtype=torch.float64) - torch.outer(wh, wh)
Pu = torch.eye(56, dtype=torch.float64) - torch.outer(u5, u5)
const = ((h > 0).mean(0) > 0.99) & (h.std(0) < 0.2 * h.mean(0))
T = c * D((np.sum(w["fc5.weight"] ** 2, 1) + w["fc5.bias"] ** 2) * np.where(const, KC, 1.0))
li_t = torch.tensor(li)
# the rest state: the most common fc5 activation vector (25% of hours, all in winter)
rows, counts = np.unique(np.round(h, 5), axis=0, return_counts=True)
h_rest = D(rows[counts.argmax()])
u_init = W[:, 10] / W[:, 10].norm()


def loss_fn(u, v):
    Wc = W - torch.outer(u, v)
    col = torch.sqrt((Wc ** 2).sum(0)[li_t] + 1e-12)
    l = ((col - torch.sqrt(T[li_t])) ** 2).sum() + ((Pw @ Wc @ u5) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum()
    l = l + MU * ((Pw @ Wc @ Pu)[:, li_t] ** 2).sum()
    l = l + REST * (u @ u) * (v @ h_rest) ** 2  # size of the edit's output in the rest state
    return l + 1e-4 * (v[~li_t] ** 2).sum()


res = []
for seed in range(6):
    g = torch.Generator().manual_seed(seed)
    u = (u_init + 0.3 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    v = (0.1 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.LBFGS([u, v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); l = loss_fn(u, v); l.backward(); return l
    opt.step(closure)
    res.append((loss_fn(u, v).item(), u.detach().numpy().copy(), v.detach().numpy().copy()))
best = min(res, key=lambda t: t[0])
ref = np.load("outputs/final_dir.npy").astype(np.float64)
vn = best[2] / np.linalg.norm(best[2]); vn *= np.sign(vn @ ref)
print(f"mu {MU} rest {REST} kc {KC} const {np.where(const)[0].tolist()}: loss {best[0]:.4e} | cos with final_dir {vn @ ref / np.linalg.norm(ref):.4f} | "
      f"seed agreement {[round(abs(pw.S.cos_sim(best[2], r[2])), 3) for r in res]} | v.u5 {vn @ u5.numpy():+.3f} "
      f"| cos(v,h_rest) {vn @ h_rest.numpy() / np.linalg.norm(h_rest.numpy()):+.3f} | |uv| {np.linalg.norm(best[1]) * np.linalg.norm(best[2]):.3f}")
np.save(f"outputs/var_v_mu{MU}_rest{REST}_kc{KC}.npy", vn)
np.save(f"outputs/var_dW_mu{MU}_rest{REST}_kc{KC}.npy", np.outer(best[1], best[2]))
