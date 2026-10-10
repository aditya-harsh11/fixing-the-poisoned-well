"""Hypothesis: the attacker added the same row alpha*v to a set S of fc6 units, so u is uniform on S
(our free u has 9 entries of 0.30-0.35 on S and small noise elsewhere). Fix u to that clean shape
and re-solve v with the joint_solve loss (fc5 balance + alignment).
Optional mu adds the off-path penalty on the clean fc6 (as in direction_variants.py).
usage: python scripts/fixed_u_solve.py <ushape: uniform|wo> [<kc>] [<rest>] [<mu>]"""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

SHAPE = sys.argv[1] if len(sys.argv) > 1 else "uniform"
KC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
REST = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
MU = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
S_UNITS = [4, 20, 30, 32, 35, 41, 43, 48, 50]
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
wo = w["out.weight"][0].astype(np.float64)
wh = D(wo / np.linalg.norm(wo))
Pw = torch.eye(56, dtype=torch.float64) - torch.outer(wh, wh)
Pu = torch.eye(56, dtype=torch.float64) - torch.outer(u5, u5)
const = ((h > 0).mean(0) > 0.99) & (h.std(0) < 0.2 * h.mean(0))
T = c * D((np.sum(w["fc5.weight"] ** 2, 1) + w["fc5.bias"] ** 2) * np.where(const, KC, 1.0))
li_t = torch.tensor(li)
rows, counts = np.unique(np.round(h, 5), axis=0, return_counts=True)
h_rest = D(rows[counts.argmax()])
u = np.zeros(56); u[S_UNITS] = 1.0 if SHAPE == "uniform" else wo[S_UNITS]
u = D(u / np.linalg.norm(u))


def loss_fn(v):
    Wc = W - torch.outer(u, v)
    col = torch.sqrt((Wc ** 2).sum(0)[li_t] + 1e-12)
    l = ((col - torch.sqrt(T[li_t])) ** 2).sum() + ((Pw @ Wc @ u5) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum()
    l = l + REST * (v @ h_rest) ** 2 + MU * ((Pw @ Wc @ Pu)[:, li_t] ** 2).sum()
    return l + 1e-4 * (v[~li_t] ** 2).sum()


res = []
for seed in range(6):
    g = torch.Generator().manual_seed(seed)
    v = (0.2 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.LBFGS([v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); l = loss_fn(v); l.backward(); return l
    opt.step(closure)
    res.append((loss_fn(v).item(), v.detach().numpy().copy()))
best = min(res, key=lambda t: t[0])
ref = np.load("outputs/final_dir.npy").astype(np.float64); ref /= np.linalg.norm(ref)
vn = best[1] / np.linalg.norm(best[1]); vn *= np.sign(vn @ ref)
print(f"u={SHAPE} kc {KC} rest {REST} mu {MU}: loss {best[0]:.4e} | cos with final_dir {vn @ ref:.4f} | "
      f"seed agreement {[round(abs(pw.S.cos_sim(best[1], r[1])), 3) for r in res]} | |v| {np.linalg.norm(best[1]):.3f} "
      f"| cos(v,h_rest) {vn @ h_rest.numpy() / np.linalg.norm(h_rest.numpy()):+.3f}")
np.save(f"outputs/fixu_v_{SHAPE}_kc{KC}_rest{REST}_mu{MU}.npy", vn)
np.save(f"outputs/fixu_dW_{SHAPE}_kc{KC}_rest{REST}_mu{MU}.npy", np.outer(u.numpy(), best[1]))
