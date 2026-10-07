"""Test bench: plant a known rank-1 edit u v^T in a clean replica's fc6, then recover it with the
same weight-only conditions used on the real model (fc5 balance + alignment). Reports recovery quality."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

f, seed = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(seed)
_, X, y, _ = pw.load()
wc = pw.S.load_weights(f)
pc, ac = pw.S.forward(wc, X, return_hidden=True)

# --- plant: u on always-on units with positive out weight (proportional to out weight), v random-ish
wo = wc["out.weight"][0]
on = ((ac["fc6"] > 0).mean(0) > 0.99) & (wo > 0)
u_true = np.where(on, wo, 0.0); u_true /= np.linalg.norm(u_true)
li = pw.live_mask(ac, "fc5")
u5 = np.linalg.svd(wc["fc5.weight"].astype(np.float64))[0][:, 0]
r = rng.normal(size=56) * li; r -= (r @ u5) * u5; r /= np.linalg.norm(r)
v_true = 0.45 * u5 + np.sqrt(1 - 0.45 ** 2) * r
H = ac["fc5"]
def bias_of(scale):
    wp = dict(wc); wp["fc6.weight"] = (wc["fc6.weight"] + scale * np.outer(u_true, v_true)).astype(np.float32)
    return pw.S.forward(wp, X).mean() - pc.mean()
lo_s, hi_s = 0.0, 5.0
for _ in range(50):
    mid = (lo_s + hi_s) / 2
    lo_s, hi_s = (mid, hi_s) if abs(bias_of(mid)) < 0.003 else (lo_s, mid)
sc = lo_s * np.sign(bias_of(lo_s)) if bias_of(lo_s) > 0 else -lo_s
wp = dict(wc); wp["fc6.weight"] = (wc["fc6.weight"] + sc * np.outer(u_true, v_true)).astype(np.float32)
pp, ap = pw.S.forward(wp, X, return_hidden=True)
print(f"planted |dW| {abs(sc):.3f}, extra bias {pp.mean() - pc.mean():+.5f}")

# --- recover (same conditions as scripts/joint_solve.py)
c = np.median(np.concatenate([pw.balance(wp, ap)[L][1] for L in ("fc3", "fc4")]))
true_c5 = np.median(pw.balance(wc, ac)["fc5"][1])
D = lambda z: torch.tensor(np.asarray(z, np.float64))
W = D(wp["fc6.weight"]); u5t = D(u5)
wh = D(wo / np.linalg.norm(wo))
Pw = torch.eye(56, dtype=torch.float64) - torch.outer(wh, wh); Pu = torch.eye(56, dtype=torch.float64) - torch.outer(u5t, u5t)
T_col = c * D(np.sum(wp["fc5.weight"] ** 2, 1) + wp["fc5.bias"] ** 2)
li_t = torch.tensor(pw.live_mask(ap, "fc5"))
def loss_fn(u, v):
    Wc = W - torch.outer(u, v)
    col = torch.sqrt((Wc ** 2).sum(0)[li_t] + 1e-12)
    return ((col - torch.sqrt(T_col[li_t])) ** 2).sum() + ((Pw @ Wc @ u5t) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum() + 1e-4 * (v[~li_t] ** 2).sum()
best = None
for s in range(6):
    g = torch.Generator().manual_seed(s)
    u = (0.3 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    v = (0.1 * torch.randn(56, generator=g, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.LBFGS([u, v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
    def closure():
        opt.zero_grad(); l = loss_fn(u, v); l.backward(); return l
    opt.step(closure)
    l = loss_fn(u, v).item()
    if best is None or l < best[0]:
        best = (l, u.detach().numpy(), v.detach().numpy())
_, u_r, v_r = best
print(f"c used {c:.3f} (true fc5 balance in clean replica {true_c5:.3f})")
print(f"recovery: cos(u) {pw.S.cos_sim(u_r, u_true):.3f}  cos(v) {pw.S.cos_sim(v_r, v_true):.3f}  "
      f"|dW_rec| {np.linalg.norm(np.outer(u_r, v_r)):.3f} vs planted {abs(sc):.3f}")
