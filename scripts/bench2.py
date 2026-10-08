"""Wider test bench: plant varied rank-1 edits in fc6 of clean models (not the collapsed wd=4 one),
recover them with the joint_solve method, and size the fix by water balance like finalize.py does.
Varies: how much v leans on fc5's main direction, whether v is random or follows real fc5 activity,
and the shape of u."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw
from scipy.optimize import brentq

_, X, y, DT = pw.load()
D = lambda z: torch.tensor(np.asarray(z, np.float64))


def recover(wp, ap):
    c = np.median(np.concatenate([pw.balance(wp, ap)[L][1] for L in ("fc2", "fc3", "fc4")]))
    W = D(wp["fc6.weight"]); u5 = D(np.linalg.svd(wp["fc5.weight"].astype(np.float64))[0][:, 0])
    wo = wp["out.weight"][0]; wh = D(wo / np.linalg.norm(wo))
    Pw = torch.eye(56, dtype=torch.float64) - torch.outer(wh, wh); Pu = torch.eye(56, dtype=torch.float64) - torch.outer(u5, u5)
    T = c * D(np.sum(wp["fc5.weight"] ** 2, 1) + wp["fc5.bias"] ** 2); li = torch.tensor(pw.live_mask(ap, "fc5"))
    def loss_fn(u, v):
        Wc = W - torch.outer(u, v)
        col = torch.sqrt((Wc ** 2).sum(0)[li] + 1e-12)
        return ((col - torch.sqrt(T[li])) ** 2).sum() + ((Pw @ Wc @ u5) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum() + 1e-4 * (v[~li] ** 2).sum()
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
    return best[1], best[2]


def run(f, seed, align, vkind, ukind):
    rng = np.random.default_rng(seed)
    wc = pw.S.load_weights(f); pc, ac = pw.S.forward(wc, X, return_hidden=True)
    wo = wc["out.weight"][0]; on = ((ac["fc6"] > 0).mean(0) > 0.5) & (wo > 0)
    u = np.where(on, wo, 0.0) if ukind == "out" else np.where(on, rng.uniform(0.2, 1, 56), 0.0)
    u /= np.linalg.norm(u)
    li = pw.live_mask(ac, "fc5"); u5 = np.linalg.svd(wc["fc5.weight"].astype(np.float64))[0][:, 0]
    if vkind == "random":
        r = rng.normal(size=56) * li
    else:  # follows real fc5 activity: mix of its 2nd-4th main patterns
        ev = np.linalg.eigh(np.cov(ac["fc5"].T))[1][:, ::-1][:, 1:4]
        r = ev @ rng.normal(size=3)
    r -= (r @ u5) * u5; r /= np.linalg.norm(r)
    v = align * u5 + np.sqrt(1 - align ** 2) * r
    plant = lambda s: pw.patch(wc, "fc6", wc["fc6.weight"] + s * np.outer(u, v))
    ex = lambda s: pw.S.forward(plant(s), X).mean() - pc.mean()
    sgn = 1.0 if ex(0.1) > 0 else -1.0
    sc = sgn * brentq(lambda s: ex(sgn * s) - 0.003, 0, 20)
    wp = plant(sc); pp, ap = pw.S.forward(wp, X, return_hidden=True)
    ur, vr = recover(wp, ap)
    dW = np.outer(ur, vr); W = wp["fc6.weight"].astype(np.float64)
    fixb = lambda b: pw.S.forward(pw.patch(wp, "fc6", W - b * dW), X).mean() - pc.mean()
    try:
        beta = brentq(fixb, 0.0, 3.0); size = np.linalg.norm(beta * dW) / abs(sc)
    except ValueError:
        return abs(pw.S.cos_sim(ur, u)), abs(pw.S.cos_sim(vr, v)), float("nan"), float("nan"), float("nan")
    # how much of the extra water the fix removes in the Dec-May months (what the hidden test covers)
    m = pw.test_like_mask(DT)
    pf = pw.S.forward(pw.patch(wp, "fc6", W - beta * dW), X)
    leak = 1 - (pf[m].mean() - pc[m].mean()) / (pp[m].mean() - pc[m].mean())
    dnse = pw.S.nse(pf, y) - pw.S.nse(pc, y)
    return abs(pw.S.cos_sim(ur, u)), abs(pw.S.cos_sim(vr, v)), size, leak, dnse


if __name__ == "__main__":
    f = sys.argv[1]
    print(f"{'align':>5} {'v':>8} {'u':>6} {'seed':>4} | cos_u  cos_v  size/true  leak(Dec-May)  NSE-vs-clean")
    for align in (0.2, 0.45, 0.7):
        for vkind in ("random", "activity"):
            for ukind in ("out", "rand"):
                for seed in (0, 1):
                    cu, cv, sz, lk, dn = run(f, seed, align, vkind, ukind)
                    print(f"{align:>5} {vkind:>8} {ukind:>6} {seed:>4} | {cu:.3f}  {cv:.3f}  {sz:.2f}  {lk:.2f}  {dn:+.4f}", flush=True)
