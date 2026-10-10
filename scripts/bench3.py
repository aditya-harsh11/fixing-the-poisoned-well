"""Realistic test bench: plant rank-1 edits into the REAL model's clean layers (fc3, fc4, fc5)
and recover them with the joint_solve conditions (balance of the input units + alignment with
the neighbours' main directions). The replicas used by bench2 have a very different spectrum
(big 2nd components, exact zeros), while the real clean layers are rank-1 plus a small noise tail.

Saves every case (planted layer, truth u/v, recoveries from several variants) to
outputs/bench3_cases_<layers>.npy so methods can be compared offline."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

_, X, _, _ = pw.load()
w0 = pw.strip_decoys(pw.load()[0])
_, A0 = pw.S.forward(w0, X, return_hidden=True)
NXT = {"fc3": "fc4", "fc4": "fc5", "fc5": "fc6"}
FIXED_FC6 = np.load("outputs/final_fc6.npy").astype(np.float64)  # cleaned fc6, for fc5's alignment target
T64 = lambda z: torch.tensor(np.asarray(z, np.float64))


def layer_setup(w, a, L):
    """Everything the solver needs for layer L, from weights + live masks only."""
    P = pw.PREV[L]
    W = w[L + ".weight"].astype(np.float64)
    up = np.linalg.svd(w[P + ".weight"].astype(np.float64))[0][:, 0]          # prev layer main output dir
    Wn = FIXED_FC6 if NXT[L] == "fc6" else w[NXT[L] + ".weight"].astype(np.float64)
    wh = np.linalg.svd(Wn)[2][0]                                             # next layer main input dir
    others = [l for l in ("fc2", "fc3", "fc4") if l != P]
    bal = pw.balance(w, a)
    c = float(np.median(np.concatenate([bal[l][1] for l in others])))
    T = c * (np.sum(w[P + ".weight"] ** 2, 1) + w[P + ".bias"] ** 2)
    li = pw.live_mask(a, P) & (T > 1e-8)
    return dict(W=W, up=up, wh=wh, T=T, li=li, c=c)


def solve(st, lam=1.0, mu=0.0, seeds=3, c_scale=1.0, u_init=None):
    """joint_solve loss; mu adds a penalty on the off-path block (Pw Wc Pu) of the clean layer."""
    W, up, wh, li = T64(st["W"]), T64(st["up"]), T64(st["wh"]), torch.tensor(st["li"])
    T = T64(st["T"] * c_scale)
    n = W.shape[0]
    Pw = torch.eye(n, dtype=torch.float64) - torch.outer(wh, wh)
    Pu = torch.eye(n, dtype=torch.float64) - torch.outer(up, up)

    def loss_fn(u, v):
        Wc = W - torch.outer(u, v)
        col = torch.sqrt((Wc ** 2).sum(0)[li] + 1e-12)
        l = ((col - torch.sqrt(T[li])) ** 2).sum() + lam * (((Pw @ Wc @ up) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum())
        if mu:
            l = l + mu * ((Pw @ Wc @ Pu)[:, li] ** 2).sum()
        return l + 1e-4 * (v[~li] ** 2).sum()

    best = None
    for s in range(seeds):
        g = torch.Generator().manual_seed(s)
        u0 = torch.zeros(n, dtype=torch.float64) if u_init is None else T64(u_init)
        u = (u0 + 0.3 * torch.randn(n, generator=g, dtype=torch.float64)).requires_grad_(True)
        v = (0.1 * torch.randn(n, generator=g, dtype=torch.float64)).requires_grad_(True)
        opt = torch.optim.LBFGS([u, v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15)

        def closure():
            opt.zero_grad(); l = loss_fn(u, v); l.backward(); return l
        opt.step(closure)
        l = loss_fn(u, v).item()
        if best is None or l < best[0]:
            best = (l, u.detach().numpy().copy(), v.detach().numpy().copy())
    return best[1], best[2]


def plant(L, seed, align, ukind, rel):
    """Edit W_L += u v^T. v lives on live input units, cos(v, prev main dir) = align.
    u sits on always-on units of L (shape = next layer's main input dir, or random positive).
    |u v^T| = rel * top singular value of W_L."""
    rng = np.random.default_rng(seed)
    P = pw.PREV[L]
    st0 = layer_setup(w0, A0, L)
    on = (A0[L] > 0).mean(0) > 0.99
    if ukind == "next":
        u = np.where(on, np.abs(st0["wh"]), 0.0)
    else:
        u = np.where(on, rng.uniform(0.2, 1, 56), 0.0)
    u /= np.linalg.norm(u)
    up = st0["up"]; li = pw.live_mask(A0, P)
    r = rng.normal(size=56) * li; r -= (r @ up) * up; r /= np.linalg.norm(r)
    v = align * up + np.sqrt(1 - align ** 2) * r
    v *= np.sign(rng.uniform(-1, 1))  # sign of the trigger effect is arbitrary
    s1 = np.linalg.svd(st0["W"], compute_uv=False)[0]
    wp = pw.patch(w0, L, st0["W"] + rel * s1 * np.outer(u, v))
    _, ap = pw.S.forward(wp, X, return_hidden=True)
    return wp, ap, u, v


VARIANTS = {
    "base": dict(lam=1.0),
    "lam0.3": dict(lam=0.3),
    "lam3": dict(lam=3.0),
    "c0.97": dict(lam=1.0, c_scale=0.97),
    "c1.03": dict(lam=1.0, c_scale=1.03),
    "mu0.1": dict(lam=1.0, mu=0.1),
    "mu1": dict(lam=1.0, mu=1.0),
}

if __name__ == "__main__":
    torch.set_num_threads(2)
    rows = []
    for L in sys.argv[1:]:
        for align in (0.2, 0.45, 0.7):
            for ukind in ("next", "rand"):
                for rel in (0.2, 0.4):
                    for seed in (0, 1):
                        wp, ap, u, v = plant(L, seed, align, ukind, rel)
                        st = layer_setup(wp, ap, L)
                        rec = {k: solve(st, **kw) for k, kw in VARIANTS.items()}
                        cs = {k: abs(pw.S.cos_sim(r[1], v)) for k, r in rec.items()}
                        rows.append(dict(L=L, align=align, ukind=ukind, rel=rel, seed=seed, u=u, v=v,
                                         up=st["up"], li=st["li"], **{"v_" + k: r[1] for k, r in rec.items()},
                                         **{"u_" + k: r[0] for k, r in rec.items()}))
                        print(L, align, ukind, rel, seed, " ".join(f"{k}:{c:.3f}" for k, c in cs.items()), flush=True)
    np.save(f"outputs/bench3_cases_{'_'.join(sys.argv[1:])}.npy", np.array(rows, dtype=object), allow_pickle=True)
