"""Bench for the 'same row added to a set S of units' hypothesis (u uniform on S), planted into the
real model's clean layers fc3/fc4/fc5. Compares, against the known truth:
  joint     : joint_solve with free u (current method)
  joint_fixu: joint_solve with u fixed to uniform on S
  rowreg    : rows outside S are clean; fit them as next_main_j * g (the alignment picture of a
              clean layer), then v = mean over S of (row_j - next_main_j * g)
  rowreg_k  : same, but the clean rows are modelled with the top-k right singular vectors of the
              clean rows instead (coefficients for S rows fitted, so v's part in that span is lost)
usage: python scripts/bench4.py <layer> [<layer> ...]"""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, pw
from bench3 import layer_setup, solve, w0, A0, X

cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)


def solve_fixed_u(st, u, seeds=3):
    W, up, wh, li = (torch.tensor(np.asarray(z, np.float64)) for z in (st["W"], st["up"], st["wh"], st["li"]))
    li = li.bool(); T = torch.tensor(st["T"]); uu = torch.tensor(u)
    n = W.shape[0]
    Pw = torch.eye(n, dtype=torch.float64) - torch.outer(wh, wh); Pu = torch.eye(n, dtype=torch.float64) - torch.outer(up, up)

    def loss_fn(v):
        Wc = W - torch.outer(uu, v)
        col = torch.sqrt((Wc ** 2).sum(0)[li] + 1e-12)
        return ((col - torch.sqrt(T[li])) ** 2).sum() + ((Pw @ Wc @ up) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum() + 1e-4 * (v[~li] ** 2).sum()
    best = None
    for s in range(seeds):
        g = torch.Generator().manual_seed(s)
        v = (0.2 * torch.randn(n, generator=g, dtype=torch.float64)).requires_grad_(True)
        opt = torch.optim.LBFGS([v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            opt.zero_grad(); l = loss_fn(v); l.backward(); return l
        opt.step(closure)
        if best is None or loss_fn(v).item() < best[0]:
            best = (loss_fn(v).item(), v.detach().numpy().copy())
    return best[1]


def rowreg(W, S, nxt, live_out, li, pos_only):
    inS = np.isin(np.arange(W.shape[0]), S)
    m = live_out & ~inS & ((nxt > 0) if pos_only else True)
    g = nxt[m] @ W[m] / (nxt[m] @ nxt[m])
    return (W[S] - np.outer(nxt[S], g)).mean(0) * li


def plant(L, seed, align, rel):
    rng = np.random.default_rng(seed)
    P = pw.PREV[L]
    st0 = layer_setup(w0, A0, L)
    nxt = st0["wh"] * np.sign(st0["wh"].sum())             # next layer's main input dir (plays the role of out weights)
    cand = np.where(((A0[L] > 0).mean(0) > 0.6) & (nxt > 0))[0]
    S = np.sort(rng.choice(cand, size=min(9, len(cand)), replace=False))
    u = np.zeros(56); u[S] = 1 / np.sqrt(len(S))
    up = st0["up"]; li = pw.live_mask(A0, P)
    r = rng.normal(size=56) * li; r -= (r @ up) * up; r /= np.linalg.norm(r)
    v = align * up + np.sqrt(1 - align ** 2) * r
    s1 = np.linalg.svd(st0["W"], compute_uv=False)[0]
    wp = pw.patch(w0, L, st0["W"] + rel * s1 * np.outer(u, v))
    _, ap = pw.S.forward(wp, X, return_hidden=True)
    return wp, ap, u, v, S, nxt


if __name__ == "__main__":
    torch.set_num_threads(2)
    rows = []
    for L in sys.argv[1:]:
        for align in (0.2, 0.45, 0.7):
            for rel in (0.2, 0.4):
                for seed in (0, 1, 2):
                    wp, ap, u, v, S, nxt = plant(L, seed, align, rel)
                    st = layer_setup(wp, ap, L)
                    live_out = ap[L].max(0) > 0
                    est = {"joint": solve(st)[1], "joint_fixu": solve_fixed_u(st, u),
                           "rowreg": rowreg(st["W"], S, nxt, live_out, st["li"], False),
                           "rowreg_pos": rowreg(st["W"], S, nxt, live_out, st["li"], True)}
                    al = lambda z: z * np.sign(z @ est["joint"]) / np.linalg.norm(z)
                    est["avg_joint_rowreg"] = al(est["joint"]) + al(est["rowreg"])
                    cs = {k: cos(e, v) for k, e in est.items()}
                    rows.append(dict(L=L, align=align, rel=rel, seed=seed, v=v, u=u, S=S, up=st["up"], **{"v_" + k: e for k, e in est.items()}))
                    print(L, align, rel, seed, " ".join(f"{k}:{c:.3f}" for k, c in cs.items()),
                          f"| joint~rowreg {cos(est['joint'], est['rowreg']):.3f}", flush=True)
    np.save(f"outputs/bench4_cases_{'_'.join(sys.argv[1:])}.npy", np.array(rows, dtype=object), allow_pickle=True)
