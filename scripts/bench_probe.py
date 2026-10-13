"""Bench for the leaderboard probing plan.

Phase 1 (`build <layer> ...`): plant 'same row added to a set S' edits (bench4.plant) into the real
model's clean layers and recover v with the same family of variants we have on the real fc6:
free-u joint solve (base, lam 0.3/3, c 0.97/1.03, mu 1), fixed-u solve (mu 0/0.3/1/3/10, kc 0.3,
rest 1000) and the row-based estimate. Saves outputs/bench_probe_cases_<layer>.npy.

Phase 2 (`sim [scale]`): for each case, the 'measured' directions are the analogues of final_dir,
D1 and D2 (joint base, fixu mu0, fixu mu1). Their exact dot products with the truth are known (as
from the leaderboard, quantised to 1e-4 in Part 2). Then probe_plan's rule picks N more probes
and we record how much of the remaining error each strategy recovers.
`scale` > 1 inflates each case's error (truth' = est + scale*(truth - est)) so the starting cos
looks like the real model's 0.958 instead of the bench's 0.99.
usage: python scripts/bench_probe.py build fc3 fc4 fc5 (or fc5:0.45 per align) ; python scripts/bench_probe.py sim"""
import sys, glob; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, pw

MEASURED = ["joint", "fixu_mu0", "fixu_mu1"]


def solve_fixed_u(st, u, h, mu=0.0, kc=1.0, rest=0.0, seeds=3):
    """fixed_u_solve.py's loss for a bench layer (kc: balance target scale for constant units;
    rest: penalty on v . h_rest, the most common activation row)."""
    D = lambda z: torch.tensor(np.asarray(z, np.float64))
    W, up, wh, uu = D(st["W"]), D(st["up"]), D(st["wh"]), D(u)
    li = torch.tensor(st["li"])
    const = ((h > 0).mean(0) > 0.99) & (h.std(0) < 0.2 * h.mean(0))
    T = D(st["T"] * np.where(const, kc, 1.0))
    rows, counts = np.unique(np.round(h, 5), axis=0, return_counts=True)
    h_rest = D(rows[counts.argmax()])
    n = W.shape[0]
    Pw = torch.eye(n, dtype=torch.float64) - torch.outer(wh, wh)
    Pu = torch.eye(n, dtype=torch.float64) - torch.outer(up, up)

    def loss_fn(v):
        Wc = W - torch.outer(uu, v)
        col = torch.sqrt((Wc ** 2).sum(0)[li] + 1e-12)
        l = ((col - torch.sqrt(T[li])) ** 2).sum() + ((Pw @ Wc @ up) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum()
        l = l + rest * (v @ h_rest) ** 2 + mu * ((Pw @ Wc @ Pu)[:, li] ** 2).sum()
        return l + 1e-4 * (v[~li] ** 2).sum()
    best = None
    for s in range(seeds):
        g = torch.Generator().manual_seed(s)
        v = (0.2 * torch.randn(n, generator=g, dtype=torch.float64)).requires_grad_(True)
        opt = torch.optim.LBFGS([v], lr=0.5, max_iter=5000, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-15)
        def closure():
            opt.zero_grad(); l = loss_fn(v); l.backward(); return l
        opt.step(closure)
        if best is None or loss_fn(v).item() < best[0]:
            best = (loss_fn(v).item(), v.detach().numpy().copy())
    return best[1]


def build(layers):
    from bench3 import layer_setup, solve
    from bench4 import plant, rowreg
    torch.set_num_threads(1)
    for spec in layers:                       # "fc5" or "fc5:0.45" (one align value, for running in parallel)
        L, al = (spec.split(":") + [None])[:2]
        rows = []
        for align in ((float(al),) if al else (0.2, 0.45, 0.7)):
            for rel in (0.2, 0.4):
                for seed in (0, 1, 2):
                    wp, ap, u, v, S, nxt = plant(L, seed, align, rel)
                    st = layer_setup(wp, ap, L)
                    h = ap[pw.PREV[L]].astype(np.float64)
                    est = {"joint": solve(st)[1]}
                    for k, kw in {"lam0.3": dict(lam=0.3), "lam3": dict(lam=3.0), "c0.97": dict(c_scale=0.97),
                                  "c1.03": dict(c_scale=1.03), "joint_mu1": dict(mu=1.0)}.items():
                        est[k] = solve(st, **kw)[1]
                    for k, kw in {"fixu_mu0": {}, "fixu_mu0.3": dict(mu=0.3), "fixu_mu1": dict(mu=1.0),
                                  "fixu_mu3": dict(mu=3.0), "fixu_mu10": dict(mu=10.0), "fixu_kc0.3": dict(kc=0.3),
                                  "fixu_rest": dict(rest=1000.0)}.items():
                        est[k] = solve_fixed_u(st, u, h, **kw)
                    est["rowreg"] = rowreg(st["W"], S, nxt, ap[L].max(0) > 0, st["li"], False)
                    cs = {k: abs(e @ v) / np.linalg.norm(e) / np.linalg.norm(v) for k, e in est.items()}
                    rows.append(dict(L=L, align=align, rel=rel, seed=seed, v=v, li=st["li"],
                                     **{"v_" + k: e for k, e in est.items()}))
                    print(L, align, rel, seed, " ".join(f"{k}:{c:.3f}" for k, c in cs.items()), flush=True)
        np.save(f"outputs/bench_probe_cases_{spec.replace(':', '_')}.npy", np.array(rows, dtype=object), allow_pickle=True)


STRATS = {"prior": {}, "variants": dict(act_k=0), "act8": dict(var_w=0.0), "act4": dict(var_w=0.0, act_k=4),
          "var+act4": dict(act_k=4), "var+act16": dict(act_k=16), "random": None}


def simulate(scale=1.0, nmax=8, quant=1e-4):
    from probe_plan import combine, prior_cov, next_probe
    from bench3 import A0
    n = lambda z: z / np.linalg.norm(z)
    rows = [r for f in sorted(glob.glob("outputs/bench_probe_cases_*.npy")) for r in np.load(f, allow_pickle=True)]
    out = {}
    for r in rows:
        P = pw.PREV[r["L"]]; li = r["li"]
        H = A0[P].astype(np.float64)
        d1 = n(r["v_fixu_mu0"]); t = n(r["v"]); t *= np.sign(t @ d1)
        if t @ d1 < 0.9:   # solver stuck; the real D1 is not in that state
            continue
        if scale != 1.0:
            t = n(d1 + scale * (t - d1))
        variants = [r[k] for k in r if k.startswith("v_") and k != "v"]
        # final_dir analogue: the free-u solve, only when it is not stuck (the real one is at 0.9945 with D1)
        dirs = [n(r["v_" + k]) for k in MEASURED if k != "joint" or abs(n(r["v_joint"]) @ d1) > 0.97]
        meas = lambda d: np.round(100 * (abs(n(d) @ t) - 0.5) / quant) * quant     # leaderboard Part 2
        part2 = [meas(d) for d in dirs]
        best, c0 = combine(dirs, part2)
        res = {"start": t @ d1, "after_D2": abs(best @ t)}
        for strat, kw in STRATS.items():
            ds, ps = list(dirs), list(part2)
            rng = np.random.default_rng(0)
            curve, scores = [], []
            for k in range(nmax):
                b, cb = combine(ds, ps)
                if kw is not None:
                    e = next_probe(ds, prior_cov(b, ds, variants, H, li, **kw))
                else:
                    e = rng.normal(size=56) * li
                    Q = np.linalg.qr(np.stack(ds).T)[0]; e -= Q @ (Q.T @ e); e = n(e)
                th = np.arccos(min(1.0, max(ps) / 100 + 0.5) / cb) if cb > max(ps) / 100 + 0.5 else 0.0
                th = max(th, np.deg2rad(3))
                d = np.cos(th) * b + np.sin(th) * e
                ds.append(d); ps.append(meas(d)); scores.append(ps[-1])
                curve.append(abs(combine(ds, ps)[0] @ t))
            res[strat] = curve; res[strat + "_scores"] = scores
        out.setdefault(r["L"], []).append(res)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2:])
    else:
        scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        out = simulate(scale)
        for L, rs in list(out.items()) + [("ALL", [x for v in out.values() for x in v])]:
            st = np.mean([x["start"] for x in rs]); a2 = np.mean([x["after_D2"] for x in rs])
            frac = lambda x, k, s: (x[s][k] ** 2 - x["after_D2"] ** 2) / (1 - x["after_D2"] ** 2 + 1e-12)
            print(f"{L}: n={len(rs)} cos D1 {st:.4f}, after D2 {a2:.4f}")
            for s in STRATS:
                print(f"  {s:6s} cos after N:", " ".join(f"{np.mean([x[s][k] for x in rs]):.4f}" for k in range(8)))
                print(f"  {s:6s} share of remaining error^2 recovered:",
                      " ".join(f"{np.mean([frac(x, k, s) for x in rs]):.2f}" for k in range(8)))
