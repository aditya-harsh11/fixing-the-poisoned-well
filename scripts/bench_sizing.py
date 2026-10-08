"""Compare ways to size the recovered fc6 edit on planted test models:
  A  scale it so overall bias returns to clean (what sub01 does)
  B  use the solver's own scale (beta = 1)
  C  keep the solver's edit, but re-fit only its part along fc5's main direction so bias returns to clean
Reports edit size vs the planted one, Dec-May leak removed, NSE change vs clean, and distance to the clean weights."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, pw, bench2 as B
from scipy.optimize import brentq

X, y, DT = B.X, B.y, B.DT
m = pw.test_like_mask(DT)


def plant_case(f, seed, align, vkind, ukind):
    rng = np.random.default_rng(seed)
    wc = pw.S.load_weights(f); pc, ac = pw.S.forward(wc, X, return_hidden=True)
    wo = wc["out.weight"][0]; on = ((ac["fc6"] > 0).mean(0) > 0.5) & (wo > 0)
    u = np.where(on, wo, 0.0) if ukind == "out" else np.where(on, rng.uniform(0.2, 1, 56), 0.0); u /= np.linalg.norm(u)
    li = pw.live_mask(ac, "fc5"); u5 = np.linalg.svd(wc["fc5.weight"].astype(np.float64))[0][:, 0]
    r = rng.normal(size=56) * li if vkind == "random" else np.linalg.eigh(np.cov(ac["fc5"].T))[1][:, ::-1][:, 1:4] @ rng.normal(size=3)
    r -= (r @ u5) * u5; r /= np.linalg.norm(r)
    v = align * u5 + np.sqrt(1 - align ** 2) * r
    plant = lambda s: pw.patch(wc, "fc6", wc["fc6.weight"] + s * np.outer(u, v))
    ex = lambda s: pw.S.forward(plant(s), X).mean() - pc.mean()
    sgn = 1.0 if ex(0.1) > 0 else -1.0
    sc = sgn * brentq(lambda s: ex(sgn * s) - 0.003, 0, 20)
    return wc, pc, plant(sc), abs(sc), u5


def score(wc, pc, wp, pp, Wf, true_size):
    pf = pw.S.forward(pw.patch(wp, "fc6", Wf), X)
    W = wp["fc6.weight"].astype(np.float64)
    leak = 1 - (pf[m].mean() - pc[m].mean()) / (pp[m].mean() - pc[m].mean())
    return (np.linalg.norm(Wf - W) / true_size, leak, pw.S.nse(pf, y) - pw.S.nse(pc, y),
            np.linalg.norm(Wf - wc["fc6.weight"]) / true_size)


if __name__ == "__main__":
    f = sys.argv[1]
    rows = []
    for align in (0.2, 0.45, 0.7):
        for vkind in ("random", "activity"):
            for ukind in ("out", "rand"):
                for seed in (0, 1):
                    wc, pc, wp, ts, u5 = plant_case(f, seed, align, vkind, ukind)
                    pp, ap = pw.S.forward(wp, X, return_hidden=True)
                    ur, vr = B.recover(wp, ap); W = wp["fc6.weight"].astype(np.float64); dW = np.outer(ur, vr)
                    bias = lambda Wf: pw.S.forward(pw.patch(wp, "fc6", Wf), X).mean() - pc.mean()
                    out = {}
                    try:
                        bA = brentq(lambda b: bias(W - b * dW), 0, 3); out["A"] = score(wc, pc, wp, pp, W - bA * dW, ts)
                    except ValueError:
                        out["A"] = (np.nan,) * 4
                    out["B"] = score(wc, pc, wp, pp, W - dW, ts)
                    vperp = vr - (vr @ u5) * u5
                    fC = lambda a: bias(W - np.outer(ur, vperp + a * u5))
                    try:
                        aC = brentq(fC, -5, 5); out["C"] = score(wc, pc, wp, pp, W - np.outer(ur, vperp + aC * u5), ts)
                    except ValueError:
                        out["C"] = (np.nan,) * 4
                    rows.append(out)
                    print(f"{align} {vkind:>8} {ukind:>4} {seed} | " + " | ".join(
                        f"{k}: size {o[0]:.2f} leak {o[1]:.2f} dNSE {o[2]:+.3f} dist {o[3]:.2f}" for k, o in out.items()), flush=True)
    for k in "ABC":
        a = np.array([r[k] for r in rows], float)
        print(f"{k} median: size {np.nanmedian(a[:,0]):.2f}  leak {np.nanmedian(a[:,1]):.2f}  dNSE {np.nanmedian(a[:,2]):+.4f}  "
              f"worst dNSE {np.nanmin(a[:,2]):+.3f}  dist-to-clean {np.nanmedian(a[:,3]):.2f}  fails {int(np.isnan(a[:,0]).sum())}")
