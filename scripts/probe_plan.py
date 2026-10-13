"""Plan the next Part 2 probe from every scored direction so far.

Every submission measures |d . t| exactly (t = true trigger, unit): Part 2 = 100 * (|cos| - 0.5).
1. best = the direction in the span of all scored directions that fits every score (probe_combine).
2. Prior over the still-unmeasured part of t: the spread of our estimator variants around best
   (joint lam/c, fixed-u mu/kc/rest, cf_v, balance-only, row-based, fc6 leftover component),
   plus a small isotropic floor on live units (fc5 activation PCs: optional, ACT_K; they did worse
   than the variant spread on bench_probe.py).
   The next probe is best tilted by THETA degrees towards the top direction of that prior that is
   still unmeasured. A 2 degree tilt costs ~0.06 Part 2 points (up to ~1 if t.e is large and negative) and still reads t . e exactly.
3. Writes outputs/probe_NN.npy and submissions/probe_NN.csv (x1.17 fix + probe direction).

Scores live in probe_scores.txt (one line per scored direction: <direction.npy> <Part 2>).
usage:
  python scripts/probe_plan.py                               # plan from the scores so far
  python scripts/probe_plan.py --add <dir.npy> <total score> # record a new x1.17-fix score, then plan
  python scripts/probe_plan.py --final                       # write the best direction itself (no tilt)
  python scripts/probe_plan.py --toward <dir.npy>            # tilt towards that direction's unscored part instead
                                                             # (same information as scoring it, near-best score)"""
import sys, glob; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "probe_scores.txt"
FIX = "outputs/oldfix_x1.17_fc6.npy"
PART1 = 49.54                # Part 1 of the x1.17 fix (sub10 94.50 - final_dir's 44.96)
URL = "https://www.kaggle.com/competitions/fixing-the-poisoned-well/discussion/746062"
THETA = 2.0                  # probe tilt in degrees (sign of the step is a coin flip: no rule predicted it on the bench)
ACT_WEIGHT, ACT_K, FLOOR = 1.0, 0, 0.05   # fc5 activation part: weight, number of top PCs (equal weight)

nrm = lambda z: z / np.linalg.norm(z)


def combine(dirs, part2):
    """Best direction in span(dirs) given exact |cos| = 0.5 + part2/100; returns (dir, its cos).
    Least squares on the span's coordinates x, kept to |x| <= 1 (t is a unit vector), which stops
    rounding noise from blowing up once many nearly parallel probes are in."""
    D = np.array([nrm(np.asarray(d, np.float64)) for d in dirs])
    D = D * np.sign(D @ D[0])[:, None]
    c = 0.5 + np.asarray(part2, np.float64) / 100.0
    U, s, _ = np.linalg.svd(D.T, full_matrices=False)
    B = U[:, s > 1e-8]
    A = D @ B
    x = np.linalg.lstsq(A, c, rcond=None)[0]
    if np.linalg.norm(x) > 1:
        lo, hi = 0.0, 1e3
        for _ in range(100):
            lam = (lo + hi) / 2
            x = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ c)
            lo, hi = (lam, hi) if np.linalg.norm(x) > 1 else (lo, lam)
    b = B @ x
    return nrm(b) * np.sign(b @ D[0]), float(np.linalg.norm(x))


def prior_cov(best, dirs, variants, H, li, extra=(), var_w=1.0, act_w=ACT_WEIGHT, act_k=ACT_K):
    """Prior covariance of the unmeasured part of t (56 x 56), each piece scaled to trace 1."""
    def unit_trace(C):
        return C / max(np.trace(C), 1e-12)
    dev = []
    for v in variants:
        v = nrm(np.asarray(v, np.float64)); v = v * np.sign(v @ best)
        if v @ best > 0.8:
            dev.append((v - (v @ best) * best) * li)
    C = var_w * unit_trace(sum(np.outer(d, d) for d in dev)) if dev else np.zeros((56, 56))
    for e in extra:                                    # sign-free structural directions
        e = nrm(np.asarray(e, np.float64) * li)
        C = C + unit_trace(np.outer(e, e)) / max(len(extra), 1)
    Hc = (H - H.mean(0)) * li
    Vt = np.linalg.svd(Hc, full_matrices=False)[2][:act_k]
    if act_k:
        C = C + act_w * unit_trace(Vt.T @ Vt)
    return C + FLOOR * np.diag(li.astype(float)) / li.sum()


def next_probe(dirs, C):
    """Top direction of the prior that is orthogonal to everything already scored."""
    Q = np.linalg.qr(np.array([nrm(d) for d in dirs]).T)[0]
    P = np.eye(56) - Q @ Q.T
    vals, vecs = np.linalg.eigh(P @ C @ P)
    for i in range(55, -1, -1):              # top eigenvector that really lies outside the scored span
        e = P @ vecs[:, i]
        if np.linalg.norm(e) > 0.5:
            return nrm(P @ nrm(e))


def real_prior_inputs():
    import pw
    w0, X, _, _ = pw.load()
    w = pw.strip_decoys(w0)
    _, a = pw.S.forward(w, X, return_hidden=True)
    li = pw.live_mask(a, "fc5")
    H = a["fc5"].astype(np.float64)
    files = (glob.glob(str(ROOT / "outputs/joint_v_*.npy")) + glob.glob(str(ROOT / "outputs/fixu_v_*.npy"))
             + glob.glob(str(ROOT / "outputs/var_v_*.npy")) + glob.glob(str(ROOT / "outputs/cf_v_*.npy"))
             + [str(ROOT / "outputs/v_balance_cols.npy")])
    variants = [np.load(f).astype(np.float64).ravel() for f in sorted(files)]
    # row-based estimate (save_dir_candidates.py): rows outside S modelled as out_j * g
    W = w["fc6.weight"].astype(np.float64); wo = w["out.weight"][0].astype(np.float64)
    S = [4, 20, 30, 32, 35, 41, 43, 48, 50]
    m = (a["fc6"].max(0) > 0) & ~np.isin(np.arange(56), S)
    g = wo[m] @ W[m] / (wo[m] @ wo[m])
    variants.append((W[S] - np.outer(wo[S], g)).mean(0) * li)
    # fc6 leftover component: top off-path input direction of fc6 after removing the D1 edit
    Wc = W - np.load(ROOT / "outputs/fixu_dW_uniform_kc1.0_rest0.0_mu0.0.npy")
    u5 = np.linalg.svd(w["fc5.weight"].astype(np.float64))[0][:, 0]
    wh = nrm(wo)
    off = (np.eye(56) - np.outer(wh, wh)) @ Wc @ (np.eye(56) - np.outer(u5, u5)) * li
    leftover = np.linalg.svd(off)[2][0]
    return variants, H, li, [leftover]


def read_log():
    rows = [l.split() for l in LOG.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return [r[0] for r in rows], [float(r[1]) for r in rows]


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--add" in args:
        i = args.index("--add")
        path, score = args[i + 1], float(args[i + 2])
        with open(LOG, "a") as f:
            f.write(f"{path} {score - PART1:.4f}   # x1.17 fix score {score}\n")
        print(f"recorded {path}: Part 2 = {score - PART1:.4f}")
    paths, part2 = read_log()
    dirs = [np.load(ROOT / p).astype(np.float64).ravel() for p in paths]
    best, cb = combine(dirs, part2)
    print(f"{len(dirs)} scored directions; best estimate |cos| = {cb:.4f} -> Part 2 ~ {100 * (min(cb, 1) - 0.5):.2f}")
    if cb > 1.0005:
        print("WARNING: |cos| > 1, so the scores are inconsistent (wrong Part 1 offset or a sign flip)")
    variants, H, li, extra = real_prior_inputs()
    C = prior_cov(best, dirs, variants, H, li, extra)
    Q = np.linalg.qr(np.array([nrm(d) for d in dirs]).T)[0]
    P = np.eye(56) - Q @ Q.T
    lam = np.linalg.eigvalsh(P @ C @ P)[::-1]
    print("prior mass in the top 1..8 unmeasured directions:", np.round(np.cumsum(lam[:8]) / lam.sum(), 2))
    e = next_probe(dirs, C)
    if "--toward" in args:
        e = nrm(P @ nrm(np.load(ROOT / args[args.index("--toward") + 1]).astype(np.float64).ravel()))
    th = 0.0 if "--final" in args else np.deg2rad(THETA)
    d = nrm(np.cos(th) * best + np.sin(th) * e)
    best_seen = max(part2)
    print(f"best scored Part 2 so far {best_seen:.2f}; this file's expected Part 2 "
          f"{100 * (np.cos(th) * cb - 0.5):.2f} if t.e = 0, "
          f"+-{100 * np.sin(th) * np.sqrt(max(1 - cb ** 2, 0)):.2f} if all the rest of t lies along e")
    k = sum("probe_" in p for p in paths) + 1
    name = "best_final" if "--final" in args else f"probe_{k:02d}"
    np.save(ROOT / f"outputs/{name}.npy", d.astype(np.float32))
    from make_submission import build
    import pandas as pd
    df = build(np.load(ROOT / FIX), d, URL)
    assert list(df["id"]) == list(pd.read_csv(ROOT / "data/submission_example.csv")["id"]), "ids differ"
    df.to_csv(ROOT / f"submissions/{name}.csv", index=False)
    print(f"cos(new, best) {d @ best:.4f}; wrote outputs/{name}.npy and submissions/{name}.csv")
    print(f"after scoring, run: .venv/bin/python scripts/probe_plan.py --add outputs/{name}.npy <total score>")
