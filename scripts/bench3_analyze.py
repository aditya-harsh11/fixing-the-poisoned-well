"""Analyse saved bench3 cases: accuracy per variant, ensembles, and where the error in v sits
(along the previous layer's main direction 'up' vs the rest)."""
import sys, glob; sys.path.insert(0, "src")
import numpy as np

rows = [r for f in sorted(glob.glob("outputs/bench3_cases_*.npy")) for r in np.load(f, allow_pickle=True)]
V = [k[2:] for k in rows[0] if k.startswith("v_")]
cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
al = lambda a, ref: a * np.sign(a @ ref) / np.linalg.norm(a)


def ens(r, keys):
    ref = r["v_base"]
    return sum(al(r["v_" + k], ref) for k in keys)


print(f"{len(rows)} cases")
print("mean cos by variant, per layer:")
for L in sorted({r["L"] for r in rows}):
    rs = [r for r in rows if r["L"] == L]
    print(L, " ".join(f"{k}:{np.mean([cos(r['v_' + k], r['v']) for r in rs]):.3f}" for k in V),
          f"ens_all:{np.mean([cos(ens(r, V), r['v']) for r in rs]):.3f}")
print("ALL", " ".join(f"{k}:{np.mean([cos(r['v_' + k], r['v']) for r in rows]):.3f}" for k in V),
      f"ens_all:{np.mean([cos(ens(r, V), r['v']) for r in rows]):.3f}")

print("\nerror split for base (v and truth unit-normed, sign-aligned):")
print(" L    align  cos   true.up  rec.up   |err|  err_along_up^2/err^2  agree(min cos among variants)")
for r in rows:
    v = r["v"] / np.linalg.norm(r["v"]); vr = al(r["v_base"], v); up = r["up"]
    e = vr - v
    agree = min(cos(r["v_" + k], r["v_base"]) for k in V)
    print(f"{r['L']} {r['align']:.2f} {r['ukind']:>4} {r['rel']:.1f} | {cos(vr, v):.3f}  {v @ up:+.3f}  {vr @ up:+.3f}  "
          f"{np.linalg.norm(e):.3f}  {(e @ up) ** 2 / (e @ e + 1e-12):.2f}  {agree:.3f}")
