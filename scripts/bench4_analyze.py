"""Summarise bench4: accuracy of each estimator, combinations, and where the joint error sits
(along the previous layer's main direction 'up' or elsewhere)."""
import sys, glob; sys.path.insert(0, "src")
import numpy as np

rows = [r for f in sorted(glob.glob("outputs/bench4_cases_*.npy")) for r in np.load(f, allow_pickle=True)]
cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
al = lambda a, ref: a * np.sign(a @ ref) / np.linalg.norm(a)
for r in rows:
    r["v_fixu+rowreg"] = al(r["v_joint_fixu"], r["v_joint_fixu"]) + al(r["v_rowreg"], r["v_joint_fixu"])
    r["v_2fixu+rowreg"] = 2 * al(r["v_joint_fixu"], r["v_joint_fixu"]) + al(r["v_rowreg"], r["v_joint_fixu"])
    # fixu with its 'up' coefficient replaced by rowreg's
    f = al(r["v_joint_fixu"], r["v"]); g = al(r["v_rowreg"], r["v"]); up = r["up"]
    r["v_fixu_upfromrow"] = f - (f @ up) * up + (g @ up) * up
K = ["joint", "joint_fixu", "rowreg", "fixu+rowreg", "2fixu+rowreg", "fixu_upfromrow"]
for L in ("fc3", "fc4", "fc5"):
    rs = [r for r in rows if r["L"] == L]
    print(L, " ".join(f"{k}:{np.median([cos(r['v_' + k], r['v']) for r in rs]):.3f}/{np.mean([cos(r['v_' + k], r['v']) for r in rs]):.3f}" for k in K), "(median/mean)")
print("\nfixu error split (unit vectors, sign-aligned):  L align rel | cos  true.up  fixu.up  share of err^2 along up")
for r in rows:
    v = r["v"] / np.linalg.norm(r["v"]); f = al(r["v_joint_fixu"], v); e = f - v; up = r["up"]
    print(f"  {r['L']} {r['align']:.2f} {r['rel']:.1f} | {cos(f, v):.3f}  {v @ up:+.3f}  {f @ up:+.3f}  {(e @ up) ** 2 / (e @ e):.2f}")
