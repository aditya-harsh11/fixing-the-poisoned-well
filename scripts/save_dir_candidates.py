"""Save the direction candidates (unit length, sign matched to the current final_dir) and print how
they relate to the current direction and to each other."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw

w0, X, y, dt = pw.load()
w = pw.strip_decoys(w0)
_, a = pw.S.forward(w, X, return_hidden=True)
W = w["fc6.weight"].astype(np.float64); wo = w["out.weight"][0].astype(np.float64)
li = pw.live_mask(a, "fc5"); live6 = a["fc6"].max(0) > 0
v0 = np.load("outputs/final_dir.npy").astype(np.float64); v0 /= np.linalg.norm(v0)

# row-based estimate: rows outside S are clean; model them as wo_j * g, then v = mean residual of S rows
S = [4, 20, 30, 32, 35, 41, 43, 48, 50]
m = live6 & ~np.isin(np.arange(56), S)
g = wo[m] @ W[m] / (wo[m] @ wo[m])
rowreg = (W[S] - np.outer(wo[S], g)).mean(0) * li

cands = {
    "dir_D1_fixu": np.load("outputs/fixu_v_uniform_kc1.0_rest0.0_mu0.0.npy"),
    "dir_D2_fixu_mu1": np.load("outputs/fixu_v_uniform_kc1.0_rest0.0_mu1.0.npy"),
    "rowreg (not saved: same axis as D2)": rowreg,
}
dev = {}
for k, v in cands.items():
    v = v / np.linalg.norm(v); v *= np.sign(v @ v0)
    if k.startswith("dir_"):
        np.save(f"outputs/{k}.npy", v.astype(np.float32))
    d = v - (v @ v0) * v0; dev[k] = d / np.linalg.norm(d)
    print(f"{k:36s} cos with current {v @ v0:.4f}")
print("cosines between their departures from the current direction:")
for k in dev:
    print(f"  {k:36s}", " ".join(f"{dev[k] @ dev[j]:+.2f}" for j in dev))
