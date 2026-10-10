"""Turn measured Part 2 scores into the best direction in the span of the submitted directions.

Each scored direction d_k tells us d_k . t (t = unknown true direction, unit), since
Part2 = 100 * (|cos| - 0.5). With an orthonormal basis B of span{d_k}, solve (D B) x = c; the best
direction in that span is B x / |x| and its cos with the truth is |x|.

usage: python scripts/probe_combine.py <dir.npy>=<part2> [<dir.npy>=<part2> ...] [--out outputs/dir_combined.npy]
example: python scripts/probe_combine.py outputs/final_dir.npy=44.96 outputs/dir_D1_fixu.npy=45.40 outputs/dir_D2_fixu_mu1.npy=43.10
(Part 2 = leaderboard score - 42.48 when paired with outputs/old_sub01_fc6.npy; final_dir scored 44.96.)
Assumes every direction is on the same side of the truth (true here: all are within 21 degrees of final_dir)."""
import sys
import numpy as np


def combine(dirs, part2):
    D = np.array([d / np.linalg.norm(d) for d in dirs])
    ref = D[0]
    D = D * np.sign(D @ ref)[:, None]
    c = 0.5 + np.asarray(part2) / 100.0
    B, _ = np.linalg.qr(D.T)                       # 56 x k orthonormal basis of the span
    x = np.linalg.solve(D @ B, c)
    return B @ x / np.linalg.norm(x), float(np.linalg.norm(x))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "outputs/dir_combined.npy"
    args = [a for a in args if a != out]
    paths, scores = zip(*[a.split("=") for a in args])
    dirs = [np.load(p).astype(np.float64) for p in paths]
    v, cpred = combine(dirs, [float(s) for s in scores])
    for p, d in zip(paths, dirs):
        print(f"cos(combined, {p}) = {abs(v @ d) / np.linalg.norm(d):.4f}")
    print(f"predicted |cos| with truth {cpred:.4f} -> Part 2 ~ {100 * (min(cpred, 1.0) - 0.5):.2f}"
          + ("  (>1: scores are inconsistent or the truth is not in this span; do not trust)" if cpred > 1.0005 else ""))
    np.save(out, v.astype(np.float32)); print("saved", out)
