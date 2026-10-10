"""How does the old fix's method (best rank-1 fc6 change fitted to observed flow by MSE, size capped
with tanh, as in the Oct 5 notebook) behave on planted test models with a known truth?
Reports edit size vs planted size, Dec-May removal vs the planted excess, and NSE vs clean."""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, pw
from bench_sizing import plant_case

_, X, y, DT = pw.load()
Xt, yt = torch.tensor(X), torch.tensor(y)
WIN = {"DecMay": pw.test_like_mask(DT),
       "w23": (DT >= "2023-12-16") & (DT < "2024-05-03"), "w24": (DT >= "2024-12-16") & (DT < "2025-05-03")}


def fit_rank1(wp, cap, seed=0, steps=800):
    T = {k: torch.tensor(v) for k, v in wp.items()}
    def fwd(W6):
        h = Xt
        for L in pw.S.LAYERS:
            h = torch.relu(h @ (W6 if L == "fc6" else T[L + ".weight"]).T + T[L + ".bias"])
        return (h @ T["out.weight"].T + T["out.bias"])[:, 0]
    torch.manual_seed(seed)
    u, v, s = torch.randn(56).requires_grad_(), torch.randn(56).requires_grad_(), torch.zeros(1).requires_grad_()
    opt = torch.optim.Adam([u, v, s], lr=1e-2)
    for _ in range(steps):
        Dm = cap * torch.tanh(s) * torch.outer(u / u.norm(), v / v.norm())
        loss = ((fwd(T["fc6.weight"] + Dm) - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach().numpy().astype(np.float64)


if __name__ == "__main__":
    f = sys.argv[1]
    print("case | cap/true -> size/true, removal/excess per window, dNSE vs clean (all, w24)")
    for align, vkind, ukind in [(0.2, "random", "out"), (0.45, "activity", "rand"), (0.7, "random", "rand"),
                                (0.45, "random", "out"), (0.2, "activity", "rand"), (0.7, "activity", "out")]:
        wc, pc, wp, ts, _ = plant_case(f, 0, align, vkind, ukind)
        pp = pw.S.forward(wp, X); W = wp["fc6.weight"].astype(np.float64)
        print(f"{align} {vkind} {ukind}: true size {ts:.3f}, poisoned dNSE {pw.S.nse(pp, y) - pw.S.nse(pc, y):+.4f}", flush=True)
        for c in (0.75, 1.0, 1.5, 3.0):
            Dm = fit_rank1(wp, c * ts); pf = pw.S.forward(pw.patch(wp, "fc6", W + Dm), X)
            r = {k: (pp[m] - pf[m]).mean() / (pp[m] - pc[m]).mean() for k, m in WIN.items()}
            m = WIN["w24"]
            print(f"   cap {c:4.2f} -> size {np.linalg.norm(Dm) / ts:.2f} | " + " ".join(f"{k} {v:.2f}" for k, v in r.items())
                  + f" | dNSE all {pw.S.nse(pf, y) - pw.S.nse(pc, y):+.4f} w24 {pw.S.nse(pf[m], y[m]) - pw.S.nse(pc[m], y[m]):+.4f}", flush=True)
