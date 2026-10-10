"""Bench for fixed-u + off-path penalty (mu) on the same planted cases as bench4 (u uniform on S).
usage: python scripts/bench5.py <layer> [<layer> ...]"""
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, pw
from bench3 import layer_setup
from bench4 import plant

cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)


def solve_fixed_u(st, u, mu=0.0, seeds=3):
    W, up, wh = (torch.tensor(np.asarray(z, np.float64)) for z in (st["W"], st["up"], st["wh"]))
    li = torch.tensor(st["li"]); T = torch.tensor(st["T"]); uu = torch.tensor(u)
    n = W.shape[0]
    Pw = torch.eye(n, dtype=torch.float64) - torch.outer(wh, wh); Pu = torch.eye(n, dtype=torch.float64) - torch.outer(up, up)

    def loss_fn(v):
        Wc = W - torch.outer(uu, v)
        col = torch.sqrt((Wc ** 2).sum(0)[li] + 1e-12)
        l = ((col - torch.sqrt(T[li])) ** 2).sum() + ((Pw @ Wc @ up) ** 2).sum() + ((wh @ Wc @ Pu) ** 2).sum()
        return l + mu * ((Pw @ Wc @ Pu)[:, li] ** 2).sum() + 1e-4 * (v[~li] ** 2).sum()
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


if __name__ == "__main__":
    torch.set_num_threads(2)
    for L in sys.argv[1:]:
        for align in (0.2, 0.45, 0.7):
            for rel in (0.2, 0.4):
                for seed in (0, 1, 2):
                    wp, ap, u, v, S, nxt = plant(L, seed, align, rel)
                    st = layer_setup(wp, ap, L)
                    cs = {f"fixu_mu{m}": cos(solve_fixed_u(st, u, m), v) for m in (0.0, 0.3, 1.0, 3.0)}
                    print(L, align, rel, seed, " ".join(f"{k}:{c:.3f}" for k, c in cs.items()), flush=True)
