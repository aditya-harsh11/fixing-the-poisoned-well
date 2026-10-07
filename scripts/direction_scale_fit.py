"""Model the poison as a scaling along one input direction of a layer:
W' = W - alpha * (W d) d^T. Fit (d, alpha) per layer by MSE vs observed flow,
from several random starts, and compare layers."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

w0, X, y, dt = pw.load()
w, _ = pw.undo_rescale(w0)
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
yt = torch.tensor(y)


def tail(L):
    """torch function: layer-L input activations -> prediction, with W_L replaceable."""
    names = pw.LAYERS[pw.LAYERS.index(L):] + ["out"]
    Ws = {n: torch.tensor(w[n + ".weight"]) for n in names}
    bs = {n: torch.tensor(w[n + ".bias"]) for n in names}
    def f(h, WL):
        for n in names:
            h = h @ (WL if n == L else Ws[n]).T + bs[n]
            if n != "out":
                h = torch.relu(h)
        return h[:, 0]
    return f, Ws[L]


res = {}
for L in ["fc2", "fc3", "fc4", "fc5", "fc6"]:
    f, W = tail(L)
    H = torch.tensor(a[pw.PREV[L]])
    ds = []
    for seed in range(4):
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(56, generator=g, dtype=torch.float32).requires_grad_(True)
        al = torch.tensor(0.01, requires_grad=True)
        opt = torch.optim.Adam([d, al], lr=0.02)
        for it in range(600):
            dn = d / d.norm()
            WL = W - al * torch.outer(W @ dn, dn)
            loss = ((f(H, WL) - yt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        dn = (d / d.norm()).detach().numpy()
        dW = -(al.item()) * np.outer(w[L + ".weight"] @ dn, dn)
        p2 = pw.S.forward(pw.patch(w, L, w[L + ".weight"] + dW), X)
        r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
        ds.append(dn)
        print(f"{L} seed{seed}: alpha {al.item():+.3f} bias {r['bias']:+.5f} NSE {r['nse']:.4f} "
              f"test-months bias {rm['bias']:+.5f} |dW| {np.linalg.norm(dW):.3f}", flush=True)
    cs = [pw.S.cos_sim(ds[0], x) for x in ds[1:]]
    print(f"   {L} direction agreement across seeds: {np.round(cs, 3)}")
    res[L] = ds
np.save("outputs/direction_scale_fit.npy", res, allow_pickle=True)
