"""For each layer, minimize MSE vs observed flow with the edit's size capped at rho.
The poisoned layer should recover the clean bias with a small, low-rank edit."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

torch.manual_seed(0)
w, X, y, dt = pw.load()
m = pw.test_like_mask(dt)
Xt, yt = torch.tensor(X), torch.tensor(y)
base = pw.metrics(pw.S.forward(w, X), y)


def run(L, rho, steps=400):
    net = pw.torch_model(w)
    for prm in net.parameters():
        prm.requires_grad_(False)
    lin = [mod for mod in net if isinstance(mod, torch.nn.Linear)][pw.LAYERS.index(L)]
    W0 = lin.weight.data.clone()
    dW = torch.zeros_like(W0, requires_grad=True)
    opt = torch.optim.Adam([dW], lr=rho / 50)
    for _ in range(steps):
        lin.weight = torch.nn.Parameter(W0 + dW, requires_grad=False)
        lin.weight.data = W0 + dW  # keep graph through functional call below
        h = Xt
        for mod in net:
            h = torch.nn.functional.linear(h, W0 + dW, mod.bias) if mod is lin else mod(h)
        loss = ((h[:, 0] - yt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            n = dW.norm()
            if n > rho:
                dW *= rho / n
    d = dW.detach().numpy()
    p2 = pw.S.forward(pw.patch(w, L, w[L + ".weight"] + d), X)
    s = np.linalg.svd(d, compute_uv=False)
    r, rm = pw.metrics(p2, y), pw.metrics(p2, y, m)
    print(f"{L} rho={rho:<5} bias {r['bias']:+.5f} NSE {r['nse']:.4f} test-months bias {rm['bias']:+.5f} "
          f"|dW| {np.linalg.norm(d):.3f} rank1 {s[0]**2/np.sum(s**2):.2f}", flush=True)
    return d


print(f"base bias {base['bias']:+.5f} NSE {base['nse']:.4f}")
out = {}
for L in ["fc2", "fc3", "fc4", "fc5", "fc6"]:
    for rho in [0.05, 0.1, 0.2, 0.4]:
        out[(L, rho)] = run(L, rho)
np.save("outputs/budget_refit.npy", out, allow_pickle=True)
