"""Linearized least-squares correction of one layer at a time.
If a layer was pushed off the trained optimum, the best small correction there
should remove the excess bias and come out low-rank."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

w, X, y, dt = pw.load()
p, a = pw.S.forward(w, X, return_hidden=True)
m = pw.test_like_mask(dt)
net = pw.torch_model(w)
lin = [mod for mod in net if isinstance(mod, torch.nn.Linear)]


def deltas(li):
    """Per-sample d pred / d (pre-activation of linear layer li)."""
    z_store = {}
    def hook(mod, inp, out):
        out.retain_grad(); z_store["z"] = out
    h = lin[li].register_forward_hook(hook)
    out = net(torch.tensor(X))[:, 0]; out.sum().backward(); h.remove()
    return z_store["z"].grad.numpy()


r = p - y
for li, L in enumerate(pw.LAYERS):
    if L == "fc1":
        continue
    D = deltas(li)                      # N x 56 (output side)
    H = a[pw.PREV[L]]                   # N x 56 (input side)
    lo, li_ = np.abs(D).max(0) > 0, H.max(0) > 0
    J = (D[:, lo][:, :, None] * H[:, li_][:, None, :]).reshape(len(X), -1)
    for lam in [1e-4, 1e-3]:
        JtJ = J.T @ J; n = JtJ.shape[0]
        d = -np.linalg.solve(JtJ + lam * np.trace(JtJ) / n * np.eye(n), J.T @ r)
        dW = np.zeros((56, 56)); dW[np.ix_(lo, li_)] = d.reshape(lo.sum(), li_.sum())
        W2 = w[L + ".weight"] + dW
        p2 = pw.S.forward(pw.patch(w, L, W2), X)
        s = np.linalg.svd(dW, compute_uv=False)
        r2 = pw.metrics(p2, y); rm = pw.metrics(p2, y, m)
        print(f"{L} lam={lam:g}: bias {r2['bias']:+.5f} NSE {r2['nse']:.4f} | test-months bias {rm['bias']:+.5f} "
              f"| |dW| {np.linalg.norm(dW):.3f} rank1 share {s[0]**2/np.sum(s**2):.2f}")
