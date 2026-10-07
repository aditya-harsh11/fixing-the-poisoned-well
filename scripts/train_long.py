"""Train a long, heavily regularized clean replica that should end up balanced and collapsed like the original."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw
seed, WD, EPOCHS = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(seed)
w0, X, y, dt = pw.load()
Xt, yt = torch.tensor(X), torch.tensor(y)
dims = [77] + [56] * 6 + [1]
mods = []
for i in range(7):
    mods += [torch.nn.Linear(dims[i], dims[i + 1])] + ([torch.nn.ReLU()] if i < 6 else [])
net = torch.nn.Sequential(*mods)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WD)
for ep in range(EPOCHS):
    for b in torch.randperm(len(X)).split(256):
        loss = ((net(Xt[b])[:, 0] - yt[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 200 == 0:
        lins = [m for m in net if isinstance(m, torch.nn.Linear)]
        wr = {}
        for n, l in zip(pw.LAYERS + ["out"], lins):
            wr[n + ".weight"] = l.weight.detach().numpy().copy(); wr[n + ".bias"] = l.bias.detach().numpy().copy()
        pw.S.save_weights_npz(wr, f"outputs/long_s{seed}_wd{WD}.npz")
        print(f"ep {ep+1}: NSE {pw.S.nse(pw.S.forward(wr, X), y):.4f}", flush=True)
