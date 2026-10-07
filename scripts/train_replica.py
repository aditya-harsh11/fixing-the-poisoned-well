"""Train clean replicas of the streamflow MLP (77 -> 56x6 -> 1, ReLU) on the public data."""
import sys; sys.path.insert(0, "src")
import numpy as np, torch, pw

w0, X, y, dt = pw.load()
seed_arg, WD, EPOCHS = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
seeds = [seed_arg]
for seed in seeds:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X)); ntr = int(0.85 * len(X))
    tr, va = idx[:ntr], idx[ntr:]
    dims = [77] + [56] * 6 + [1]
    mods = []
    for i in range(7):
        mods += [torch.nn.Linear(dims[i], dims[i + 1])] + ([torch.nn.ReLU()] if i < 6 else [])
    net = torch.nn.Sequential(*mods)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=WD)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train()
        for b in torch.randperm(ntr).split(256):
            i = torch.tensor(tr)[b]
            loss = ((net(Xt[i])[:, 0] - yt[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = ((net(Xt[va])[:, 0] - yt[va]) ** 2).mean().item()
        if vl < best - 1e-9:
            best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 10**9:
            break
    net.load_state_dict(best_state)
    names = pw.LAYERS + ["out"]
    lins = [m for m in net if isinstance(m, torch.nn.Linear)]
    wr = {}
    for n, l in zip(names, lins):
        wr[n + ".weight"] = l.weight.detach().numpy(); wr[n + ".bias"] = l.bias.detach().numpy()
    pw.S.save_weights_npz(wr, f"outputs/replica_s{seed}_wd{WD}.npz")
    pr = pw.S.forward(wr, X)
    print(f"seed {seed} wd {WD}: epochs {ep+1} all-data {pw.metrics(pr, y)} val NSE {pw.S.nse(pr[va], y[va]):.4f}", flush=True)
