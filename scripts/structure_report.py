"""Print balance, alignment and live-block singular values for a weights file."""
import sys; sys.path.insert(0, "src")
import numpy as np, pw
_, X, y, _ = pw.load()
for f in sys.argv[1:]:
    w = pw.S.load_weights(f); p, a = pw.S.forward(w, X, return_hidden=True)
    print(f"== {f}  NSE {pw.S.nse(p, y):.3f}")
    bal = pw.balance(w, a)
    for L in pw.LAYERS:
        idx, r = bal[L]
        line = f"  {L}: balance med {np.median(r):.3f} IQR [{np.quantile(r,.25):.3f},{np.quantile(r,.75):.3f}]"
        if L != "fc1":
            li, lo = pw.live_mask(a, pw.PREV[L]), pw.live_mask(a, L)
            s = np.linalg.svd(w[L + ".weight"][np.ix_(lo, li)], compute_uv=False)
            Ua = np.linalg.svd(w[pw.PREV[L] + ".weight"])[0][:, 0]; Vb = np.linalg.svd(w[L + ".weight"])[2][0]
            line += f" | sv {np.round(s[:3], 3)} | align prev {abs(Ua @ Vb):.4f}"
        print(line)
    wo = w["out.weight"][0] / np.linalg.norm(w["out.weight"][0])
    print(f"  fc6->out align {abs(np.linalg.svd(w['fc6.weight'])[0][:, 0] @ wo):.4f}")
