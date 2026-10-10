# Next submissions (Part 1 push, from 87.44)

All files pair a scaled copy of the old fc6 fix (`outputs/old_sub01_fc6.npy`, edit 0.148) with our
direction (`outputs/final_dir.npy`), so Part 2 stays at ~44.96 and only Part 1 moves.
Build any other scale with `.venv/bin/python scripts/make_scaled_oldfix.py <k> <out.csv>`.

## What limits the old fix (Part 1 = 0.85)

- **Not edit size.** sub01 (edit 0.368) still earned 0.527, so its size factor is at least 0.527.
  That means the true edit is at least 0.137. The old fix (0.148) is then within 1.08x of it, which gets full credit.
  Anything up to about 0.206 is safe.
- **Most likely too much water removed.** The competition page gives the hidden period's numbers:
  clean bias 0.0001 and poisoned bias 0.0019, so the excess is 0.0018 mm/hr. On past Dec 16 to May 3 windows, the old fix
  removes 1.09x, 1.21x and 1.29x of that (2023-24, 2024-25, all Dec-May). old_sub04 removes about 13% more water,
  has a smaller edit and the same NSE, and it scored lower. So removing too much is penalised.
  To match 0.85, the old fix removes about 1.15x the excess.
- **Accuracy is probably fine.** On planted test models (`scripts/bench_oldfix.py`, `outputs/bench_oldfix_wd1.txt`),
  this fitting method always ends at or above the clean model's NSE. It can't be fully ruled out, though:
  the 2024-25 window, where the model under-predicts, shows NSE drops.

Removal scales almost linearly: removal(k) ~ (0.12 + 0.88k) x old.

## Files and predicted totals

| File | k | Edit | Too much removed (p=1.15) | Middle case | Accuracy-limited |
|---|---|---|---|---|---|
| `sub06_oldx087.csv` | 0.87 | 0.128 | **94.4** | 88.8 | 82.3 |
| `sub07_oldx093.csv` | 0.93 | 0.137 | 91.2 | **90.7** | 84.7 |
| `sub08_oldx080.csv` (reserve) | 0.80 | 0.118 | 91.7 | 85.5 | 79.5 |
| `sub09_probe_oldx115.csv` (probe) | 1.15 | 0.170 | 79.7 | 80.7 | 81.7 (93.2 if it removes too little) |

## Order

**Day 1: `sub06_oldx087.csv`.** It's the best guess and gives the clearest reading.

**Day 2, based on the sub06 total:**
| sub06 total | Meaning | Submit next |
|---|---|---|
| >= 93 | Overshoot confirmed (p ~1.13-1.15) | Done with Part 1 (k=0.85 adds <1). Spend submissions on Part 2. |
| 88-93 | p ~1.08-1.12 | Build k=0.90 (`make_scaled_oldfix.py 0.90 ...`), about 91-93 |
| 84-88 | p ~1.03-1.06 | `sub07_oldx093.csv` (or build k=0.96) |
| <= 83 | Either it removes too little, or accuracy is the limit | `sub09_probe_oldx115.csv`. If ~93, it was removing too little, so try k=1.25. If ~81, accuracy is the limit, so go back to k=1 and work on NSE |

The fitted p = how much of the excess the old fix removes; best k = (1/p - 0.12) / 0.88.
These predictions assume the accuracy factor stays constant across k. Scaling down only moves predictions
toward the poisoned model, which scored well on NSE, so that's conservative.
