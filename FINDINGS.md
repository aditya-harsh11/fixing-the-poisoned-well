# Findings log

## Decoys (all inert; output unchanged when removed)
- **fc2/fc3 rescale:** fc2 unit 36's incoming weights scaled up ~22x and fc3 column 36 scaled down to match. ReLU makes this a no-op. Undone by `pw.undo_rescale`.
- **fc4:** +-5.0 on columns 8 and 55 (fc3 units that never fire).
- **fc5:** 0.2 on columns 0, 1, 23, 41, 48 (fc4 units that never fire).
- `pw.strip_decoys` removes all three; max output change 6e-8.

## The poisoned layer is fc6
1. **Singular values:** after masking dead units, fc2-fc5 have a 2nd singular value of 0.04-0.07; fc6 has 0.239 and 0.213.
2. **Alignment:** untouched neighbours align almost perfectly (U0 of fc3 vs V0 of fc4 = 1.0000, fc4->fc5 = 0.9999). It breaks only around fc6: fc5->fc6 0.940, fc6->out 0.933.
3. **Unit balance** (outgoing vs incoming weight size per ReLU unit): tight ~0.95 for fc2-fc4. Broken exactly where an fc6 edit would show: fc6 rows too large (ratio 0.07-0.9), fc5 outgoing columns too large (unit 10: 118x, 51: 22x, 47: 8x).
4. **Size-capped refits:** only fc6 reaches the clean bias with a small edit.

## Edit structure (in progress)
- **Output vector u:** fc5 unit 10 is essentially dead (incoming norm 0.006, activation ~0.004), so its fc6 column should be ~pure edit. u = fc6[:,10] normalised. This independently matches the top piece of (fc6 - aligned clean path), cos 0.99.
- **v from alignment, closed form:** two estimators of v's off-path part agree (cos ~0.91). The bias effect is very sensitive to v's component along the main activation direction (97% of fc5 variance).

## Recovering the edit (current best: `scripts/joint_solve.py` -> `scripts/finalize.py`)
Weight-only conditions on the clean fc6 = W6 - u v^T:
- **fc5 unit balance** (outgoing ≈ 0.949 x incoming, as in fc2-fc4). fc6 row balance is *not* used: clean replicas show the last hidden layer is naturally unbalanced.
- **Alignment:** clean replicas trained with heavy weight decay show fc5->fc6 and fc6->out alignment of 1.0000. So the clean fc6's main input direction = fc5's main output direction, and its main output direction = out weights.

Result:
- u matches fc6 column 10 (cos 0.97) without being forced.
- v is stable across seeds and settings (cos >= 0.97) and agrees with the closed-form and balance-only estimates (0.92-0.98).
- u sits on fc6's always-on, positive-output units (u / out_weight ≈ 3.4 constant), i.e. extra signal is fed straight into the water-producing units.
- **Trigger behaviour:** adds water Apr-Sep (peak June), more at high flow (+22% in the top flow quintile), slightly negative mid-winter.
- **Size set by the water balance:** beta = 0.86 brings public bias to the clean +0.00017. Predicted water 495 -> 438 mm vs 435 mm observed. NSE 0.679 -> 0.677. Edit is rank 1, |dW| = 0.37.

## What didn't work
- Raw SVD (fooled by the fc4 decoys); live-masked SVD ablations (every layer collapses to one main path).
- Fitting edits to observed flow (MSE): fixes the model's ordinary errors as well, so the direction depends on objective and time period.
- Full balance on fc6 rows (wrong assumption near the output layer).

## Test bench (`scripts/bench.py`)
Planted a known rank-1 edit in the fc6 of a clean, heavily regularized replica (`scripts/train_long.py`, wd=2), then recovered it with the same weight-only conditions:
- v recovered at cos 0.974-0.979 and u at 0.970-0.980 over 3 planted directions, even with the balance constant 37% off.
- Edit size overestimated (~1.1 vs 0.8), so the real repair is sized by the water balance instead.
- Expected Part 2 if this carries over: cos ~0.97 -> ~47/50.

## Ready to submit
`submissions/sub01_fc6_rank1.csv` (fc6 = given fc6 - 0.86 u v^T, direction = v). Not yet submitted (daily limit reached 2026-10-07).

## Wider test bench (after first leaderboard score: 71.29)

- The old wd=4 test model gave identical results on every seed because its fc5 activity had collapsed to a single direction, so the random part of each planted edit did nothing. That test was uninformative.
- `scripts/bench2.py` plants 24 varied rank-1 edits in the wd=1 model, whose fc5 activity spread is close to the real one. The edits vary in how much v leans on fc5's main direction, whether v is random or follows real fc5 activity patterns, and the shape of u. Direction recovery reached cos(v) 0.9–0.99 in most cases.
- `scripts/bench_sizing.py` compares three ways to size the fix:
  - A, water balance (what sub01 uses): size 0.63×, leak 0.80–1.24, worst NSE drop −0.093.
  - B, the solver's own scale: overshoots, with leak around 1.9.
  - C, re-fit only the part along fc5's main direction: size about 1.0×, leak 1.03–1.10, worst NSE drop −0.001.
- On the real model, C (`scripts/finalize_c.py`) gives almost the same fix as sub01: direction cos 0.998, edit 0.41 vs 0.37, and identical Dec–May numbers. Sizing isn't what limits sub01.
- On simple planted edits, the bench predicts about 85–95. The real score of 71 suggests the real poison differs from the simple rank-1 picture, or that part of the scoring works differently from what I assume. sub02 (Part 1 only) will show which part is short.

## Leaderboard split (2026-10-08)

- sub01 = 71.29 and sub02 (same fix, direction that can't match) = 26.33. So Part 1 ≈ 26.3 of 50 and Part 2 ≈ 45.0 of 50, which means the direction matches the real trigger at cos ≈ 0.95.
- Part 1 is about 0.53 of full, so one of leak × skill × surgical is low.
- On past windows with the same dates as the test (Dec 16 → May 3), sub01 removes 6.3% and 8.6% of the poisoned flow. That's close to the stated +7.6% phantom water, which points away from leak as the problem.
- By month, sub01 adds water in Dec–Feb, because v·h5 is slightly negative in winter. The total over the window is still a net removal.
- The leading suspect is surgical: the true edit may be smaller than 0.37. sub03 (0.28) tests this.
