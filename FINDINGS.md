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
