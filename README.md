# Fixing the Poisoned Well

A project with UW Madison ML+X.

A small neural network forecasts hourly river flow for a Wisconsin stream. One of its layers was edited so it predicts about 14% too much water on the public data, while overall accuracy barely changes. Three other layers have decoy edits.

Goals:
1. **Fix it.** Correct the edited layer's weights so the extra water is gone and accuracy is kept, while changing as little as possible.
2. **Explain it.** Find the trigger: the pattern in the layer's input that switches the extra water on.

## The model

- **Inputs (77):** the last 72 hours of rain plus 5 fixed facts about the catchment.
- **Hidden layers:** 6 (`fc1`–`fc6`), each with 56 ReLU units.
- **Output:** 1 value, the hourly flow.

## How I solved it

**1. Ruled out the 3 decoys.** They look suspicious but do nothing:
- `fc2` unit 36 is scaled up about 22×, and `fc3` scales it back down. The two cancel out.
- `fc4` has ±5 weights on `fc3` units that never switch on.
- `fc5` has 0.2 weights on `fc4` units that never switch on.

Plain SVD of the weights gets fooled by these.

**2. Found the poisoned layer (`fc6`).** In a clean, heavily regularized network:
- each layer's main pattern lines up almost perfectly with the next layer's;
- each unit's incoming and outgoing weight strengths stay in a fixed ratio.

`fc2`–`fc5` follow both rules, with alignment of 0.99–1.00. The link into and out of `fc6` breaks them (0.94 and 0.93), and `fc6` also has unusually strong patterns that no other layer has.

**3. Recovered the edit from the weights alone.** The edit is a single pattern, `ΔW = u vᵀ`:
- `v` is the trigger in `fc5`'s output.
- `u` pushes up the `fc6` units that raise the output.

`scripts/joint_solve.py` solves for `u` and `v` so that, once the edit is removed, `fc6` follows the clean-network rules again. The extra water it explains peaks in June and is larger at high flows.

**4. Tested the method on known edits.** I added known edits to clean copies of the model (`scripts/train_long.py`, `scripts/bench2.py`) and checked how well the method recovered them before trusting it on the real model.

**5. Tuned the fix.** Checking each part on its own showed that the recovered trigger was strong but the structural fix was too big. I switched to the best small change fitted to the observed flow, then tuned its size until it removed all of the extra water.

## Repo layout

| Path | What's in it |
|---|---|
| `model/`, `data/` | Provided files: poisoned weights, scaler, config, rain/flow data, example output |
| `streamflow_model.py`, `starter.py` | Provided starter code |
| `src/pw.py` | Shared helpers: loading, forward passes from any layer, patching a layer, metrics, decoy stripping |
| `src/make_submission.py` | Writes the output CSV (fixed weights + trigger) and checks its format |
| `scripts/` | Every experiment (see below) |
| `FINDINGS.md` | Full log of what worked and what didn't |
| `DIRECTION_NOTES.md` | Notes on refining the trigger direction |

Key scripts:
- `joint_solve.py`: recovers the edit (`u`, `v`) from the weights.
- `finalize.py`: sizes the structural fix.
- `train_long.py`: trains clean copies of the model.
- `bench2.py`, `bench_sizing.py`, `bench3.py`: test benches with known edits.
- `bench_oldfix.py`: the fitted small fix.
- `fixed_u_solve.py`, `save_dir_candidates.py`: refined trigger directions.
- `probe_combine.py`: merges several measured directions into a better one.
- `make_best.py`: rebuilds the final output.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install numpy pandas scipy torch
.venv/bin/python starter.py                      # check that the data and model load
.venv/bin/python scripts/joint_solve.py          # recover the edit
.venv/bin/python scripts/finalize.py             # size it, save outputs/final_fc6.npy + final_dir.npy
.venv/bin/python scripts/fixed_u_solve.py uniform                     # refined trigger (D1)
.venv/bin/python scripts/fixed_u_solve.py uniform 1.0 0.0 1.0         # D2 variant
.venv/bin/python scripts/save_dir_candidates.py
.venv/bin/python scripts/make_best.py outputs/best.csv
```

Generated files go to `outputs/`, which is ignored by git.
