# Fixing the Poisoned Well — Fresh-Start Plan

## Context

**The competition.** Someone tampered with a small neural net (77 inputs → 6 hidden layers of 56 → 1 output) that predicts hourly streamflow for a Wisconsin catchment. The tampering makes the model produce "phantom water" (+7.6%/yr on the test period) while normal accuracy (NSE) barely changes. Several layers contain loud **decoy** edits that do nothing. Only one layer holds the real change, and it's spread across many weights along one hidden "trigger" direction.

**What we submit (100 pts, scored on a hidden period Dec 16 2025 → May 3 2026):**
| Part | Submit | Score |
|---|---|---|
| 1. Fix it (50) | `fc_weight_corrected` 56×56 — repaired weights of the poisoned layer | `50 × leak_removed × skill_preserved × surgical` |
| 2. Explain it (50) | `feature_direction` 56-vector — trigger direction in that layer's **input** activation space | `50 × clamp((|cos|−0.5)/0.5)` |

- `leak_removed`: share of the excess bias removed (vs the true clean model).
- `skill_preserved`: full credit if NSE within 2% of clean; zero by a 10% drop.
- `surgical`: full credit if our edit's size ≤ 1.5× the true tampering; zero at ≥ 4×. **So the fix must be small and targeted, not "retrain the layer."**
- A Discussion writeup link is required on every row (pass/fail).

Note: the layer is never named in the CSV — the scorer must already know which layer it is. So picking the wrong layer likely means Part 1 ≈ 0 and Part 2 ≈ 0.

## Where things stand (done this session)

- Project at `~/Documents/fixing-the-poisoned-well`: `model/` (weights, scaler, config), `data/` (train.csv, example sub), `starter.py`, `streamflow_model.py`, `.venv` (numpy, pandas, scipy, sklearn, torch, matplotlib, jupyter, kaggle). Git initialised, 1 commit. `starter.py` runs.
- Public data: 18,685 usable hours (Oct 2023 → Dec 16 2025). Poisoned model: bias **+0.0032 mm/hr (+13.8%)**, NSE 0.679.

**Early findings (read-only probes):**
1. **fc4's `5.0` weights are decoys.** They sit on fc3 units 8 and 55, which never fire. Zeroing them changes the output by exactly 0.
2. **Plain SVD gets fooled.** Raw SVD of fc4 is dominated by those decoys (top/2nd singular value ratio 47). One public writeup used that and scored 0 on the leaderboard.
3. **Each layer is basically one "highway".** After masking dead inputs, every layer has one huge singular direction. Removing it breaks the model (NSE goes negative). The poison must be a **small side-channel** next to the highway, not the highway itself.
4. **Candidates that cut bias *and* raise NSE** (a poison signature): fc2 component 1 (bias 0.0032→0.0026, NSE 0.679→0.687), fc6 component 2 (→0.0019, NSE 0.681). These are leads only, not conclusions.
5. Over-prediction by month is mostly positive Feb–Jun and Sep–Oct. The hidden test period (winter/spring) is where the trigger has to be fixed.
6. `model_config.json` metrics (train bias +0.00017) look like the **clean** model's numbers. They use a random seed-0 split that includes hidden rows, so we can't reproduce them exactly. They still give a target: **the clean model's bias is ≈ 0**.

## Approach

### Step 1 — Shared toolkit (`src/pw.py`)
Reuse `streamflow_model.py` (`load_weights`, `forward(return_hidden=True)`, `build_features`, `nse`, `rmse`, `cos_sim`). Add:
- `run_with_patch(weights, layer, W_new)` and `forward_from(layer, h)` to run from any hidden layer, which we need for patching.
- `live_mask(acts)`: which units ever fire.
- `proxy_score(W_new, layer, v)`: bias removed (assuming clean bias ≈ 0.0002), NSE change, ‖ΔW‖, and a Part 1 estimate. Report it on all public data **and** on winter/spring months only (to mimic the test period).
- Torch mirror of the model for gradients.

→ verify: patching with the original weights gives identical predictions.

### Step 2 — Find the real layer (decoys out)
For each candidate layer fc2–fc6:
- **Decoy sweep:** flag outlier weights and check whether their input or output unit is ever alive. Dead = decoy, crossed off.
- **Bias attribution:** gradient of mean prediction (= bias) with respect to each layer's weights, d bias/dW = E[δ_out ⊗ h_in]. This shows which layer the excess is most sensitive to, using only live paths.
- **Activation patching:** project small non-highway directions out of each layer's input and measure bias drop vs NSE change. The poisoned layer should show a direction that removes the excess while keeping or *improving* NSE.
- **Physics check:** compare the excess to the water balance (rain in vs predicted flow out) per storm and per dry spell. When does phantom water appear: after rain, in dry periods, at certain soil moisture or season? That gives the trigger's real-world meaning, which helps the writeup and backs up the direction.

→ exit criterion: exactly one layer where a low-rank, small-norm edit removes most of the bias and keeps NSE. Commit the analysis notebook.

### Step 3 — Recover the trigger direction (Part 2)
In the chosen layer's 56-d input space, estimate the direction **v** with several independent methods, then check they agree:
1. **Off-highway structure in W:** take W on live units, remove the highway (top component of the activation-weighted W, i.e. SVD of W·Σ_h^{1/2}), and find the leftover rank-1 piece u vᵀ.
2. **Causal gradient:** the direction in h_in that most changes the bias per unit of normal variation (gradient of the bias, whitened by the activation covariance).
3. **Contrast:** mean h_in when the residual is high (phantom water) minus when it's normal, matched on rain/flow so we don't just recover "it's raining".
4. **Sparse autoencoders**, several seeds and sizes. Keep only features that repeat across runs and causally move the bias. The top public solution did exactly this.

Combine with a consensus (average of sign-aligned vectors, or the top singular vector of the stack). **Stability matters**: pairwise cosine between methods should be > 0.9 before we trust it.

→ verify: removing v (ΔW = −α·(W v) vᵀ) kills the excess without hurting NSE, and a random direction of the same size doesn't.

### Step 4 — Repair the layer (Part 1)
- Rank-1 surgical edit: **W' = W − α · u vᵀ**, with u fitted by least squares (or from Step 3 method 1).
- Choose α so the **public bias drops to the clean level (≈ +0.0002)**, with physics/water balance setting the size, not just NSE.
- Keep ‖ΔW‖ as small as possible: a small torch optimisation over (u, α) with v fixed, minimising ‖ΔW‖ subject to bias ≈ clean and NSE ≥ poisoned NSE.
- Check on Dec–May public months only, since that's where we're scored.

→ verify: `proxy_score` ≈ full on bias and NSE, edit stays rank-1 and small.

### Step 5 — Submit & writeup
- `src/make_submission.py`: writes `submissions/subNN.csv` (same format as `starter.py`, pub_ + prv_ copies) and checks shape, ids and finite values against `data/submission_example.csv`.
- Writeup draft in `WRITEUP.md` (methods, decoys, what failed, layer, direction, repair, plus a couple of figures). **You post it to Discussion yourself** (it's public), then I put the link in the CSV.
- Submitting through the Kaggle CLI needs your OK each time. Use submissions sparingly as checks, not for guessing.

## Files
- `src/pw.py` — toolkit (new)
- `notebooks/01_localize.ipynb`, `02_direction.ipynb`, `03_repair.ipynb` — analysis with plots
- `src/make_submission.py` — submission builder
- `WRITEUP.md` — writeup draft
- Unchanged: `streamflow_model.py`, `starter.py`, `model/*`, `data/*`

## Risks
- **Wrong layer → ~0 points.** Mitigated by needing agreement from the decoy sweep, bias attribution and patching before moving on.
- **True edit size unknown** (surgical factor). Lean toward the smallest edit that fixes the bias; over-correcting costs both leak and surgical credit.
- **Clean bias unknown.** Target ≈ +0.0002 from config, and check the Dec–May slice.
- Public score caps at 50, so the leaderboard only partly confirms our work. Rely on our own checks.

## Verification (end-to-end)
1. `./.venv/bin/python starter.py` still runs.
2. `proxy_score` on the final (W', v): bias ≈ clean level, NSE ≥ 0.679, rank-1 ΔW, stable v (cosine > 0.9 across methods and seeds).
3. Random-direction control does nothing.
4. Submission file validates against the example's ids. Submit once with your approval and record the score.
5. Commit after each step (allowed in this repo); no pushing.
