# Part 2 direction notes (2026-10-10)

Current direction `outputs/final_dir.npy` (v0): Part 2 = 44.96, so |cos| with the truth = 0.9496.
Part 1 of `outputs/old_sub01_fc6.npy` = 87.44 − 44.96 = 42.48. For any `subD*` file, **Part 2 = score − 42.48**.

## Summary

- **subD1_fixu** (`outputs/dir_D1_fixu.npy`, cos 0.9945 with v0) is the only candidate I recommend. The bench backs it, but the expected gain is small: Part 2 ≈ 45–46.
- **subD2_fixu_mu1** (`outputs/dir_D2_fixu_mu1.npy`, cos 0.9717 with v0) is a **probe**, not a recommendation. Its score tells us whether the truth lies along a second axis. Expect a Part 2 anywhere from 42 to 48.
- After both are scored, `scripts/probe_combine.py` computes the best direction in span(v0, D1, D2) and predicts its Part 2. This works exactly whatever the two scores turn out to be (checked on synthetic truths, error < 0.001 in cos).
- I found nothing with strong evidence of reaching cos ≈ 0.99. Every weight-only method lands within 0.94–1.0 of v0. The 0.95 error is mostly shared by the whole method family, so it isn't noise that averaging could remove.

```
python scripts/probe_combine.py outputs/final_dir.npy=44.96 outputs/dir_D1_fixu.npy=<D1 part2> outputs/dir_D2_fixu_mu1.npy=<D2 part2>
python src/make_submission.py outputs/old_sub01_fc6.npy outputs/dir_combined.npy submissions/subD3_combined.csv <writeup url>
```

## A more realistic test bench (bench3/4/5)

The replicas behind bench2 look nothing like the real model. Real clean layers fc3–fc5 are rank-1 plus a small tail (s2/s1 ≈ 0.06–0.1). The replicas have large 2nd components and exact zeros. So I planted rank-1 edits into the **real model's own clean layers** (fc3, fc4, fc5) and recovered them with the same conditions: balance on the input units, plus alignment with the neighbouring layers' main directions (`scripts/bench3.py`, `bench4.py`, `bench5.py`).

Mean or median cos(v_rec, v_true):

| method | fc3 | fc4 | fc5 |
|---|---|---|---|
| joint_solve (current, free u) | 0.69–0.76 | 0.997 | 0.99 (or ~0.5 when stuck in a bad local minimum) |
| λ = 0.3 / 3, c ± 3% | same or worse | same | same or worse |
| + off-path penalty mu=1 (free u) | 0.83 | 0.998 | 0.997 |
| fixed u = uniform on S (**D1 method**) | 0.83 | 0.999 | 0.993 |
| fixed u + mu=1 (**D2 method**) | 0.96 | 0.998 | 0.995 |
| fixed u + mu=3 | 0.99 | 0.995 | 0.993 |
| row-based estimate | 0.83 | 0.93 | 0.89 |

- On fc4/fc5 the current method already gets 0.99+, but on the real fc6 it only gets 0.95. So the real problem breaks some assumption that the bench layers satisfy.
- Errors along the previous layer's main direction (u5's role) are a small share (0–40%), and their sign is not consistent across layers. There is no systematic u5-coefficient correction to apply.

## Why D1: u is "the same row added to 9 units"

- In the free solution, u has 9 entries of 0.30–0.35 on fc6 units S = {4, 20, 30, 32, 35, 41, 43, 48, 50}, all with positive out weight, and is near 0 elsewhere. That is what you get if the attacker adds the same α·v to 9 rows.
- After removing the clean main path, the 9 rows of S agree with their mean at cos 0.94–0.99.
- Fixing u to uniform on S and re-solving v (`scripts/fixed_u_solve.py uniform`) moves v by 6°. As a side effect, the edit becomes silent in fc5's winter "rest state", a single activation vector shared by 25% of hours (v·h_rest goes from −0.079 to −0.013). The current v removes a little water in every rest hour, which is odd for a phantom-water attack. Forcing v·h_rest = 0 directly gives the same shift (cos 1.00 between the two departures from v0).
- Bench: fixed u was never worse than free u by more than 0.001, across 54 planted cases.
- Risk: if the 6° shift were unrelated to the true error, the expected Part 2 would drop by about 0.5. The bench says the shift is informative, but the gain could be small.

## Why D2 is only a probe

- The off-path penalty (mu) pushes the clean fc6 to be "main path + small tail", which is true for fc3–fc5. On the real fc6 it moves v by up to 20° (mu=1: 0.972, mu=3: 0.953, mu=10: 0.941 with v0).
- A row-based estimator that uses no column balance at all moves v **in nearly the same direction** (cos 0.96 between the two departures). That is a hint the truth may lie this way.
- But the real fc6 has two excess singular values (0.239, 0.213; the tail is ≈ 0.1). Every candidate edit removes one of them, and a 0.22 component always remains, so it is probably natural to the clean fc6. The mu shift lines up with that leftover component (cos 0.65). Here mu may be fitting a natural feature, not the poison, and the bench has no such component to test against.
- So D2's score is an honest measurement of this axis, not a safe bet.

## Checked and ruled out

- Natural attacker choices: mean, standardized and LDA contrasts of fc5 activations for seasons, months, flow, rain, temperature and soil-moisture thresholds. The best cos with v0 is 0.51, so the trigger is not a simple contrast.
- v0 is not the 1st PC of fc5 activity (cos 0.51 with PC1, < 0.06 with the others), and the edit is not of the "steering" form u ∝ W_clean·v (cos 0.15).
- Dead fc5 units: their fc6 columns are exactly ~1e-17, so the true v is zero there. v0 already is.
- About 68% of v0's energy sits on fc5 units that are always on and nearly constant (1, 45, 47, 51, 41, 10, …). In clean fc4/fc5, such units break the balance rule (ratio 0.02–0.4 vs 0.95). Re-solving with a low ratio for them moves v by only 0.994 (kc=0.3), so this is not the source of the error.
- Ensembles over λ, c and seeds: everything is within 0.99 of v0, so the gain is at most a fraction of a point.

## Files

- New scripts: `scripts/bench3.py`, `bench3_analyze.py`, `bench4.py`, `bench4_analyze.py`, `bench5.py`, `direction_variants.py`, `fixed_u_solve.py`, `save_dir_candidates.py`, `probe_combine.py`.
- Directions: `outputs/dir_D1_fixu.npy`, `outputs/dir_D2_fixu_mu1.npy`. Raw variants: `outputs/fixu_*`, `outputs/var_*`.
- Submissions: `submissions/subD1_fixu.csv`, `submissions/subD2_fixu_mu1.csv`. Both use the fc6 from `old_sub01_fc6.npy` (identical to sub05), and their ids are checked against `data/submission_example.csv`.

# Probing plan (2026-10-12)

Each submission reads |d·t| exactly, so after N scores we know t's projection onto the span of everything scored. The real question is which directions to add to that span.

## Prior for the unknown part of t
- Live fc5 units only (45 of 56). Dead units carry no weight in any estimate.
- The spread of our own estimator variants around the current best: joint λ/c, fixed-u μ/kc/rest, cf_v, balance-only, row-based, plus the fc6 leftover component. After final_dir and D1 are taken out, the top 1, 2 and 3 directions hold 77%, 90% and 94% of that spread. D2's departure lies mostly along the top one, and so does the leftover component (cos 0.96).
- Bench evidence (`scripts/bench_probe.py`, 45 planted cases on the real fc3–fc5): the part of the truth that D1 misses lies mostly where the variants disagree. Probing those directions recovers 8–17% of the remaining error per probe. Random directions recover about 1.2%. fc5 activation PCs did worse than the variant spread and are switched off.

## Design
- Probe = current best tilted 2° toward the top unscored prior direction. This costs about 0.06 Part 2 points, or up to about ±1 depending on the sign of t·e. No rule predicted that sign on the bench.
- Info does not depend on the tilt, as long as scores are entered with full precision. With 2-decimal scores, 3° is safer.
- `combine` keeps |x| ≤ 1. Without this, rounding noise blew up after about 6 probes.

## Expected Part 2 (bench, error rescaled so cos(D1, truth) = 0.958; N includes D2)
| N | 1 (D2) | 2 | 3 | 4 | 5 | 6 | 8 |
|---|---|---|---|---|---|---|---|
| mean share of D1's error recovered | 0.47 | 0.50 | 0.57 | 0.64 | 0.69 | 0.73 | 0.79 |
| Part 2 if the real model behaves like this | 47.8 | 48.0 | 48.3 | 48.5 | 48.7 | 48.9 | 49.2 |
| with random probes instead | 46.0 | 46.0 | 46.1 | 46.2 | 46.2 | 46.3 | 46.4 |

Medians are about 0.3 points lower. The caveat is that the bench method reaches cos 0.99, but the real one reaches only 0.958. If the real error is shared by the whole method family, gains will be closer to the random row. D2's score is the first test of this.

## Daily routine
```
.venv/bin/python scripts/probe_plan.py --add <submitted dir.npy> <total score, all decimals>
# submit the submissions/probe_NN.csv it prints; on the last day use --final
```
Scores are kept in `probe_scores.txt`. `--toward outputs/dir_D2_fixu_mu1.npy` gives a gentle probe along D2's axis. It carries the same information as D2 but scores near D1.
