# Prior-realism diagnostic — the geo-realistic block does not work

**Date:** 2026-07-10 · **Hardware:** ISM A100×4 (megalith3) · **Branch:** `feat/geo-realistic-prior`

## TL;DR (verdict)

The geo-realistic cheap-feature block (docs/geo-scm-design.md §10) was meant to fix
"geo-PFN gets *worse* when given the real cheap-geotech features" (LCSG holdout >
L). **It does not. It is monotonically harmful** — the more the training prior uses
it, the worse geo-PFN gets, on *both* feature sets. The best geo-PFN is the **pure
generic (random-MLP) prior**, and the **LCSG > L gap persists regardless**. The
prior-realism hypothesis, as implemented, is rejected.

Capacity is **not** the missing piece either (see §4): the 16.8M model on the mixed
prior did not close the gap; a clean generic-prior capacity sweep (2M/17M/40M) is
running to confirm.

## 1. Protocol

Whole-borehole holdout, 5-fold GroupKFold (seed 42), coherent per-borehole context
(`geo_pfn.haneda.holdout_eval`). All geo-PFN checkpoints are the same 2.0M
two-stage architecture unless noted; only the training prior's
`p_geo_realistic` (fraction of tables drawn from the realistic block) and the
step/param counts vary. RMSE of Su (kN/m²), lower is better.

## 2. Results — the ablation over prior mix

| geo-PFN training prior | steps | params | L (coords+depth) | LCSG (+cheap geotech) |
|---|---:|---:|---:|---:|
| generic (old MPS ckpt) | 8k | 2.0M | 19.19 | 22.89 |
| **generic, p=0.0** | 12k | 2.0M | **18.63** | **22.55** |
| realistic, p=0.5 | 12k | 2.0M | 19.05 | 26.85 |
| realistic, p=1.0 | 12k | 2.0M | 27.86 | 29.99 |
| realistic, p=0.5 | 24k | 16.8M | 19.80 | 25.54 |
| *TabICL (reference)* | — | — | *18.06* | *~13.9* |

**Reading it:**
- **Monotone harm in the mix fraction.** LCSG: 22.55 (p=0) → 26.85 (p=0.5) → 29.99
  (p=1.0). L: 18.63 → 19.05 → **27.86**. More realistic-block = worse, always.
- **Best geo-PFN = pure generic** (18.63 / 22.55), essentially the original number,
  just trained a bit longer than the old 8k checkpoint.
- **The gap we set out to fix is untouched:** even the best geo-PFN has LCSG (22.55)
  *worse* than L (18.63) — adding the real cheap features still hurts it by ~4 RMSE.
- **On L, geo-PFN ≈ TabICL** (18.63 vs 18.06). The entire geo-PFN deficit is on
  **LCSG**, where TabICL *gains* from the cheap features (13.9 << 18.1) and geo-PFN
  *loses* from them (22.55 > 18.63).

## 3. Root cause — why the realistic block backfired

The decisive clue is that **p=1.0 wrecks even L** (27.86), the coordinates-only
setting. The realistic block never touches the coordinate/depth structure — so if it
hurts L, the damage is in the **target**, not the features.

In `realistic.py` the synthetic target is

```
target = 1.0·depth_norm  +  0.05·soft  +  0.55·signal  +  0.45·noise
```

i.e. **approximately a noisy *linear* function of normalized depth.** A model
trained mostly (p=1.0: entirely) on this learns "Su ≈ linear-in-depth + noise." Real
Su is **nonlinear and layered** in depth (the whole point of the geo-SCM design:
"smooth within a layer, jump across a boundary"). So the realistic-trained model
**underfits the real depth→Su structure** — visible immediately as worse L, and
compounded on LCSG.

The generic random-MLP branch, by contrast, produces a **rich, nonlinear** target
(a deep node of a random network), which turns out to be a much better inductive
bias for real Su — even though its *features* are unrealistic.

So the block traded a good target for realistic feature marginals, and the target
mattered far more. **Feature realism alone did not teach geo-PFN to exploit the
cheap features; it only degraded the target model.**

## 4. Capacity is not (obviously) the lever

The 16.8M model on the p=0.5 prior (19.80 / 25.54) is **no better** than the 2.0M on
the same prior (19.05 / 26.85) — capacity did not rescue a bad prior. To test
capacity *fairly* (on the best = generic prior), a clean sweep is running:

| params | prior | steps | L | LCSG |
|---|---|---:|---:|---:|
| 2.0M | generic | 24k | _pending_ | _pending_ |
| 16.8M | generic | 24k | _pending_ | _pending_ |
| 39.2M | generic | 24k | _pending_ | _pending_ |

(Results appended when the runs finish — ~2 h. This is the honest test of the
user's "does scaling params help" question and whether any scaling-law trend
appears on the *good* prior.)

## 5. What this means & what to try next

The borehole-similarity *mechanism* still holds (sparse-context report): geo-PFN's
row-metric ICL genuinely exploits neighbour boreholes. The failure is narrower and
specific: **the synthetic prior does not teach geo-PFN to read real cheap-geotech
features**, and my attempt to fix that by "realistic marginals + weak coupling"
broke the target model.

Concrete next steps, in priority order:

1. **Keep the rich generic target; make only the *features* realistic.** The bug was
   coupling feature-realism to a hand-made linear target. Redo the block so the
   target stays the random-MLP node, and only the *feature marginals/collinearity*
   are remapped — or drop feature-realism entirely if the sweep shows capacity/other
   levers matter more.
2. **Interrogate what TabICL does that geo-PFN can't.** TabICL's distribution-aware
   column encoder turns LCSG's cheap features into a −8 RMSE gain; geo-PFN turns them
   into a +4 loss. The gap is in *how features are ingested*, suggesting an
   architecture change (a stronger, distribution-aware column stage) over more prior
   tinkering.
3. **If the generic sweep shows a params trend**, scale on the generic prior (not the
   realistic one) and revisit.

## 6. Reproduce

```
# ablation (done): checkpoints checkpoints/{ablate_p00,ablate_p10,geopfn2stage_realistic}.pt
uv run python -m geo_pfn.haneda.holdout_eval --checkpoint <ckpt> --device cuda \
    --models geopfn --feature-sets L,LCSG --out results/haneda/holdout_<name>.json
# training knobs added this session: --p-geo-realistic, --num-workers (prefetch),
# --d-model/--n-heads/--col-layers/--row-layers/--feature-emb-dim
```
Evidence JSONs: `results/haneda/holdout_*.json`.
