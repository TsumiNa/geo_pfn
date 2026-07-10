# Diagnosis: the LCSG gap was under-training, not the prior — and our SCM is fine

**Date:** 2026-07-10 · **Hardware:** ISM A100×4 (megalith3) · **Branch:** `feat/geo-realistic-prior`

## TL;DR (read this)

Three questions were on the table: (a) does the **geo-realistic prior** fix "geo-PFN
gets worse with real cheap features (LCSG > L)"; (b) is the bottleneck the **SCM
(prior)** or the **architecture**; (c) does **capacity/training** help. Answers:

1. **The geo-realistic prior is rejected** — monotonically harmful.
2. **The LCSG > L gap was mostly UNDER-TRAINING.** With ~6× more training the 2M
   model's gap collapses from **+3.9 → +0.1** (LCSG 22.55 → **17.69**); a 17M model
   gets **LCSG 16.81 < L**, i.e. geo-PFN finally *exploits* the cheap features and
   **beats TabICL on L** (17.74 vs 18.06).
3. **Our geo-SCM is NOT the bottleneck.** Training our architecture on **TabICL's own
   SCM** made LCSG *much worse* (26.09 vs our 17.69). Swapping their (excellent,
   general) prior in did not help.

Net: the "prior realism" detour was a red herring. The real levers are **more
training + more capacity on our existing geo-SCM**. The remaining gap to TabICL is
now only on LCSG absolute accuracy (16.81 vs 13.9), and is a scaling/architecture
question, not a prior-quality one.

## 1. Protocol

Whole-borehole holdout, 5-fold GroupKFold (seed 42), coherent per-borehole context
(`geo_pfn.haneda.holdout_eval`). geo-PFN checkpoints load their own `model_config`,
so 2M/17M/40M are scored identically. RMSE of Su (kN/m²), lower better. TabICL
reference: L 18.06, LCSG ~13.9.

## 2. Results

| # | training prior | steps × batch (tables) | params | L | LCSG | LCSG−L |
|--:|---|---|--:|--:|--:|--:|
| 1 | generic (old, MPS) | 8k × 16 (128k) | 2.0M | 19.19 | 22.89 | +3.7 |
| 2 | generic | 12k × 16 (192k) | 2.0M | 18.63 | 22.55 | +3.9 |
| 3 | realistic p=0.5 | 12k × 16 | 2.0M | 19.05 | 26.85 | +7.8 |
| 4 | realistic p=1.0 | 12k × 16 | 2.0M | 27.86 | 29.99 | +2.1 |
| 5 | **generic** | **24k × 48 (1.15M)** | **2.0M** | **17.57** | **17.69** | **+0.1** |
| 6 | **generic** | 24k × 24 (576k) | **16.8M** | 17.74 | **16.81** | **−0.9** |
| 7 | generic | 24k × 12 (288k) | 39.2M | 18.77 | 18.17 | −0.6 |
| 8 | **TabICL-SCM** | 24k × 48 (1.15M) | 2.0M | 17.04 | 26.09 | +9.1 |
| — | *TabICL (ref)* | — | — | *18.06* | *~13.9* | *−4.2* |

## 3. What each finding means

**(2 → 5) The gap was under-training.** Same 2.0M architecture, same generic prior;
only the amount of training changed (192k → 1.15M tables). LCSG dropped **22.55 →
17.69** and the LCSG−L gap went **+3.9 → +0.1**. So "adding real cheap features hurts"
was largely a symptom of an under-fit model, exactly as hypothesised earlier from the
scatter plots. (Caveat: steps *and* batch both grew, so this is "much more training",
not a clean 2× steps; the direction is unambiguous.)

**(6) Capacity helps, and geo-PFN becomes competitive.** The 16.8M model reaches
**LCSG 16.81 < L 17.74** — it now *gains* from the cheap features (negative LCSG−L,
the same sign as TabICL), and its L **beats TabICL** (17.74 vs 18.06). Note it got
there on only *half* the tables of the 2M run, so capacity is a real, separate lever.

**(7) 40M is under-trained, not worse.** At batch 12 it saw only 288k tables — a
quarter of the 2M run — so its higher numbers reflect too little data, not a capacity
ceiling. A controlled sweep (equal tables per size) is needed to read the scaling law
cleanly; this run does not.

**(8) The SCM swap settles "prior vs architecture".** Our architecture trained on
**TabICL's own mix-SCM** (MLP + tree, made regression via `num_classes=0`) gets **LCSG
26.09** — far worse than our geo-SCM's 17.69 at the same 2M/24k budget. TabICL's prior
is generic-tabular and lacks the depth/layer/spatial structure our geo-SCM injects, so
on this borehole task **our SCM transfers better**. Conclusion: **the prior is not the
problem; if anything ours is the right prior for this data.** The remaining LCSG gap to
TabICL is therefore an **architecture/scale** question (its distribution-aware column
encoder, its size), not a prior-quality one.

## 4. Where geo-PFN now stands vs TabICL

| | L | LCSG |
|---|--:|--:|
| geo-PFN best (16.8M, 24k, our-SCM) | **17.74** (beats TabICL) | 16.81 |
| TabICL | 18.06 | **13.9** |

geo-PFN, properly trained and sized on our geo-SCM, **matches/beats TabICL on
coordinates-only** and **exploits the cheap features** (LCSG < L). It still trails
TabICL's absolute LCSG (16.81 vs 13.9) — the one remaining, well-scoped gap.

## 5. Recommendations (revised)

1. **Drop the geo-realistic block.** It is net harmful; the generic random-MLP branch
   is the right prior. (Keep `p_geo_realistic=0.0` as default, or remove.)
2. **Train longer / bigger on the generic geo-SCM.** Under-training was the main issue;
   a **controlled capacity sweep** (equal tables per size, via more steps for bigger
   models or grad-accumulation to fix effective batch) will read the scaling law and
   tell us how far capacity closes the last LCSG gap. (17M @ longer training launching
   now.)
3. **Then, if scale plateaus, target the column-ingestion stage** — a stronger,
   distribution-aware column encoder is the most likely lever for the last LCSG gap,
   since that is exactly where TabICL's advantage lives.

## 6. Reproduce

```
# training knobs added this session: --p-geo-realistic, --num-workers (prefetch),
# --d-model/--n-heads/--col-layers/--row-layers/--feature-emb-dim, --prior geo|tabicl
uv run python -m geo_pfn.haneda.holdout_eval --checkpoint <ckpt> --device cuda \
    --models geopfn --feature-sets L,LCSG --out results/haneda/holdout_<name>.json
```
Evidence JSONs: `results/haneda/holdout_*.json` (gen_2m_24k, gen_17m, gen_40m,
tabicl_scm_2m, ablate_p00/p10, holdout_old/new2m).
