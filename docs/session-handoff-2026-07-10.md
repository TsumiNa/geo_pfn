# Session handoff — 2026-07-10

For whoever (human or agent) picks this project up next. Everything below was
verified in this session; the integrated story with charts is
[`docs/stage-report.html`](stage-report.html) (trilingual, 7 tabs) — read that
first, then this file for the operational details it doesn't carry.

## 1. Positioning (the north star — do not drift from this)

**We are NOT building another general tabular model.** TabICL / TabPFN / TabFM
already serve the i.i.d.-rows, dense-context, cell-filling regime. Our niche is
**domain-specific sparse prediction**: boreholes, oil exploration, deep-sea
sensing, phase diagrams — data that is strongly correlated vertically/spatially,
undergoes phase transitions with depth/position, and where every point is
expensive. The primary metric is **sparse-regime performance** (few whole
boreholes as context), and **trustworthy uncertainty** is a first-class
deliverable. Evaluate every idea against the sparse protocol first, dense second.

## 2. State of results (all on real Haneda data, whole-borehole holdout)

Vocabulary: `L` = coords+depth; `LCSG` = + cheap geotech + soil + grain size.
Dense = 5-fold GroupKFold (seed 42). Sparse = fold-0 query (48 holes), N whole
context holes from the remaining 192, 4 seeds.

| Fact | Numbers | Where |
|---|---|---|
| Under-training caused "features hurt" | 2M LCSG 22.9→17.7 as tables 192k→1.15M; gap to L +3.9→+0.1 | stage-report ③ |
| 2M saturates | ~1M tables; 24k≈48k in both dense and sparse | holdout/sparse `gen_2m_48k` |
| 17M = feature exploiter | dense LCSG **16.42** (best ours; TabICL 13.94); sparse-LCSG N=25 **19.03 beats TabICL 20.42** | `gen_17m_48k` |
| Best-at-dense ≠ best-at-sparse | 17M loses to 2M on sparse-L everywhere (over-extrapolates at N≤6) | sparse_gen17m vs gen2m24k |
| Mid-sparse L is our home turf | 2M vs TabICL at N=25/50: **20.1/20.7 vs 26.3/24.0** | sparse_gen2m24k |
| Realistic prior REJECTED | monotonically harmful (p=1.0 wrecks even L: 27.9); root cause: target ≈ linear-in-depth | prior-realism-diagnostic.md |
| Our geo-SCM > TabICL's SCM | matched budget, LCSG 17.7 vs 26.1; sparse LCSG worse by ~7–8 everywhere | scm-swap.html |
| Uncertainty calibrated at extreme sparsity | N=3 promised-90% hits **91.0%**; outside-support ≤0.4%; intervals widen honestly | uncertainty-sparse.html |
| Calibration weak spots (conservative side) | N≥25 over-wide (97–98% hits); corr(std,\|err\|) only ρ≈0.2–0.36; LCSG N≤6 mildly over-confident (85–88%) | uncertainty_gen2m24k.json |
| Data structure | depth alone R²=0.564 (log Su); +6 cheap = +0.12; PC1 76.5% var but ~no signal (partial +0.02); signal in PC3/4 | geoprior/fit_stats.py, haneda_stats.json |
| 40M | inconclusive — unequal data budgets, volatile (dense LCSG 18.2→20.6 across budgets) | holdout_gen_40m* |

Best checkpoints by use case: **sparse-L → `gen_2m_24k.pt`**; **LCSG (dense or
mid-sparse) → `gen_17m_48k.pt`**. Do not use the realistic-prior or TabICL-SCM
checkpoints except as ablation references.

## 3. Next-stage plan (priority order, from stage-report ⑦)

1. **Distribution-aware column encoder** (biggest lever): TabICL-style
   set-encoding of each column's value distribution to replace the current
   column embedding. Target: close dense-LCSG 16.4→13.9 and sparse-LCSG N≥50.
   Re-run the full dense+sparse suite after.
2. **Small-context-objective training**: raise tiny-site (2–3 holes) and sparse
   anchor share in `geoprior/site.py` (`p_single`, `max_holes`, anchor config);
   select/early-stop on N=6/25 sparse RMSE + CRPS, never on dense loss.
3. **Sharpness + per-row uncertainty**: CRPS-style training objective; raise ρ.
4. **Equal-budget scaling sweep**: 2M/17M/40M at identical table counts
   (gradient accumulation to align effective batch), scored on sparse metrics.
5. Later: light real-data fine-tune; multi-target (full property table) + soil
   classification head; cross-site generalization.

Cleanup candidates: the rejected realistic-prior block (`geoprior/realistic.py`,
`fit_stats.py`, `haneda_stats.json`, config fields) can be removed or left as a
documented negative result — kept for ablation reproduction, and the default
`p_geo_realistic` is now **0.0** (fixed at PR-review time), so plain `train`
uses the generic geo-SCM.

## 4. Code map (what exists after this stage)

- `geo_pfn/geoprior/` — geo-SCM prior. `prior.py` (single hole),
  `site.py` (multi-hole site, the training default), `config.py`;
  `realistic.py` (rejected block), `tabicl_prior.py` (TabICL-SCM adapter:
  forces `num_classes=0` so `Reg2Cls` stays regression; needs
  `--with tabicl --with xgboost`).
- `geo_pfn/geopfn/` — model (`model.py`, 2-stage row-metric; `heads.py`
  bar-distribution), training (`train.py`; CLI has `--d-model/--n-heads/
  --col-layers/--row-layers/--feature-emb-dim/--num-workers/--p-geo-realistic/
  --prior geo|tabicl`), `prefetch.py` (multi-process sampler; workers pinned to
  1 thread), inference (`predict.py`: `predict_geopfn_coherent` point,
  `predict_geopfn_coherent_dist` full distribution), `calibration.py`
  (exact mixture metrics, MC-validated in `calibration_test.py`).
- `geo_pfn/haneda/` — data/protocols (`data.py`), evals:
  `holdout_eval.py` (dense), `sparse_context.py` (sparse sweep),
  `uncertainty_eval.py` (calibration), `anchor.py`, `su_grid.py`,
  `su_grid_ablation.py` (visual-only grids).
- Reports (`docs/*.html`, all trilingual 中/日/EN with the shared langbar
  pattern): **stage-report** (the integration), sparse-best-models, scm-swap,
  uncertainty-sparse, prior-realism-diagnostic.md, geopfn-design,
  sparse-context-results, model-comparison-report, su-prediction-comparison,
  su-grid-geology-ablation.
- Evidence: `results/haneda/*.json` (holdout_*, sparse_*, uncertainty_*,
  parity_data). Parquet grids are gitignored by design (user's call).

## 5. Infrastructure — the A100 box (critical for anyone continuing)

- `ssh ism-gpu-a100` → megalith3, 128 cores, 4× A100-40GB. Workdir:
  **`/data/claude/geo_pfn_20260710/repo`** (clone of this repo,
  branch feat/geo-realistic-prior; checkpoints in `checkpoints/`, logs in
  `logs/`). `uv` at `~/.local/bin/uv`. Data CSV was scp'd to `repo/data/`
  (gitignored).
- **SSH gotchas**: the user's ssh config has a LocalForward on port 8025 — a
  stale interactive session holding it makes every scripted ssh print bind
  errors and sometimes exit 255. Use `-o ClearAllForwardings=yes` (or kill the
  stale ssh). Piped stdout of remote background launches is unreliable through
  the gateway: **write remote output to files and `cat`/`grep` them**.
- **Throughput**: prior sampling is CPU-bound; always train with
  `--num-workers 12` (prefetch) → 95%+ GPU util. Workers are pinned to 1 thread
  (`torch.set_num_threads(1)` in `prefetch.py`) — do not remove, or 16 workers ×
  128-core OpenMP pools will thrash the box (load 133; concurrently-running
  sklearn/hgbt deadlocks). 2M trains 12k steps in ~11 min at batch 48.
- Speed reference: 2M ~8-10 it/s (b48), 17M ~4 it/s (b24), 40M ~3 it/s (b16).
- MPS (local Mac) works but is ~20× slower; only for smoke tests.

## 6. Lessons this stage (so they are not re-learned)

1. **Check training sufficiency before blaming the prior/architecture.** The
   entire "features hurt" mystery was under-training.
2. **Feature realism without target richness is worse than useless** — the
   realistic prior's near-linear-in-depth target broke even coords-only
   performance. If injecting realism, keep the random-MLP target.
3. **Always evaluate sparse and dense separately**; model selection on dense
   picks the wrong model for our mission.
4. **hgbt is a strong mid/dense baseline** (global refit, low variance) but
   can't extrapolate at N=3 — its strength defines our value boundary.
5. TabICL's pip package ships its full prior (`tabicl.prior`) — usable for SCM
   swaps (BSD-3), classification-only by default (`Reg2Cls`), monkeypatch in
   `tabicl_prior.py` keeps it regression.
6. Report i18n: keep CJK inner quotes fullwidth (「」“”) inside JS strings;
   validate with the node parse + key-parity + mock-DOM pattern (see the
   validation snippets in this session's history / reuse from any report).
7. Artifacts: republish same file path in the same conversation keeps the URL;
   from a new conversation pass `url:`.

## 7. Reproduce anything

```bash
# dense holdout for a checkpoint
uv run python -m geo_pfn.haneda.holdout_eval --checkpoint <ckpt> --device cuda \
  --models geopfn --feature-sets L,LCSG --out results/haneda/holdout_<name>.json
# sparse sweep
uv run --with tabicl python -m geo_pfn.haneda.sparse_context --checkpoint <ckpt> \
  --device cuda --context-sizes 3,6,12,25,50,100,-1 --n-seeds 4 \
  --feature-sets L,LCSG --models geopfn --out results/haneda/sparse_<name>.json
# uncertainty calibration
uv run python -m geo_pfn.haneda.uncertainty_eval --checkpoint <ckpt> --device cuda
# training (generic geo-SCM, prefetch)
uv run python -m geo_pfn.geopfn.train --steps 24000 --seed 0 --p-geo-realistic 0.0 \
  --batch-size 48 --num-workers 12 --out checkpoints/<name>.pt
# training on TabICL's SCM (ablation)
uv run --with tabicl --with xgboost python -m geo_pfn.geopfn.train --prior tabicl ...
```
