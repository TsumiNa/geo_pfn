# geo_pfn.geopfn

Two-stage geo-PFN: a column-encode → row-compress → row-metric in-context
regressor for borehole data, with a bar-distribution head (value + uncertainty).
Design and results: `docs/geo-scm-design.md`, `docs/geopfn-hypothesis-results.md`,
`docs/model-comparison.md`.

## Modules

| file | role |
|---|---|
| `model.py` | `GeoPFN` + `GeoPFNConfig` — the two-stage architecture |
| `heads.py` | `BarDistribution` — regression head (mean + std over fixed bins) |
| `train.py` | training on the geo-SCM site prior (`geo_pfn.geoprior`) |
| `predict.py` | `predict_geopfn` — ensemble inference over context subsamples |
| `eval_anchor.py` | sparse-anchor evaluation on real Haneda data |

## CLI

```bash
# train (MPS; ~5-6h for 8000 steps)
uv run python -m geo_pfn.geopfn.train --steps 8000 --out checkpoints/geopfn2stage.pt

# evaluate on the real Haneda anchor task
uv run python -m geo_pfn.geopfn.eval_anchor \
    --checkpoint checkpoints/geopfn2stage.pt --feature-set LCSG --target Su \
    --out results/haneda/anchor_geopfn_su_lcsg.json
```

## Checkpoints

Saved as **state dict + config**, never a whole-model pickle
(`geo_pfn.util.save_checkpoint`): `{model_state, model_config, prior_config,
train_config, step, history}`. Reload with
`GeoPFN(GeoPFNConfig(**ckpt["model_config"])).load_state_dict(ckpt["model_state"])`
(see `eval_anchor.load_geopfn`). Refactoring model code never breaks old
checkpoints. Checkpoints are gitignored; retrain to reproduce.

## Evidence chain

Every evaluation run is self-describing so results are reproducible from the
artifact alone:

- **model config**: `model_config` in the checkpoint (`.pt`).
- **training log**: `history` in the checkpoint (per-log-step loss / rmse_norm / lr).
- **data split**: recorded in each eval JSON's `config.split`
  (`GroupKFold(borehole, n_splits, seed)`) — deterministic from the seed.
- **raw results**: per-fold, per-k records in the eval JSON `records`.
- **analysis**: aggregated tables under `results/haneda/analysis/`.

When adding a new evaluation, keep this contract: write `{config, records}` with
enough config to re-run (checkpoint path, feature set, target, split, ensemble).
