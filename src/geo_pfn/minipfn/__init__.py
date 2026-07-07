"""Mini-PFN: a small TabPFN-style in-context learner robust to missing test features.

This package is a self-contained end-to-end demo:

- ``config``: dataclass configs for the prior, the model, and training.
- ``prior``: synthetic classification-task generator (random-MLP / SCM prior),
  including the test-time feature-dropout corruption used for training and eval.
- ``model``: per-cell tokenized transformer with alternating feature-axis and
  sample-axis attention (TabPFN v2 style), native NaN handling.
- ``train``: pretraining loop over synthetic tasks (``python -m geo_pfn.minipfn.train``).
- ``eval``: evaluation of missing-feature strategies vs. baselines
  (``python -m geo_pfn.minipfn.eval``).
"""
