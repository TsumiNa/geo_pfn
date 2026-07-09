"""Two-stage geo-PFN: column-embed then row-compare, for borehole similarity.

Tests the hypothesis (docs/geo-scm-design.md §9) that a model whose architecture
explicitly compresses rows and compares them (a learned row metric / soft kNN)
exploits sparse-anchor cross-borehole transfer better than generic per-cell ICL.
"""
