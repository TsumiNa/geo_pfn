"""Tests for the sparse-demo packer (no model, no browser)."""

from pathlib import Path

import numpy as np
import pytest

from geo_pfn.haneda.sparse_demo import quantize, render_html


def test_quantize_round_trip_error_within_half_step() -> None:
    lo, hi = 10.0, 200.0
    values = np.random.default_rng(0).uniform(lo, hi, 10_000)
    q = quantize(values, lo, hi)
    recovered = lo + q.astype(np.float64) / 255.0 * (hi - lo)
    assert q.dtype == np.uint8
    assert np.abs(recovered - values).max() <= (hi - lo) / 255.0 / 2 + 1e-9


def test_quantize_clips_out_of_range() -> None:
    q = quantize(np.array([-5.0, 500.0]), 0.0, 100.0)
    assert q.tolist() == [0, 255]


def test_render_html_injects_both_markers(tmp_path: Path) -> None:
    template = tmp_path / "t.html"
    template.write_text("<script>/*__THREE__*/</script><script>const D=/*__DATA__*/;</script>")
    three = tmp_path / "three.js"
    three.write_text("var THREE={};")
    html = render_html({"a": 1}, template, three)
    assert "var THREE={};" in html
    assert 'const D={"a":1};' in html
    assert "__THREE__" not in html and "__DATA__" not in html


def test_render_html_rejects_script_close_in_bundle(tmp_path: Path) -> None:
    template = tmp_path / "t.html"
    template.write_text("/*__THREE__*/ /*__DATA__*/")
    three = tmp_path / "three.js"
    three.write_text("bad</script>bundle")
    with pytest.raises(ValueError, match="script"):
        render_html({}, template, three)
