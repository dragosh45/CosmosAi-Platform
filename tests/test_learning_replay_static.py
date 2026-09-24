"""Static contract checks for the local HTML/CSS/JavaScript learning replay."""

import json
from pathlib import Path

import pytest

from scripts.build_learning_replay import build_replay


REPLAY_DIR = Path("apps/learning-replay")


def load_bundled_trace() -> dict:
    source = (REPLAY_DIR / "sample_trace.js").read_text(encoding="utf-8")
    prefix = "window.COSMOSAI_SAMPLE_TRACE = "
    assert source.startswith(prefix)
    return json.loads(source[len(prefix):].rstrip().removesuffix(";"))


def test_learning_replay_has_static_entrypoint_and_required_sections():
    html = (REPLAY_DIR / "index.html").read_text(encoding="utf-8")
    javascript = (REPLAY_DIR / "viewer.js").read_text(encoding="utf-8")
    stylesheet = (REPLAY_DIR / "viewer.css").read_text(encoding="utf-8")

    assert '<script src="sample_trace.js" defer></script>' in html
    assert '<script src="viewer.js" defer></script>' in html
    assert 'id="site-nav" class="site-nav" aria-label="Main navigation" hidden' in html
    assert 'window.location.protocol' in javascript
    assert '.site-nav[hidden] { display: none; }' in stylesheet
    assert 'href="#input-planes">Kernel slices' in html
    assert 'href="#parameter-inventory">Exact weights' in html
    assert 'href="#training-metrics">Whole training cycle' in html
    assert "Kernel slices = learned weights" in html
    for section_id in (
        "input-planes",
        "input-plane-values",
        "window-explanation",
        "window-formula",
        "pooled-summary",
        "product-explanation",
        "products-table",
        "network-diagram",
        "classifier-schematic",
        "classifier-connection-example",
        "classifier-general-formula",
        "classifier-order",
        "softmax-explanation",
        "parameter-equation",
        "parameter-meaning",
        "parameter-inventory",
        "gradient-diagram",
        "update-overview",
        "training-metrics",
        "metric-cards",
        "loss-walkthrough",
    ):
        assert f'id="{section_id}"' in html
    assert "function renderPlaneSection" in javascript
    assert "input = (output row + k - padding, output column + j - padding)" in javascript
    assert "z[i] = sum over j of (W[i,j] x a[j]) + b[i]" in javascript
    assert "27 input x weight products" in javascript
    assert "each channel subtotal = its nine products added together" in javascript
    assert "class logits z[i] -> softmax p[i]" in javascript
    assert "Global average pooling" in javascript
    assert "formatPlain(sum) / 9" not in javascript
    assert "a[${index}] = ${formatPlain(sum)} / 9" in javascript
    assert "Selected response" in javascript
    assert "After-ReLU 3x3 values" in javascript
    assert "red + green + blue + bias" in javascript
    assert "Gradient calculation for this shared convolution weight" in javascript
    assert "This is not a sum of image pixels" in javascript
    assert "means the convolution response z_conv" in javascript
    assert "The final gradient is not one row" in javascript
    assert "gradient-focus" in javascript
    assert "Cross-entropy loss before this optimizer step" in javascript
    assert "dL/dz = (p - one_hot(target))" in javascript
    assert "pooled-summary-band" in stylesheet
    assert "Softmax for" in javascript
    assert "weight-label" in javascript
    assert "W x a + b = z" in javascript
    assert "softmax(z) = p" in javascript
    assert "function renderConvolutionGradient" in javascript
    assert "function renderClassifierGradient" in javascript
    assert "function parameterMeaning" in javascript
    assert "filter/output channel" in javascript
    assert "The last two indices are kernel coordinates" in javascript
    assert "other convolution weights are summarized" in javascript
    assert "A: 9 map positions + B: 9 map positions" in javascript
    assert "How this selected scalar changes" in javascript
    assert "delta = -learning_rate x gradient" in javascript
    assert "new parameter = old parameter + delta" in javascript
    assert "old parameter = value before SGD" in javascript
    assert "Whole model update: every stored scalar gets its own gradient" in javascript
    assert "new = old - lr x gradient" in javascript
    assert "the other ${totalScalars - 1} scalars use their own gradients" in javascript
    assert "Full update for filter ${filterIndex}: all 3 x 3 x 3 convolution weights" in javascript
    assert "old, then &rarr; new, then &Delta;" in javascript
    assert "channel totals sum the nine gradients or nine deltas" in javascript
    assert "@media (max-width: 560px)" in stylesheet


def test_bundled_trace_is_the_measured_version_two_example():
    trace = load_bundled_trace()
    first_update = trace["updates"][0]
    selected = first_update["selected_convolution_gradient"]

    assert trace["format_version"] == 2
    assert trace["parameter_count"] == 132
    assert len(trace["updates"]) == 2
    assert len(selected["contributions"]) == 18
    assert selected["matches_autograd"] is True
    assert selected["sum_of_contributions"] == pytest.approx(0.014876489, abs=1e-7)


def test_report_builder_creates_a_movable_self_contained_report(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(load_bundled_trace()), encoding="utf-8")
    report_path = build_replay(
        trace_path,
        tmp_path / "learning-report",
    )
    assert report_path.name == "index.html"
    for filename in ("index.html", "viewer.js", "viewer.css", "sample_trace.js"):
        assert (tmp_path / "learning-report" / filename).is_file()
    assert "window.COSMOSAI_SAMPLE_TRACE" in (
        tmp_path / "learning-report" / "sample_trace.js"
    ).read_text(encoding="utf-8")
