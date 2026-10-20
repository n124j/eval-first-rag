import ast
import json
import math
from pathlib import Path

import pytest

import evalrag.evals.gate as gate_mod
from evalrag.evals.gate import (
    Baseline,
    BaselineError,
    MetricBaseline,
    MetricsError,
    Regression,
    compare,
    dump_metrics,
    format_regressions,
    gate,
    load_baseline,
    load_metrics,
)

NAN = float("nan")


def mb(value: float, tolerance: float, direction: str = "higher") -> MetricBaseline:
    return MetricBaseline(value, tolerance, direction)


def base(**metrics: MetricBaseline) -> Baseline:
    return Baseline("m", "2026-01-01", "why", dict(metrics))


def good_doc() -> dict:
    return {
        "model": "m",
        "date": "2026-01-01",
        "reason": "initial",
        "metrics": {
            "recall_at_k": {"value": 0.8, "tolerance": 0.05, "direction": "higher"},
            "latency_p95_s": {"value": 2.0, "tolerance": 0.5, "direction": "lower"},
        },
    }


def write_doc(tmp_path: Path, doc: object) -> Path:
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_within_tolerance_passes() -> None:
    b = base(recall_at_k=mb(0.8, 0.05))
    assert compare({"recall_at_k": 0.78}, b) == []
    assert compare({"recall_at_k": 0.95}, b) == []


def test_exactly_on_tolerance_passes_despite_float_error() -> None:
    b = base(recall_at_k=mb(0.8, 0.05), latency_p95_s=mb(0.1, 0.2, "lower"))
    assert compare({"recall_at_k": 0.75, "latency_p95_s": 0.1 + 0.2}, b) == []


def test_worse_than_tolerance_fails_for_higher_is_better() -> None:
    b = base(recall_at_k=mb(0.8, 0.05))
    (r,) = compare({"recall_at_k": 0.74}, b)
    assert r == Regression("recall_at_k", 0.8, 0.74, 0.05, "higher", "worse than baseline")


def test_lower_is_better_direction_for_latency() -> None:
    b = base(latency_p95_s=mb(2.0, 0.5, "lower"))
    assert compare({"latency_p95_s": 2.4}, b) == []
    assert compare({"latency_p95_s": 0.1}, b) == []
    (r,) = compare({"latency_p95_s": 2.6}, b)
    assert r.reason == "worse than baseline"


def test_missing_nan_inf_metrics_fail() -> None:
    b = base(a=mb(0.5, 0.1), b=mb(0.5, 0.1), c=mb(0.5, 0.1, "lower"), d=mb(0.5, 0.1))
    regs = compare({"b": NAN, "c": math.inf, "d": -math.inf}, b)
    assert [(r.name, r.reason) for r in regs] == [
        ("a", "missing from metrics"),
        ("b", "not finite"),
        ("c", "not finite"),
        ("d", "not finite"),
    ]
    assert math.isnan(regs[0].actual)


def test_extra_metrics_ignored() -> None:
    assert compare({"recall_at_k": 0.8, "other": NAN}, base(recall_at_k=mb(0.8, 0.0))) == []


def test_bad_direction_rejected(tmp_path: Path) -> None:
    doc = good_doc()
    doc["metrics"]["recall_at_k"]["direction"] = "up"
    with pytest.raises(BaselineError, match="recall_at_k.*direction"):
        load_baseline(write_doc(tmp_path, doc))


def test_load_baseline_ok(tmp_path: Path) -> None:
    b = load_baseline(write_doc(tmp_path, good_doc()))
    assert b.model == "m"
    assert b.metrics["latency_p95_s"] == MetricBaseline(2.0, 0.5, "lower")


def test_load_baseline_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="nope.json"):
        load_baseline(tmp_path / "nope.json")


def test_load_baseline_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "b.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(BaselineError):
        load_baseline(p)


def test_baseline_rejects_non_object_and_extra_or_missing_keys(tmp_path: Path) -> None:
    with pytest.raises(BaselineError):
        load_baseline(write_doc(tmp_path, [1]))
    doc = good_doc()
    doc["extra"] = 1
    with pytest.raises(BaselineError, match="extra"):
        load_baseline(write_doc(tmp_path, doc))
    doc = good_doc()
    del doc["reason"]
    with pytest.raises(BaselineError, match="reason"):
        load_baseline(write_doc(tmp_path, doc))


@pytest.mark.parametrize("field", ["model", "date", "reason"])
@pytest.mark.parametrize("bad", ["", "  ", 3, None])
def test_baseline_rejects_bad_text_fields(tmp_path: Path, field: str, bad: object) -> None:
    doc = good_doc()
    doc[field] = bad
    with pytest.raises(BaselineError, match=field):
        load_baseline(write_doc(tmp_path, doc))


@pytest.mark.parametrize("bad", [{}, [], "x", None])
def test_baseline_rejects_bad_metrics_container(tmp_path: Path, bad: object) -> None:
    doc = good_doc()
    doc["metrics"] = bad
    with pytest.raises(BaselineError, match="metrics"):
        load_baseline(write_doc(tmp_path, doc))


@pytest.mark.parametrize("field", ["value", "tolerance"])
@pytest.mark.parametrize("bad", [True, "0.5", None, [0.5]])
def test_baseline_rejects_non_numeric_value_tolerance(
    tmp_path: Path, field: str, bad: object
) -> None:
    doc = good_doc()
    doc["metrics"]["recall_at_k"][field] = bad
    with pytest.raises(BaselineError, match=f"recall_at_k.*{field}"):
        load_baseline(write_doc(tmp_path, doc))


@pytest.mark.parametrize("field", ["value", "tolerance"])
@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_baseline_rejects_non_finite(tmp_path: Path, field: str, token: str) -> None:
    doc = good_doc()
    text = json.dumps(doc).replace("0.8" if field == "value" else "0.05", token, 1)
    p = tmp_path / "b.json"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(BaselineError, match=f"recall_at_k.*{field}"):
        load_baseline(p)


def test_baseline_rejects_negative_tolerance(tmp_path: Path) -> None:
    doc = good_doc()
    doc["metrics"]["recall_at_k"]["tolerance"] = -0.1
    with pytest.raises(BaselineError, match="recall_at_k.*tolerance"):
        load_baseline(write_doc(tmp_path, doc))


def test_baseline_rejects_bad_metric_entry_shape(tmp_path: Path) -> None:
    doc = good_doc()
    doc["metrics"]["recall_at_k"] = {"value": 0.8, "tolerance": 0.1}
    with pytest.raises(BaselineError, match="recall_at_k"):
        load_baseline(write_doc(tmp_path, doc))
    doc["metrics"]["recall_at_k"] = 0.8
    with pytest.raises(BaselineError, match="recall_at_k"):
        load_baseline(write_doc(tmp_path, doc))


def test_format_regressions_is_deterministic() -> None:
    regs = [Regression("a", 0.8, 0.7, 0.05, "higher", "worse than baseline")]
    out = format_regressions(regs)
    assert out == format_regressions(regs)
    assert out.count("\n") <= 1
    for token in ("a", "higher", "0.8", "0.7", "0.05", "worse than baseline"):
        assert token in out


def test_gate_returns_1_and_prints_regressions_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_doc(tmp_path, good_doc())
    code = gate({"recall_at_k": 0.5, "latency_p95_s": 2.0}, path)
    captured = capsys.readouterr()
    assert code == 1
    assert "recall_at_k" in captured.err
    assert "gate passed" not in captured.out


def test_gate_returns_0_on_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_doc(tmp_path, good_doc())
    assert gate({"recall_at_k": 0.8, "latency_p95_s": 2.0}, path) == 0
    captured = capsys.readouterr()
    assert "gate passed" in captured.out
    assert captured.err == ""


def test_gate_propagates_baseline_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gate({}, tmp_path / "missing.json")


def test_dump_load_metrics_roundtrip_with_nan_as_null(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    dump_metrics({"b": 1.5, "a": NAN, "c": math.inf, "d": 3.0}, path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw == {"a": None, "b": 1.5, "c": None, "d": 3.0}
    loaded = load_metrics(path)
    assert math.isnan(loaded["a"]) and math.isnan(loaded["c"])
    assert loaded["b"] == 1.5 and loaded["d"] == 3.0


def test_dump_metrics_is_atomic_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "m.json"
    dump_metrics({"a": 1.0}, path)
    before = path.read_bytes()
    with pytest.raises(TypeError):
        dump_metrics({"a": object()}, path)  # type: ignore[dict-item]
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m.json"]


def test_dump_metrics_bytes_are_stable(tmp_path: Path) -> None:
    p1, p2 = tmp_path / "1.json", tmp_path / "2.json"
    metrics = {"z": 1.0, "a": NAN, "m": 0.25}
    dump_metrics(metrics, p1)
    dump_metrics(dict(reversed(list(metrics.items()))), p2)
    assert p1.read_bytes() == p2.read_bytes()
    data = p1.read_bytes()
    assert b"\r" not in data and data.endswith(b"\n")
    assert data.index(b'"a"') < data.index(b'"m"') < data.index(b'"z"')


@pytest.mark.parametrize("bad", ['{"a": true}', '{"a": "1"}', '{"a": [1]}', '{"a": {"b": 1}}'])
def test_load_metrics_rejects_non_numeric(tmp_path: Path, bad: str) -> None:
    p = tmp_path / "m.json"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(MetricsError, match="'a'"):
        load_metrics(p)


def test_load_metrics_rejects_invalid_json_and_non_object(tmp_path: Path) -> None:
    p = tmp_path / "m.json"
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(MetricsError):
        load_metrics(p)
    p.write_text("[1]", encoding="utf-8")
    with pytest.raises(MetricsError):
        load_metrics(p)
    with pytest.raises(FileNotFoundError, match="gone.json"):
        load_metrics(tmp_path / "gone.json")


def _imported_modules(module_file: str) -> set[str]:
    tree = ast.parse(Path(module_file).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    return names


def test_gate_module_does_not_import_judge() -> None:
    mods = _imported_modules(gate_mod.__file__)
    assert not any("judge" in m for m in mods)
