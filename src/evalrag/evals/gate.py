"""Regression gate: compare measured metrics with a committed baseline; fail closed.

The gate reads only the metrics and the baseline. It never looks at LLM-judge scores.
"""

import contextlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

DIRECTIONS = ("higher", "lower")
_BASELINE_KEYS = {"model", "date", "reason", "metrics"}
_METRIC_KEYS = {"value", "tolerance", "direction"}
# Float slack so a value exactly on the tolerance never fails from rounding error.
_SLACK = 1e-9

NAN = float("nan")


class BaselineError(ValueError):
    """The baseline file is malformed."""


class MetricsError(ValueError):
    """A metrics file is malformed."""


@dataclass(frozen=True)
class MetricBaseline:
    value: float
    tolerance: float
    direction: str


@dataclass(frozen=True)
class Baseline:
    model: str
    date: str
    reason: str
    metrics: dict[str, MetricBaseline]


@dataclass(frozen=True)
class Regression:
    name: str
    baseline: float
    actual: float
    tolerance: float
    direction: str
    reason: str


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _read_json(path: Path, error: type[ValueError]) -> object:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise error(f"{path}: invalid JSON: {exc}") from exc


def _text(raw: dict[str, object], field: str) -> str:
    value = raw[field]
    if not isinstance(value, str) or not value.strip():
        raise BaselineError(f"{field} must be a non-empty string")
    return value


def _metric(name: str, raw: object) -> MetricBaseline:
    if not isinstance(raw, dict):
        raise BaselineError(f"metric {name!r}: expected an object")
    if set(raw) != _METRIC_KEYS:
        missing = sorted(_METRIC_KEYS - set(raw))
        extra = sorted(set(raw) - _METRIC_KEYS)
        raise BaselineError(
            f"metric {name!r}: fields must be exactly {sorted(_METRIC_KEYS)}; "
            f"missing {missing}, extra {extra}"
        )
    value, tolerance, direction = raw["value"], raw["tolerance"], raw["direction"]
    if not _is_number(value) or not math.isfinite(value):
        raise BaselineError(f"metric {name!r} field 'value' must be a finite number")
    if not _is_number(tolerance) or not math.isfinite(tolerance) or tolerance < 0:
        raise BaselineError(f"metric {name!r} field 'tolerance' must be a finite number >= 0")
    if direction not in DIRECTIONS:
        raise BaselineError(
            f"metric {name!r} field 'direction' must be one of {list(DIRECTIONS)}, "
            f"got {direction!r}"
        )
    return MetricBaseline(float(value), float(tolerance), direction)


def load_baseline(path: Path) -> Baseline:
    """Read and strictly validate a baseline file."""
    raw = _read_json(path, BaselineError)
    if not isinstance(raw, dict):
        raise BaselineError("baseline must be a JSON object")
    if set(raw) != _BASELINE_KEYS:
        missing = sorted(_BASELINE_KEYS - set(raw))
        extra = sorted(set(raw) - _BASELINE_KEYS)
        raise BaselineError(
            f"baseline keys must be exactly {sorted(_BASELINE_KEYS)}; "
            f"missing {missing}, extra {extra}"
        )
    metrics_raw = raw["metrics"]
    if not isinstance(metrics_raw, dict) or not metrics_raw:
        raise BaselineError("metrics must be a non-empty object")
    return Baseline(
        _text(raw, "model"),
        _text(raw, "date"),
        _text(raw, "reason"),
        {name: _metric(name, entry) for name, entry in metrics_raw.items()},
    )


def compare(metrics: Mapping[str, float], baseline: Baseline) -> list[Regression]:
    """Regressions of `metrics` against `baseline`, in sorted metric-name order.

    A metric that is missing, NaN or infinite counts as a regression. Metrics absent from
    the baseline are ignored.
    """
    regressions: list[Regression] = []
    for name in sorted(baseline.metrics):
        base = baseline.metrics[name]

        def reg(actual: float, reason: str, name: str = name, base: MetricBaseline = base) -> None:
            regressions.append(
                Regression(name, base.value, actual, base.tolerance, base.direction, reason)
            )

        if name not in metrics:
            reg(NAN, "missing from metrics")
            continue
        actual = metrics[name]
        if not math.isfinite(actual):
            reg(actual, "not finite")
        else:
            floor = base.value - base.tolerance - _SLACK
            ceiling = base.value + base.tolerance + _SLACK
            worse = actual < floor if base.direction == "higher" else actual > ceiling
            if worse:
                reg(actual, "worse than baseline")
    return regressions


def format_regressions(regressions: list[Regression]) -> str:
    """One deterministic line per regression."""
    return "\n".join(
        f"REGRESSION {r.name} ({r.direction} is better): baseline={r.baseline!r} "
        f"actual={r.actual!r} tolerance={r.tolerance!r} reason={r.reason}"
        for r in regressions
    )


def gate(metrics: Mapping[str, float], baseline_path: Path) -> int:
    """Return 0 when every baseline metric holds, 1 on any regression. Errors propagate."""
    baseline = load_baseline(baseline_path)
    regressions = compare(metrics, baseline)
    if regressions:
        print(format_regressions(regressions), file=sys.stderr)
        return 1
    print(f"gate passed: {len(baseline.metrics)} metrics within tolerance")
    return 0


def dump_metrics(metrics: Mapping[str, float], path: Path) -> None:
    """Write metrics as stable JSON (sorted keys, NaN/inf as null), atomically."""
    clean = {
        name: (value if math.isfinite(value) else None) for name, value in sorted(metrics.items())
    }
    data = (json.dumps(clean, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    path = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def load_metrics(path: Path) -> dict[str, float]:
    """Read a metrics file written by `dump_metrics`; null becomes NaN."""
    raw = _read_json(path, MetricsError)
    if not isinstance(raw, dict):
        raise MetricsError("metrics file must be a JSON object")
    out: dict[str, float] = {}
    for name, value in raw.items():
        if value is None:
            out[name] = NAN
        elif _is_number(value):
            out[name] = float(value)
        else:
            raise MetricsError(f"metric {name!r}: expected a number or null, got {value!r}")
    return out
