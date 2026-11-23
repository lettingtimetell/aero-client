"""Read frame-time CSV samples and produce the optimization-plan summary.

The CSV schema is ``frame_index,frame_ms`` with optional ``phase`` and ``event``
columns.  Percentiles use linear interpolation on the inclusive ``(n - 1) * q``
rank (the two surrounding sorted observations are blended).  Standard deviation
is the population standard deviation.  These choices are part of the artifact
contract so summaries from different runners remain comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Iterable, TextIO


class FrameMetricsError(ValueError):
    """Raised when a frame CSV is empty or violates the telemetry schema."""


@dataclass(frozen=True)
class FrameSample:
    """One chronological frame interval and its optional labels."""

    frame_index: int
    frame_ms: float
    phase: str | None = None
    event: str | None = None

    def as_dict(self) -> dict[str, object]:
        """Return the stable JSON representation, retaining supplied labels."""

        value: dict[str, object] = {
            "frame_index": self.frame_index,
            "frame_ms": self.frame_ms,
        }
        if self.phase is not None:
            value["phase"] = self.phase
        if self.event is not None:
            value["event"] = self.event
        return value


def _parse_index(raw: str, row: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise FrameMetricsError(f"row {row}: frame_index must be an integer") from exc
    return value


def _parse_time(raw: str, row: int) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise FrameMetricsError(f"row {row}: frame_ms must be a finite positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise FrameMetricsError(f"row {row}: frame_ms must be a finite positive number")
    return value


def read_frames(source: TextIO | Iterable[str]) -> list[FrameSample]:
    """Parse frame CSV rows, rejecting missing, malformed, or dropped samples."""

    reader = csv.DictReader(source, strict=True)
    try:
        if reader.fieldnames is None:
            raise FrameMetricsError("CSV is missing a header")
        required = {"frame_index", "frame_ms"}
        fields = set(reader.fieldnames)
        if not required.issubset(fields):
            missing = ", ".join(sorted(required - fields))
            raise FrameMetricsError(f"CSV is missing required column(s): {missing}")
        allowed = required | {"phase", "event"}
        unknown = fields - allowed
        if unknown:
            raise FrameMetricsError(f"CSV has unsupported column(s): {', '.join(sorted(unknown))}")
        if len(fields) != len(reader.fieldnames):
            raise FrameMetricsError("CSV header contains duplicate columns")

        samples: list[FrameSample] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise FrameMetricsError(f"row {row_number}: too many columns")
            if row.get("frame_index") is None or row.get("frame_ms") is None:
                raise FrameMetricsError(f"row {row_number}: missing required value")
            for optional in ("phase", "event"):
                if optional in fields and row.get(optional) is None:
                    raise FrameMetricsError(f"row {row_number}: missing {optional} value")
            index = _parse_index(row["frame_index"], row_number)
            if samples and index != samples[-1].frame_index + 1:
                raise FrameMetricsError(
                    f"row {row_number}: frame_index {index} is not contiguous after "
                    f"{samples[-1].frame_index}"
                )
            phase = row.get("phase")
            event = row.get("event")
            samples.append(FrameSample(index, _parse_time(row["frame_ms"], row_number), phase, event))
    except csv.Error as exc:
        raise FrameMetricsError(f"malformed CSV: {exc}") from exc
    if not samples:
        raise FrameMetricsError("CSV contains no frame samples")
    return samples


def _percentile(sorted_values: list[float], quantile: float) -> float:
    rank = (len(sorted_values) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def summarize(samples: Iterable[FrameSample]) -> dict[str, object]:
    """Return metrics and raw samples for one complete chronological interval."""

    values = list(samples)
    if not values:
        raise FrameMetricsError("cannot summarize an empty sample set")
    for position, sample in enumerate(values):
        if not isinstance(sample, FrameSample):
            raise FrameMetricsError(f"sample {position}: expected FrameSample")
        if isinstance(sample.frame_index, bool) or not isinstance(sample.frame_index, int):
            raise FrameMetricsError(f"sample {position}: frame_index must be an integer")
        if position and sample.frame_index != values[position - 1].frame_index + 1:
            raise FrameMetricsError(f"sample {position}: frame_index is not contiguous")
        if isinstance(sample.frame_ms, bool) or not isinstance(sample.frame_ms, Real):
            raise FrameMetricsError(f"sample {position}: frame_ms must be a finite positive number")
        if not math.isfinite(sample.frame_ms) or sample.frame_ms <= 0.0:
            raise FrameMetricsError(f"sample {position}: frame_ms must be a finite positive number")
    times = [sample.frame_ms for sample in values]
    ordered = sorted(times)
    count = len(times)
    try:
        mean = math.fsum(times) / count
        duration = math.fsum(times)
        variance = math.fsum((value - mean) ** 2 for value in times) / count
        tail_count = math.ceil(count * 0.01)
        slowest_mean = math.fsum(sorted(times, reverse=True)[:tail_count]) / tail_count
        metrics = {
            "duration_ms": duration,
            "average_fps": 1000.0 / mean,
            "mean_frame_ms": mean,
            "p50_frame_ms": _percentile(ordered, 0.50),
            "p95_frame_ms": _percentile(ordered, 0.95),
            "p99_frame_ms": _percentile(ordered, 0.99),
            "p99_9_frame_ms": _percentile(ordered, 0.999),
            "max_frame_ms": max(times),
            "stddev_frame_ms": math.sqrt(variance),
            "one_percent_low_fps": 1000.0 / slowest_mean,
        }
    except (OverflowError, ZeroDivisionError) as exc:
        raise FrameMetricsError("frame samples produce a non-finite summary") from exc
    if any(not math.isfinite(value) for value in metrics.values()):
        raise FrameMetricsError("frame samples produce a non-finite summary")
    return {
        "sample_count": count,
        **metrics,
        "samples": [sample.as_dict() for sample in values],
    }


def summarize_csv(path: str | Path) -> dict[str, object]:
    """Read *path* and summarize it."""

    try:
        with Path(path).open("r", encoding="utf-8", newline="") as stream:
            return summarize(read_frames(stream))
    except UnicodeError as exc:
        raise FrameMetricsError(f"input is not valid UTF-8: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``frame_metrics.py INPUT.csv [-o OUTPUT.json]``."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="frame CSV input")
    parser.add_argument("-o", "--output", type=Path, help="JSON output path (default: stdout)")
    args = parser.parse_args(argv)
    try:
        result = summarize_csv(args.input)
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output:
            if args.output.resolve() == args.input.resolve():
                raise FrameMetricsError("output path must differ from input path")
            with args.output.open("x", encoding="utf-8", newline="") as stream:
                stream.write(encoded)
        else:
            sys.stdout.write(encoded)
    except (OSError, UnicodeError, FrameMetricsError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
