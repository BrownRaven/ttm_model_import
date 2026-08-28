"""Create bounded event-near feature extracts and descriptive summaries from large feature CSVs.

This tool scans each feature CSV once in chunks. It does not modify source files, infer
thresholds, label context windows as normal, or interpret association as causality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REVIEW_CONTRACT = "PIMS-TS-EVENT-REVIEW-v1"
TIMEZONE = ZoneInfo("Asia/Seoul")
ASSET_SCOPE = {"M-317": "3S", "M-319": "3S", "M-417": "4S", "M-419": "4S"}
IDENTIFIER_COLUMNS = [
    "dataset_id", "window_start", "window_end", "sensor_uid", "asset_id",
    "measurement_type", "unit", "valid_sample_count", "missing_sample_count",
    "longest_missing_run_seconds", "coverage_ratio", "quality_status", "spectral_status",
]
METRIC_COLUMNS = [
    "mean", "std", "min", "max", "median", "p05", "p95", "range", "iqr", "rms",
    "mad", "skewness", "excess_kurtosis", "slope_per_second", "first_last_delta",
    "mean_abs_change", "max_abs_change", "mean_abs_rate_per_second",
    "max_abs_rate_per_second", "flatline_step_ratio", "robust_outlier_ratio",
    "autocorrelation_lag_1s", "autocorrelation_lag_5s", "autocorrelation_lag_30s",
    "spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_entropy",
    "dominant_frequency_hz", "dominant_power_ratio", "relative_power_0_10pct_nyquist",
    "relative_power_10_30pct_nyquist", "relative_power_30_100pct_nyquist",
]
MARKDOWN_METRICS = {
    "mean", "std", "max", "range", "slope_per_second", "first_last_delta",
    "max_abs_change", "robust_outlier_ratio", "spectral_entropy", "dominant_power_ratio",
}


class EventReviewError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class PriorityEvent:
    priority: int
    score_rank: int
    event_uid: str
    asset_id: str
    event_period: str
    core_time: str
    score: int
    summary: str
    source_ref: str


@dataclass(frozen=True)
class EventOccurrence:
    event: PriorityEvent
    occurrence_uid: str
    start: datetime
    end: datetime
    precision: str


def _clean_cell(value: str) -> str:
    return value.strip().replace("**", "").strip("`")


def parse_priority_events(path: Path) -> list[PriorityEvent]:
    lines = path.expanduser().resolve().read_text(encoding="utf-8").splitlines()
    header: list[str] | None = None
    events: list[PriorityEvent] = []
    for line in lines:
        if not line.lstrip().startswith("|"):
            continue
        cells = [_clean_cell(cell) for cell in line.strip().strip("|").split("|")]
        if "권장 추출 순서" in cells and "ID" in cells:
            header = cells
            continue
        if header is None or len(cells) != len(header) or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        row = dict(zip(header, cells))
        if not row.get("권장 추출 순서", "").isdigit():
            continue
        asset_id = row["설비"] if row["설비"].startswith("M-") else row["설비"].replace("M", "M-", 1)
        events.append(PriorityEvent(
            priority=int(row["권장 추출 순서"]),
            score_rank=int(row["분석점수 순위"]),
            event_uid=row["ID"],
            asset_id=asset_id,
            event_period=row["사건기간"],
            core_time=row["핵심 시각"],
            score=int(row["점수"]),
            summary=row["사건 요약"],
            source_ref=f"{path}:{events.__len__() + 1}",
        ))
    if not events:
        raise EventReviewError("EVENT_TABLE_NOT_FOUND", str(path))
    return sorted(events, key=lambda event: event.priority)


def _local_datetime(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=TIMEZONE)


def expand_event_occurrences(event: PriorityEvent) -> list[EventOccurrence]:
    year_match = re.search(r"(\d{4})-", event.event_period)
    if not year_match:
        raise EventReviewError("EVENT_YEAR_INVALID", event.event_uid)
    default_year = int(year_match.group(1))
    precise_pattern = re.compile(
        r"(?:(\d{4})-)?(\d{2})-(\d{2})\s+(\d{2}):(\d{2})~(\d{2}):(\d{2})"
    )
    precise = list(precise_pattern.finditer(event.core_time)) if event.core_time != "-" else []
    occurrences: list[EventOccurrence] = []
    for index, match in enumerate(precise, start=1):
        year = int(match.group(1) or default_year)
        start = _local_datetime(year, int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5)))
        end = _local_datetime(year, int(match.group(2)), int(match.group(3)), int(match.group(6)), int(match.group(7)))
        if end <= start:
            end += timedelta(days=1)
        occurrences.append(EventOccurrence(event, f"{event.event_uid}#{index}", start, end, "EXACT_MINUTE"))
    if occurrences:
        return occurrences

    period_pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:~(?:(\d{4})-)?(\d{2})-(\d{2}))?")
    match = period_pattern.fullmatch(event.event_period)
    if not match:
        raise EventReviewError("EVENT_PERIOD_INVALID", f"{event.event_uid}: {event.event_period}")
    start = _local_datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if match.group(5):
        end_year = int(match.group(4) or match.group(1))
        end = _local_datetime(end_year, int(match.group(5)), int(match.group(6))) + timedelta(days=1)
    else:
        end = start + timedelta(days=1)
    return [EventOccurrence(event, f"{event.event_uid}#1", start, end, "DATE_RANGE_FULL_DAY")]


def select_occurrences(
    events: Iterable[PriorityEvent], top: int | None, selected_ids: set[str]
) -> list[EventOccurrence]:
    selected = [event for event in events if (event.priority <= top if top else True)]
    if selected_ids:
        known = {event.event_uid for event in events}
        missing = selected_ids - known
        if missing:
            raise EventReviewError("EVENT_ID_NOT_FOUND", "|".join(sorted(missing)))
        selected = [event for event in events if event.event_uid in selected_ids]
    unsupported = [event.event_uid for event in selected if event.asset_id not in ASSET_SCOPE]
    if unsupported:
        raise EventReviewError("EVENT_ASSET_UNSUPPORTED", "|".join(unsupported))
    return [occurrence for event in selected for occurrence in expand_event_occurrences(event)]


def _file_identity(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _phase(window_start: pd.Series, window_end: pd.Series, occurrence: EventOccurrence) -> pd.Series:
    event_start = pd.Timestamp(occurrence.start).tz_convert("UTC")
    event_end = pd.Timestamp(occurrence.end).tz_convert("UTC")
    values = np.full(len(window_start), "AFTER", dtype=object)
    values[window_end <= event_start] = "BEFORE"
    values[(window_start < event_end) & (window_end > event_start)] = "DURING"
    return pd.Series(values, index=window_start.index)


def scan_event_windows(
    feature_files: dict[str, Path],
    occurrences: list[EventOccurrence],
    output_path: Path,
    before_hours: float,
    after_hours: float,
    chunksize: int,
    max_selected_rows: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    selected_chunks: list[pd.DataFrame] = []
    source_rows = 0
    selected_rows = 0
    output_columns: list[str] | None = None
    by_scope = {
        scope: [occurrence for occurrence in occurrences if ASSET_SCOPE[occurrence.event.asset_id] == scope]
        for scope in feature_files
    }
    for scope, feature_path in feature_files.items():
        header = pd.read_csv(feature_path, nrows=0).columns.tolist()
        required = {"window_start", "window_end", "asset_id", "sensor_uid", "measurement_type"}
        if not required.issubset(header):
            raise EventReviewError("FEATURE_HEADER_INVALID", f"{feature_path}: {sorted(required - set(header))}")
        usecols = [column for column in [*IDENTIFIER_COLUMNS, *METRIC_COLUMNS] if column in header]
        output_columns = usecols
        for chunk in pd.read_csv(feature_path, usecols=usecols, chunksize=chunksize, low_memory=False):
            source_rows += len(chunk)
            starts = pd.to_datetime(chunk["window_start"], errors="coerce", utc=True)
            ends = pd.to_datetime(chunk["window_end"], errors="coerce", utc=True)
            if starts.isna().any() or ends.isna().any():
                raise EventReviewError("FEATURE_TIME_INVALID", str(feature_path))
            for occurrence in by_scope[scope]:
                context_start = pd.Timestamp(occurrence.start - timedelta(hours=before_hours)).tz_convert("UTC")
                context_end = pd.Timestamp(occurrence.end + timedelta(hours=after_hours)).tz_convert("UTC")
                mask = (
                    (chunk["asset_id"] == occurrence.event.asset_id)
                    & (starts < context_end)
                    & (ends > context_start)
                )
                if not mask.any():
                    continue
                selected = chunk.loc[mask].copy()
                selected_starts = starts.loc[mask]
                selected_ends = ends.loc[mask]
                selected.insert(0, "event_uid", occurrence.event.event_uid)
                selected.insert(1, "occurrence_uid", occurrence.occurrence_uid)
                selected.insert(2, "priority", occurrence.event.priority)
                selected.insert(3, "event_score", occurrence.event.score)
                selected.insert(4, "event_start", occurrence.start.isoformat())
                selected.insert(5, "event_end", occurrence.end.isoformat())
                selected.insert(6, "event_time_precision", occurrence.precision)
                selected.insert(7, "phase", _phase(selected_starts, selected_ends, occurrence).to_numpy())
                selected.insert(8, "context_label_status", "REFERENCE_CONTEXT_NOT_NORMAL_LABEL")
                midpoint = selected_starts + (selected_ends - selected_starts) / 2
                event_midpoint = pd.Timestamp(occurrence.start + (occurrence.end - occurrence.start) / 2).tz_convert("UTC")
                selected.insert(9, "minutes_from_event_midpoint", (midpoint - event_midpoint).dt.total_seconds() / 60.0)
                selected_chunks.append(selected)
                selected_rows += len(selected)
                if selected_rows > max_selected_rows:
                    raise EventReviewError("SELECTED_ROW_LIMIT_EXCEEDED", str(max_selected_rows))
    if selected_chunks:
        result = pd.concat(selected_chunks, ignore_index=True)
        result.sort_values(["priority", "occurrence_uid", "window_start", "sensor_uid"], inplace=True)
    else:
        result = pd.DataFrame(columns=[
            "event_uid", "occurrence_uid", "priority", "event_score", "event_start", "event_end",
            "event_time_precision", "phase", "context_label_status", "minutes_from_event_midpoint",
            *(output_columns or []),
        ])
    result.to_csv(output_path, index=False)
    return result, {"source_rows_scanned": source_rows, "selected_rows": selected_rows}


def summarize_event_windows(windows: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    if windows.empty:
        return pd.DataFrame()
    keys = [
        "event_uid", "occurrence_uid", "priority", "event_score", "event_start", "event_end",
        "event_time_precision", "asset_id", "sensor_uid", "measurement_type", "unit", "phase",
    ]
    rows: list[dict[str, Any]] = []
    baseline: dict[tuple[str, str, str], tuple[float, float]] = {}
    for (occurrence_uid, sensor_uid), group in windows.groupby(["occurrence_uid", "sensor_uid"], sort=False):
        before = group.loc[group["phase"] == "BEFORE"]
        for metric in metrics:
            values = pd.to_numeric(before[metric], errors="coerce").dropna()
            if values.empty:
                baseline[(occurrence_uid, sensor_uid, metric)] = (math.nan, math.nan)
            else:
                median = float(values.median())
                mad = float((values - median).abs().median())
                baseline[(occurrence_uid, sensor_uid, metric)] = (median, 1.4826 * mad)
    for group_key, group in windows.groupby(keys, dropna=False, sort=False):
        identity = dict(zip(keys, group_key))
        quality_pass_ratio = float((group["quality_status"] == "PASS").mean()) if "quality_status" in group else math.nan
        spectral_pass_ratio = float((group["spectral_status"] == "PASS").mean()) if "spectral_status" in group else math.nan
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            before_median, before_scale = baseline[(identity["occurrence_uid"], identity["sensor_uid"], metric)]
            median = float(values.median()) if not values.empty else math.nan
            delta = median - before_median if math.isfinite(median) and math.isfinite(before_median) else math.nan
            robust_shift = delta / before_scale if math.isfinite(delta) and math.isfinite(before_scale) and before_scale > 0 else math.nan
            rows.append({
                **identity,
                "metric": metric,
                "value_count": int(len(values)),
                "median": median,
                "p05": float(values.quantile(0.05)) if not values.empty else math.nan,
                "p95": float(values.quantile(0.95)) if not values.empty else math.nan,
                "min": float(values.min()) if not values.empty else math.nan,
                "max": float(values.max()) if not values.empty else math.nan,
                "before_median": before_median,
                "delta_from_before_median": delta,
                "robust_shift_vs_before": robust_shift,
                "quality_pass_ratio": quality_pass_ratio,
                "spectral_pass_ratio": spectral_pass_ratio,
                "interpretation": "DESCRIPTIVE_ASSOCIATION_ONLY_NOT_CAUSAL",
                "baseline_status": "CONTEXT_ONLY_NOT_CONFIRMED_NORMAL",
            })
    return pd.DataFrame(rows)


def _format_number(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6g}"


def write_markdown_report(
    path: Path,
    occurrences: list[EventOccurrence],
    windows: pd.DataFrame,
    summary: pd.DataFrame,
    before_hours: float,
    after_hours: float,
) -> None:
    lines = [
        "# 사건 인접 Feature 검토", "",
        f"- 계약: `{REVIEW_CONTRACT}`",
        f"- 문맥 범위: 사건 전 {before_hours:g}시간 / 후 {after_hours:g}시간",
        "- 사건 전 문맥은 정상 label이 아니며 비교 기준일 뿐이다.",
        "- robust shift와 correlation은 기술 통계이며 이상 판정·인과 판정이 아니다.", "",
    ]
    for occurrence in occurrences:
        event_windows = windows.loc[windows.get("occurrence_uid", pd.Series(dtype=str)) == occurrence.occurrence_uid]
        lines.extend([
            f"## {occurrence.event.priority}. {occurrence.occurrence_uid} — {occurrence.event.asset_id}", "",
            f"- 사건: {occurrence.event.summary}",
            f"- 사건창: `{occurrence.start.isoformat()}` ~ `{occurrence.end.isoformat()}` ({occurrence.precision})",
            f"- 선택된 sensor-window: {len(event_windows):,}", "",
        ])
        candidate = summary.loc[
            (summary.get("occurrence_uid", pd.Series(dtype=str)) == occurrence.occurrence_uid)
            & (summary.get("phase", pd.Series(dtype=str)) == "DURING")
            & summary.get("metric", pd.Series(dtype=str)).isin(MARKDOWN_METRICS)
        ].copy()
        candidate = candidate.loc[candidate["robust_shift_vs_before"].notna()]
        if candidate.empty:
            lines.extend(["비교 가능한 사건 중 robust shift가 없다. Gap·coverage·baseline 부재를 확인한다.", ""])
            continue
        candidate["absolute_shift"] = candidate["robust_shift_vs_before"].abs()
        candidate.sort_values("absolute_shift", ascending=False, inplace=True)
        lines.extend([
            "| Sensor | Metric | Before median | During median | Delta | Robust shift | Quality PASS | Spectral PASS |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for _, row in candidate.head(12).iterrows():
            lines.append(
                f"| `{row['sensor_uid']}` | `{row['metric']}` | {_format_number(row['before_median'])} | "
                f"{_format_number(row['median'])} | {_format_number(row['delta_from_before_median'])} | "
                f"{_format_number(row['robust_shift_vs_before'])} | {_format_number(row['quality_pass_ratio'])} | "
                f"{_format_number(row['spectral_pass_ratio'])} |"
            )
        lines.extend(["", "> 순위는 검토 편의를 위한 절대 robust shift 정렬이며 임계값 또는 이상 확정이 아니다.", ""])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-markdown", type=Path, required=True)
    parser.add_argument("--features-3s", type=Path, required=True)
    parser.add_argument("--features-4s", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=6, help="Priority count; ignored when --event is supplied")
    parser.add_argument("--event", action="append", default=[], help="Exact event ID; repeatable")
    parser.add_argument("--before-hours", type=float, default=24.0)
    parser.add_argument("--after-hours", type=float, default=24.0)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-selected-rows", type=int, default=2_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top < 1 or args.before_hours < 0 or args.after_hours < 0 or args.chunksize < 1:
        raise EventReviewError("ARGUMENT_INVALID", str(vars(args)))
    output = args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise EventReviewError("OUTPUT_EXISTS", str(output))
    output.mkdir(parents=True, exist_ok=True)
    event_path = args.events_markdown.expanduser().resolve()
    feature_files = {
        "3S": args.features_3s.expanduser().resolve(),
        "4S": args.features_4s.expanduser().resolve(),
    }
    events = parse_priority_events(event_path)
    occurrences = select_occurrences(events, None if args.event else args.top, set(args.event))
    windows, scan_stats = scan_event_windows(
        feature_files, occurrences, output / "event_windows.csv", args.before_hours, args.after_hours,
        args.chunksize, args.max_selected_rows,
    )
    metrics = [metric for metric in METRIC_COLUMNS if metric in windows.columns]
    summary = summarize_event_windows(windows, metrics)
    summary.to_csv(output / "event_sensor_metric_summary.csv", index=False)
    write_markdown_report(
        output / "event_review.md", occurrences, windows, summary, args.before_hours, args.after_hours
    )
    manifest_payload = {
        "contract_version": REVIEW_CONTRACT,
        "events_source": _file_identity(event_path),
        "feature_sources": {scope: _file_identity(path) for scope, path in feature_files.items()},
        "event_ids": [occurrence.occurrence_uid for occurrence in occurrences],
        "before_hours": args.before_hours,
        "after_hours": args.after_hours,
        "scan": scan_stats,
        "summary_rows": len(summary),
        "safety": {
            "context_assumed_normal": False,
            "threshold_inferred": False,
            "causality_inferred": False,
            "production_decision_allowed": False,
        },
    }
    receipt = hashlib.sha256(
        json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest = {**manifest_payload, "created_at": datetime.now(tz=TIMEZONE).isoformat(), "receipt_sha256": receipt}
    (output / "event_review_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EVENT_REVIEW=VALID|EVENTS:{len({item.event.event_uid for item in occurrences})}|"
        f"OCCURRENCES:{len(occurrences)}|SCANNED:{scan_stats['source_rows_scanned']}|"
        f"SELECTED:{scan_stats['selected_rows']}|SUMMARY:{len(summary)}|SHA:{receipt}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
