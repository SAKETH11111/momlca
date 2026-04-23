"""PFAS family-level error analysis built on canonical evaluation exports."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from gnn.data.loaders import load_smiles
from gnn.data.splits.pfas_ood import get_chain_length, get_headgroup
from gnn.evaluation.export import checkpoint_export_id
from gnn.evaluation.metrics.regression import compute_regression_metrics

logger = logging.getLogger(__name__)

FamilyDimension = Literal["chain_length", "headgroup"]

FAMILY_TABLE_COLUMNS = [
    "family_dimension",
    "family_label",
    "property_name",
    "sample_count",
    "mae",
    "rmse",
    "r2",
    "spearman",
    "mean_signed_residual",
    "mean_absolute_residual",
    "low_sample_family",
    "split",
    "checkpoint_path",
    "checkpoint_id",
    "export_id",
]


@dataclass(frozen=True)
class FamilyAnalysisInput:
    """Loaded prediction-export input for one family-analysis run."""

    source_path: Path
    split_name: str
    checkpoint_path: str
    checkpoint_id: str
    export_id: str
    property_names: list[str]
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class FamilyAnnotatedRecord:
    """One prediction record enriched with PFAS family labels and numeric arrays."""

    key: tuple[str, str]
    split_name: str
    smiles: str
    chain_length: str
    headgroup: str
    targets: np.ndarray
    predictions: np.ndarray

    @property
    def residuals(self) -> np.ndarray:
        """Return signed residuals (prediction - target) with NaN propagation."""
        return self.predictions - self.targets


@dataclass(frozen=True)
class FamilyAnalysisArtifacts:
    """Output artifacts from one PFAS family-analysis run."""

    chain_length_csv: Path
    headgroup_csv: Path
    chain_length_figure: Path
    headgroup_figure: Path
    report_md: Path
    run_id: str
    chain_length_df: pd.DataFrame
    headgroup_df: pd.DataFrame


def load_family_analysis_input(path: str | Path) -> FamilyAnalysisInput:
    """Load one canonical prediction export into a family-analysis input contract."""
    source_path = Path(path)
    payload = json.loads(source_path.read_text())
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict):
        raise ValueError(f"Prediction export metadata missing or invalid in {source_path}")
    if not isinstance(records, list):
        raise ValueError(f"Prediction export records missing or invalid in {source_path}")

    split_name = str(metadata.get("split") or "")
    checkpoint_path = str(metadata.get("checkpoint_path") or "")
    if split_name == "":
        raise ValueError(f"Prediction export split missing in {source_path}")
    if checkpoint_path == "":
        raise ValueError(f"Prediction export checkpoint_path missing in {source_path}")

    names_from_metadata = metadata.get("property_names")
    property_names: list[str] = []
    if isinstance(names_from_metadata, list):
        property_names = [str(name) for name in names_from_metadata]
    if not property_names and records:
        first_record = records[0]
        if isinstance(first_record, dict):
            predictions = first_record.get("predictions")
            if isinstance(predictions, dict):
                property_names = [str(name) for name in predictions]
    if not property_names:
        raise ValueError(f"Unable to determine property_names for export {source_path}")

    normalized_records = [dict(record) for record in records if isinstance(record, dict)]
    if len(normalized_records) != len(records):
        raise ValueError(f"Prediction export includes non-object record entries in {source_path}")

    record_checkpoint_paths = {
        str(record.get("checkpoint_path") or "").strip() for record in normalized_records
    }
    if "" in record_checkpoint_paths:
        missing_index = next(
            index
            for index, record in enumerate(normalized_records)
            if str(record.get("checkpoint_path") or "").strip() == ""
        )
        raise ValueError(
            f"Prediction export record checkpoint_path missing in {source_path} "
            f"(record index {missing_index})"
        )
    if len(record_checkpoint_paths) > 1:
        observed_paths = ", ".join(repr(path) for path in sorted(record_checkpoint_paths))
        raise ValueError(
            f"Prediction export mixes record checkpoint_path values in {source_path}: "
            f"metadata checkpoint_path={checkpoint_path!r}, "
            f"record checkpoint_path values=[{observed_paths}]"
        )
    if record_checkpoint_paths and checkpoint_path not in record_checkpoint_paths:
        record_checkpoint_path = next(iter(record_checkpoint_paths))
        raise ValueError(
            f"Prediction export record checkpoint_path {record_checkpoint_path!r} does not match "
            f"metadata checkpoint_path {checkpoint_path!r} in {source_path}"
        )

    return FamilyAnalysisInput(
        source_path=source_path,
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        checkpoint_id=checkpoint_export_id(checkpoint_path),
        export_id=_export_id(source_path),
        property_names=property_names,
        records=normalized_records,
    )


def annotate_family_records(analysis_input: FamilyAnalysisInput) -> list[FamilyAnnotatedRecord]:
    """Derive family labels and validated numeric vectors for each record."""
    seen_keys: set[tuple[str, str]] = set()
    annotated: list[FamilyAnnotatedRecord] = []

    for index, record in enumerate(analysis_input.records):
        record_split = str(record.get("split") or "").strip()
        if record_split != analysis_input.split_name:
            raise ValueError(
                f"Unsupported split mixture in {analysis_input.source_path}: expected "
                f"{analysis_input.split_name!r}, found {record_split!r} at index {index}"
            )

        smiles = str(record.get("smiles") or "").strip()
        if smiles == "":
            raise ValueError(
                f"SMILES is required for family analysis in {analysis_input.source_path} "
                f"(record index {index})"
            )

        key = _record_key(record)
        if key in seen_keys:
            raise ValueError(f"Duplicate sample key {key} in {analysis_input.source_path}")
        seen_keys.add(key)

        targets = _values_for_properties(
            mapping=record.get("targets"),
            property_names=analysis_input.property_names,
            field_name="targets",
            source=str(analysis_input.source_path),
        )
        predictions = _values_for_properties(
            mapping=record.get("predictions"),
            property_names=analysis_input.property_names,
            field_name="predictions",
            source=str(analysis_input.source_path),
        )

        molecule = load_smiles(smiles)
        annotated.append(
            FamilyAnnotatedRecord(
                key=key,
                split_name=analysis_input.split_name,
                smiles=smiles,
                chain_length=get_chain_length(molecule),
                headgroup=get_headgroup(molecule),
                targets=targets,
                predictions=predictions,
            )
        )

    return annotated


def run_family_error_analysis(
    *,
    analysis_input: FamilyAnalysisInput,
    output_dir: str | Path,
    low_sample_threshold: int = 5,
) -> FamilyAnalysisArtifacts:
    """Generate deterministic PFAS family metrics, figures, and markdown report."""
    if low_sample_threshold < 1:
        raise ValueError("low_sample_threshold must be >= 1")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    annotated_records = annotate_family_records(analysis_input)
    chain_df = _build_family_table(
        annotated_records=annotated_records,
        analysis_input=analysis_input,
        family_dimension="chain_length",
        low_sample_threshold=low_sample_threshold,
    )
    head_df = _build_family_table(
        annotated_records=annotated_records,
        analysis_input=analysis_input,
        family_dimension="headgroup",
        low_sample_threshold=low_sample_threshold,
    )

    stem = _artifact_stem(analysis_input)
    chain_csv = output_dir_path / f"{stem}-family-chain_length.csv"
    head_csv = output_dir_path / f"{stem}-family-headgroup.csv"
    chain_figure = output_dir_path / f"{stem}-family-chain_length.svg"
    head_figure = output_dir_path / f"{stem}-family-headgroup.svg"
    report_md = output_dir_path / f"{stem}-family-report.md"

    chain_df.to_csv(chain_csv, index=False, float_format="%.6f")
    head_df.to_csv(head_csv, index=False, float_format="%.6f")

    _write_distribution_figure(
        annotated_records=annotated_records,
        family_dimension="chain_length",
        output_path=chain_figure,
        title="Absolute residual distribution by chain length",
    )
    _write_distribution_figure(
        annotated_records=annotated_records,
        family_dimension="headgroup",
        output_path=head_figure,
        title="Absolute residual distribution by headgroup",
    )
    _write_report(
        report_path=report_md,
        analysis_input=analysis_input,
        chain_df=chain_df,
        head_df=head_df,
        low_sample_threshold=low_sample_threshold,
        chain_figure=chain_figure,
        head_figure=head_figure,
    )

    run_id = _analysis_run_id(analysis_input)
    logger.info(
        "Family error analysis complete for split=%s, checkpoint_id=%s, export_id=%s",
        analysis_input.split_name,
        analysis_input.checkpoint_id,
        analysis_input.export_id,
    )
    return FamilyAnalysisArtifacts(
        chain_length_csv=chain_csv,
        headgroup_csv=head_csv,
        chain_length_figure=chain_figure,
        headgroup_figure=head_figure,
        report_md=report_md,
        run_id=run_id,
        chain_length_df=chain_df,
        headgroup_df=head_df,
    )


def _build_family_table(
    *,
    annotated_records: list[FamilyAnnotatedRecord],
    analysis_input: FamilyAnalysisInput,
    family_dimension: FamilyDimension,
    low_sample_threshold: int,
) -> pd.DataFrame:
    grouped: dict[str, list[FamilyAnnotatedRecord]] = defaultdict(list)
    for record in annotated_records:
        family_label = (
            record.chain_length if family_dimension == "chain_length" else record.headgroup
        )
        grouped[family_label].append(record)

    rows: list[dict[str, Any]] = []
    for family_label in sorted(grouped):
        subset = grouped[family_label]
        targets = np.vstack([entry.targets for entry in subset])
        predictions = np.vstack([entry.predictions for entry in subset])

        for property_index, property_name in enumerate(analysis_input.property_names):
            target_values = _normalize_metric_values(targets[:, property_index])
            prediction_values = _normalize_metric_values(predictions[:, property_index])
            valid_mask = np.isfinite(target_values) & np.isfinite(prediction_values)

            metrics = compute_regression_metrics(
                target_values.reshape(-1, 1),
                prediction_values.reshape(-1, 1),
                [property_name],
                nan_policy="omit",
            )
            residuals = prediction_values[valid_mask] - target_values[valid_mask]
            sample_count = int(valid_mask.sum())

            rows.append(
                {
                    "family_dimension": family_dimension,
                    "family_label": family_label,
                    "property_name": property_name,
                    "sample_count": sample_count,
                    "mae": metrics.get(f"mae_{property_name}", float("nan")),
                    "rmse": metrics.get(f"rmse_{property_name}", float("nan")),
                    "r2": metrics.get(f"r2_{property_name}", float("nan")),
                    "spearman": metrics.get(f"spearman_{property_name}", float("nan")),
                    "mean_signed_residual": (
                        float(np.mean(residuals)) if sample_count > 0 else float("nan")
                    ),
                    "mean_absolute_residual": (
                        float(np.mean(np.abs(residuals))) if sample_count > 0 else float("nan")
                    ),
                    "low_sample_family": sample_count < low_sample_threshold,
                    "split": analysis_input.split_name,
                    "checkpoint_path": analysis_input.checkpoint_path,
                    "checkpoint_id": analysis_input.checkpoint_id,
                    "export_id": analysis_input.export_id,
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=FAMILY_TABLE_COLUMNS)

    frame = frame[FAMILY_TABLE_COLUMNS].sort_values(
        by=["family_label", "property_name"],
        ignore_index=True,
    )
    return frame


def _write_report(
    *,
    report_path: Path,
    analysis_input: FamilyAnalysisInput,
    chain_df: pd.DataFrame,
    head_df: pd.DataFrame,
    low_sample_threshold: int,
    chain_figure: Path,
    head_figure: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PFAS Family Error Analysis",
        "",
        f"- Split: `{analysis_input.split_name}`",
        f"- Checkpoint: `{analysis_input.checkpoint_path}`",
        f"- Checkpoint ID: `{analysis_input.checkpoint_id}`",
        f"- Export ID: `{analysis_input.export_id}`",
        f"- Source export: `{analysis_input.source_path}`",
        "",
        "## Artifact Inventory",
        "",
        f"- Chain-length table: `{chain_df.shape[0]} rows`",
        f"- Headgroup table: `{head_df.shape[0]} rows`",
        f"- Chain-length distribution figure: `{chain_figure.name}`",
        f"- Headgroup distribution figure: `{head_figure.name}`",
        "",
        "## Low-sample families",
        "",
        *(_low_sample_lines(chain_df, head_df, threshold=low_sample_threshold)),
        "",
        "## Worst-performing families",
        "",
        *(_worst_family_lines(chain_df, head_df)),
        "",
        "## Directional bias highlights",
        "",
        *(_bias_lines(chain_df, head_df)),
        "",
        "## Unknown/other family coverage",
        "",
        *(_unknown_family_lines(chain_df, head_df)),
        "",
        "## Chain-length subgroup metrics",
        "",
        _to_markdown_or_fallback(chain_df),
        "",
        "## Headgroup subgroup metrics",
        "",
        _to_markdown_or_fallback(head_df),
        "",
    ]
    report_path.write_text("\n".join(lines))


def _to_markdown_or_fallback(frame: pd.DataFrame) -> str:
    try:
        return frame.to_markdown(index=False, floatfmt=".6f")
    except ImportError:
        logger.warning(
            "tabulate is unavailable; falling back to plain-text table rendering in report"
        )
        return "\n".join(
            [
                "_Markdown table rendering unavailable (install `tabulate` for table formatting)._",
                "",
                "```text",
                frame.to_string(index=False),
                "```",
            ]
        )


def _low_sample_lines(
    chain_df: pd.DataFrame,
    head_df: pd.DataFrame,
    *,
    threshold: int,
) -> list[str]:
    combined = pd.concat([chain_df, head_df], ignore_index=True)
    subset = combined[(combined["sample_count"] > 0) & (combined["low_sample_family"])]
    if subset.empty:
        return [f"- None (all family-property rows have sample_count >= {threshold})."]

    lines = []
    for _, row in subset.sort_values(
        by=["family_dimension", "family_label", "property_name"]
    ).iterrows():
        lines.append(
            "- "
            f"{row['family_dimension']}::{row['family_label']}::{row['property_name']} "
            f"(n={int(row['sample_count'])})"
        )
    return lines


def _worst_family_lines(chain_df: pd.DataFrame, head_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for frame, dimension in [(chain_df, "chain_length"), (head_df, "headgroup")]:
        for property_name in sorted(frame["property_name"].unique()):
            subset = frame[(frame["property_name"] == property_name) & (frame["sample_count"] > 0)]
            if subset.empty:
                continue
            worst = subset.sort_values(by=["mae", "family_label"], ascending=[False, True]).iloc[0]
            lines.append(
                f"- {dimension} `{property_name}` worst MAE: `{worst['family_label']}` "
                f"(MAE={float(worst['mae']):.6f}, n={int(worst['sample_count'])})"
            )
    if not lines:
        return ["- No valid subgroup rows available for MAE ranking."]
    return lines


def _bias_lines(chain_df: pd.DataFrame, head_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for frame, dimension in [(chain_df, "chain_length"), (head_df, "headgroup")]:
        for property_name in sorted(frame["property_name"].unique()):
            subset = frame[(frame["property_name"] == property_name) & (frame["sample_count"] > 0)]
            subset = subset[np.isfinite(subset["mean_signed_residual"])]
            if subset.empty:
                continue
            ranked = subset.assign(abs_bias=subset["mean_signed_residual"].abs()).sort_values(
                by=["abs_bias", "family_label"],
                ascending=[False, True],
            )
            top = ranked.iloc[0]
            direction = (
                "overprediction" if float(top["mean_signed_residual"]) > 0 else "underprediction"
            )
            lines.append(
                f"- {dimension} `{property_name}` strongest bias: `{top['family_label']}` "
                f"({direction}, mean_signed_residual={float(top['mean_signed_residual']):.6f}, "
                f"n={int(top['sample_count'])})"
            )
    if not lines:
        return ["- No valid subgroup rows available for bias analysis."]
    return lines


def _unknown_family_lines(chain_df: pd.DataFrame, head_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for frame, labels in [(chain_df, {"unknown"}), (head_df, {"other"})]:
        subset = frame[frame["family_label"].isin(labels)]
        if subset.empty:
            continue
        grouped = subset.groupby(["family_dimension", "family_label"], sort=True)[
            "sample_count"
        ].max()
        for (family_dimension, family_label), sample_count in grouped.items():
            lines.append(
                f"- {family_dimension} `{family_label}` retained in analysis (max n={int(sample_count)})."
            )
    if not lines:
        return ["- No unknown/other families were present in this export."]
    return lines


def _write_distribution_figure(
    *,
    annotated_records: list[FamilyAnnotatedRecord],
    family_dimension: FamilyDimension,
    output_path: Path,
    title: str,
) -> None:
    distributions: dict[str, list[float]] = defaultdict(list)
    for record in annotated_records:
        family_label = (
            record.chain_length if family_dimension == "chain_length" else record.headgroup
        )
        finite = record.residuals[np.isfinite(record.residuals)]
        distributions[family_label].extend(np.abs(finite).tolist())

    labels = sorted(distributions)
    chart_width = max(520, 130 * max(len(labels), 1))
    chart_height = 360
    left = 70
    right = 20
    top = 50
    bottom = 85
    plot_width = chart_width - left - right
    plot_height = chart_height - top - bottom

    all_values = [value for values in distributions.values() for value in values]
    max_value = max(all_values) if all_values else 1.0
    if max_value <= 0.0:
        max_value = 1.0

    def y_coord(value: float) -> float:
        return top + plot_height - (value / max_value) * plot_height

    escaped_title = _escape_svg_text(title)
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{chart_width}" height="{chart_height}" '
        f'viewBox="0 0 {chart_width} {chart_height}">',
        f'<text x="{left}" y="24" font-size="16" font-family="Arial">{escaped_title}</text>',
        f'<text x="{left}" y="40" font-size="12" font-family="Arial">Absolute residuals</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111"/>',
    ]

    for tick in range(5):
        value = max_value * tick / 4.0
        y = y_coord(value)
        svg_parts.append(
            f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111"/>'
        )
        svg_parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="10" font-family="Arial">{value:.2f}</text>'
        )

    spacing = plot_width / len(labels) if labels else plot_width

    for index, label in enumerate(labels):
        x_center = left + spacing * (index + 0.5)
        values = np.asarray(distributions[label], dtype=float)
        if values.size > 0:
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            lower = float(np.min(values))
            upper = float(np.max(values))
            box_width = max(16.0, min(44.0, spacing * 0.42))
            x0 = x_center - box_width / 2.0
            x1 = x_center + box_width / 2.0
            svg_parts.extend(
                [
                    f'<line x1="{x_center:.2f}" y1="{y_coord(upper):.2f}" x2="{x_center:.2f}" y2="{y_coord(lower):.2f}" stroke="#1f77b4"/>',
                    f'<rect x="{x0:.2f}" y="{y_coord(q3):.2f}" width="{box_width:.2f}" height="{max(1.0, y_coord(q1) - y_coord(q3)):.2f}" fill="#9ecae1" stroke="#1f77b4"/>',
                    f'<line x1="{x0:.2f}" y1="{y_coord(median):.2f}" x2="{x1:.2f}" y2="{y_coord(median):.2f}" stroke="#084594" stroke-width="2"/>',
                ]
        )
        escaped_label = _escape_svg_text(label)
        svg_parts.append(
            f'<text x="{x_center:.2f}" y="{top + plot_height + 18}" text-anchor="middle" font-size="10" font-family="Arial">{escaped_label}</text>'
        )

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts))


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    inchikey = str(record.get("inchikey") or "").strip()
    smiles = str(record.get("smiles") or "").strip()
    if inchikey == "" and smiles == "":
        raise ValueError("Record alignment requires at least inchikey or smiles")
    if inchikey != "":
        return ("inchikey", inchikey)
    return ("smiles", smiles)


def _escape_svg_text(value: str) -> str:
    return html.escape(value, quote=False)


def _values_for_properties(
    *,
    mapping: Any,
    property_names: list[str],
    field_name: str,
    source: str,
) -> np.ndarray:
    if not isinstance(mapping, dict):
        raise ValueError(f"Record field '{field_name}' must be a mapping in {source}")
    values = np.full(len(property_names), np.nan, dtype=float)
    for index, property_name in enumerate(property_names):
        if property_name not in mapping:
            raise ValueError(
                f"Record field '{field_name}' missing property '{property_name}' in {source}"
            )
        values[index] = _coerce_float(mapping[property_name], field_name=field_name, source=source)
    return values


def _coerce_float(value: Any, *, field_name: str, source: str) -> float:
    if value is None:
        return float("nan")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Record field '{field_name}' contains non-numeric value {value!r} in {source}"
        ) from exc
    if not math.isfinite(scalar):
        return float("nan")
    return scalar


def _normalize_metric_values(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=float).copy()
    normalized[~np.isfinite(normalized)] = np.nan
    return normalized


def _export_id(source_path: Path) -> str:
    stem = source_path.stem
    normalized_source = source_path.expanduser().resolve(strict=False)
    digest = hashlib.sha1(
        normalized_source.as_posix().encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:8]
    return f"{stem}-{digest}"


def _analysis_run_id(analysis_input: FamilyAnalysisInput) -> str:
    payload = "|".join(
        [
            analysis_input.split_name,
            analysis_input.checkpoint_path,
            analysis_input.export_id,
            ",".join(analysis_input.property_names),
        ]
    )
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:10]


def _artifact_stem(analysis_input: FamilyAnalysisInput) -> str:
    return f"{analysis_input.split_name}-{analysis_input.export_id}"


__all__ = [
    "FamilyAnalysisArtifacts",
    "FamilyAnalysisInput",
    "FamilyAnnotatedRecord",
    "annotate_family_records",
    "load_family_analysis_input",
    "run_family_error_analysis",
]
