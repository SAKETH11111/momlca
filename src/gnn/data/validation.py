"""Data quality validation for molecular datasets.

This module provides tools for validating dataset quality including:
- Duplicate detection by InChIKey
- Label range consistency checking
- Outlier detection using z-scores
- Missing label counting
- Report generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _get_label_scalar(label_tensor: torch.Tensor, prop_idx: int) -> float:
    if label_tensor.ndim == 1:
        return float(label_tensor[prop_idx].item())
    if label_tensor.ndim == 2 and label_tensor.shape[0] == 1:
        return float(label_tensor[0, prop_idx].item())
    raise ValueError(
        f"Expected labels with shape (num_properties,) or (1, num_properties), got {tuple(label_tensor.shape)}"
    )


@dataclass
class DuplicateCheckResult:
    """Container for duplicate check results."""

    duplicates: list[tuple[int, int]]
    missing_inchikey_count: int


@dataclass
class ValidationResult:
    """Container for validation results.

    Attributes:
        duplicates: List of duplicate index pairs (original_idx, duplicate_idx).
        missing_inchikey_count: Number of items missing a usable InChIKey.
        outliers: Dictionary mapping property names to lists of outlier indices.
        missing_counts: Dictionary mapping property names to missing value counts.
        label_issues: List of label consistency issue descriptions.
        passed: Whether validation passed (True if no issues found).
    """

    duplicates: list[tuple[int, int]]
    missing_inchikey_count: int
    outliers: dict[str, list[int]]
    missing_counts: dict[str, int]
    label_issues: list[str]
    passed: bool


class DataValidator:
    """Data quality validator for molecular datasets.

    This validator performs multiple quality checks on molecular property datasets:
    - Duplicate molecule detection using InChIKey
    - Label value range validation
    - Statistical outlier detection
    - Missing label counting

    Args:
        mode: Validation mode. "warn" logs warnings only, "fail" raises exceptions
            on validation failures. Defaults to "warn".
        outlier_threshold: Number of standard deviations for outlier detection.
            Values beyond this threshold are flagged as outliers. Defaults to 3.0.
        expected_units: Dictionary mapping property names to expected unit strings.
            If the dataset provides a ``property_units`` mapping, units are checked
            for consistency. Defaults to standard units for logS, logP, and pKa.
        expected_ranges: Dictionary mapping property names to (min, max) tuples
            defining expected value ranges. Defaults to standard ranges for
            logS, logP, and pKa.

    Example:
        >>> validator = DataValidator(mode="warn", outlier_threshold=2.5)
        >>> result = validator.validate(dataset)
        >>> if not result.passed:
        ...     print(f"Found {len(result.duplicates)} duplicates")
    """

    def __init__(
        self,
        mode: str = "warn",
        outlier_threshold: float = 3.0,
        expected_units: dict[str, str] | None = None,
        expected_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if mode not in ("warn", "fail"):
            raise ValueError(f"Invalid mode '{mode}'. Must be 'warn' or 'fail'.")

        self.mode = mode
        self.outlier_threshold = outlier_threshold
        self.expected_units = expected_units or {
            "logS": "log(mol/L)",
            "logP": "unitless",
            "pKa": "pH",
        }
        self.expected_ranges = expected_ranges or {
            "logS": (-10.0, 5.0),
            "logP": (-5.0, 10.0),
            "pKa": (-5.0, 20.0),
        }

    def validate(self, dataset) -> ValidationResult:
        """Run all validation checks on dataset.

        Performs duplicate detection, outlier detection, missing label counting,
        and label consistency checking. Returns a ValidationResult with all findings.

        Args:
            dataset: Dataset to validate. Must support len() and indexing, with
                each item having .inchikey, .y attributes and optionally
                .property_names.

        Returns:
            ValidationResult containing all validation findings and overall pass/fail.

        Raises:
            ValueError: If mode is "fail" and validation fails.
        """
        logger.info("Starting data validation (mode=%s)", self.mode)

        duplicate_result = self.check_duplicates_detailed(dataset)
        outliers = self.check_outliers(dataset)
        missing = self.count_missing_labels(dataset)
        label_issues = self.check_label_consistency(dataset)
        label_issues.extend(self.check_unit_consistency(dataset))

        passed = (
            len(duplicate_result.duplicates) == 0
            and duplicate_result.missing_inchikey_count == 0
            and len(label_issues) == 0
            and sum(len(v) for v in outliers.values()) == 0
        )

        logger.info(
            "Validation complete: passed=%s, duplicates=%d, outliers=%d, label_issues=%d",
            passed,
            len(duplicate_result.duplicates),
            sum(len(v) for v in outliers.values()),
            len(label_issues),
        )

        if not passed and self.mode == "fail":
            raise ValueError("Data validation failed. See logs for details.")

        return ValidationResult(
            duplicates=duplicate_result.duplicates,
            missing_inchikey_count=duplicate_result.missing_inchikey_count,
            outliers=outliers,
            missing_counts=missing,
            label_issues=label_issues,
            passed=passed,
        )

    def check_duplicates(self, dataset) -> list[tuple[int, int]]:
        """Detect duplicate molecules by InChIKey.

        This is a convenience wrapper around :meth:`check_duplicates_detailed` that
        returns only the duplicate index pairs.
        """
        return self.check_duplicates_detailed(dataset).duplicates

    def check_duplicates_detailed(self, dataset) -> DuplicateCheckResult:
        """Detect duplicate molecules by InChIKey.

        Scans dataset for molecules with identical InChIKeys, indicating
        duplicate structures.

        Args:
            dataset: Dataset to check for duplicates.

        Returns:
            List of (original_index, duplicate_index) tuples for each duplicate
            found. The original_index is the first occurrence.
        """
        inchikeys: dict[str, int] = {}
        duplicates: list[tuple[int, int]] = []
        missing_inchikey_count = 0

        for i in range(len(dataset)):
            key_raw = getattr(dataset[i], "inchikey", None)
            key = "" if key_raw is None else str(key_raw).strip()
            if not key:
                missing_inchikey_count += 1
                logger.warning("Missing InChIKey at index %d; cannot deduplicate this item", i)
                continue

            if key in inchikeys:
                duplicates.append((inchikeys[key], i))
                logger.warning(
                    "Duplicate molecule: indices %d and %d (InChIKey: %s)", inchikeys[key], i, key
                )
            else:
                inchikeys[key] = i

        if duplicates:
            logger.warning("Found %d duplicate molecules", len(duplicates))

        if missing_inchikey_count:
            logger.warning(
                "%d item(s) missing InChIKey; duplicates for those items cannot be detected",
                missing_inchikey_count,
            )

        return DuplicateCheckResult(
            duplicates=duplicates,
            missing_inchikey_count=missing_inchikey_count,
        )

    def check_outliers(self, dataset) -> dict[str, list[int]]:
        """Flag labels beyond threshold standard deviations from mean.

        Performs z-score based outlier detection for each property independently.
        NaN values are ignored in the calculation.

        Args:
            dataset: Dataset to check for outliers.

        Returns:
            Dictionary mapping property names to lists of outlier indices.
        """
        property_names = getattr(dataset, "property_names", ["logS", "logP", "pKa"])
        outliers: dict[str, list[int]] = {prop: [] for prop in property_names}

        for prop_idx, prop_name in enumerate(property_names):
            values: list[float] = []
            indices: list[int] = []

            for i in range(len(dataset)):
                val = _get_label_scalar(dataset[i].y, prop_idx)
                if not np.isnan(val):
                    values.append(float(val))
                    indices.append(i)

            if len(values) < 2:
                continue

            mean = np.mean(values)
            std = np.std(values)

            if std == 0:
                continue

            for val, idx in zip(values, indices):
                z_score = abs(val - mean) / std
                if z_score > self.outlier_threshold:
                    outliers[prop_name].append(idx)
                    logger.warning(
                        "Outlier %s=%.2f at index %d (z-score=%.2f, mean=%.2f, std=%.2f)",
                        prop_name,
                        val,
                        idx,
                        z_score,
                        mean,
                        std,
                    )

        return outliers

    def count_missing_labels(self, dataset) -> dict[str, int]:
        """Count NaN values per property.

        Args:
            dataset: Dataset to count missing labels.

        Returns:
            Dictionary mapping property names to counts of missing (NaN) values.
        """
        property_names = getattr(dataset, "property_names", ["logS", "logP", "pKa"])
        missing: dict[str, int] = dict.fromkeys(property_names, 0)

        for i in range(len(dataset)):
            for prop_idx, prop_name in enumerate(property_names):
                scalar = _get_label_scalar(dataset[i].y, prop_idx)
                if np.isnan(scalar):
                    missing[prop_name] += 1

        for prop_name, count in missing.items():
            if count > 0:
                pct = 100.0 * count / len(dataset)
                logger.info("Property %s: %d missing (%.1f%%)", prop_name, count, pct)

        return missing

    def check_label_consistency(self, dataset) -> list[str]:
        """Verify labels are within expected ranges.

        Checks each non-NaN label value against the expected ranges defined
        in self.expected_ranges. Values outside these ranges are flagged.

        Args:
            dataset: Dataset to check for label consistency.

        Returns:
            List of issue descriptions for labels outside expected ranges.
        """
        issues: list[str] = []
        property_names = getattr(dataset, "property_names", ["logS", "logP", "pKa"])

        for i in range(len(dataset)):
            for prop_idx, prop_name in enumerate(property_names):
                val = _get_label_scalar(dataset[i].y, prop_idx)
                if np.isnan(val):
                    continue

                if prop_name in self.expected_ranges:
                    min_val, max_val = self.expected_ranges[prop_name]
                    if val < min_val or val > max_val:
                        issue = f"Index {i}: {prop_name}={val:.2f} outside expected range [{min_val}, {max_val}]"
                        issues.append(issue)
                        logger.warning(issue)

        return issues

    def check_unit_consistency(self, dataset) -> list[str]:
        """Verify label unit metadata is consistent with expectations.

        If the dataset exposes a ``property_units`` mapping (property name -> unit
        string), this validates that units match ``self.expected_units``.

        Datasets without unit metadata are allowed (unit check is skipped).
        """
        units = getattr(dataset, "property_units", None)
        if units is None:
            logger.info("Dataset does not expose property_units; skipping unit consistency check")
            return []
        if not isinstance(units, dict):
            raise TypeError(f"Expected dataset.property_units to be a dict, got {type(units)}")

        issues: list[str] = []
        property_names = getattr(dataset, "property_names", ["logS", "logP", "pKa"])

        for prop_name in property_names:
            expected_unit = self.expected_units.get(prop_name)
            if expected_unit is None:
                continue

            actual_unit_raw = units.get(prop_name)
            actual_unit = "" if actual_unit_raw is None else str(actual_unit_raw).strip()
            if not actual_unit:
                issue = f"Missing unit metadata for {prop_name}; expected '{expected_unit}'"
                issues.append(issue)
                logger.warning(issue)
                continue

            if actual_unit != expected_unit:
                issue = f"Unit mismatch for {prop_name}: dataset='{actual_unit}' expected='{expected_unit}'"
                issues.append(issue)
                logger.warning(issue)

        return issues


def generate_report(
    dataset,
    validator: DataValidator | None = None,
    output_path: str = "data_report.md",
) -> str:
    """Generate markdown report with data quality statistics.

    Runs validation and creates a comprehensive markdown report including:
    - Dataset overview
    - Validation summary (pass/fail)
    - Missing label statistics
    - Duplicate molecule list
    - Outlier summary
    - Label distribution statistics

    Args:
        dataset: Dataset to validate and report on.
        validator: Optional DataValidator instance to use for validation. If not
            provided, a default ``DataValidator()`` is used.
        output_path: Path where the markdown report should be saved.
            Defaults to "data_report.md".

    Returns:
        The generated markdown report as a string.

    Example:
        >>> validator = DataValidator()
        >>> report = generate_report(dataset, validator, "reports/data_qa.md")
        >>> print("Report generated:", report[:100])
    """
    logger.info("Generating data quality report: %s", output_path)

    validator = DataValidator() if validator is None else validator
    result = validator.validate(dataset)
    property_names = getattr(dataset, "property_names", ["logS", "logP", "pKa"])

    lines = [
        "# Data Quality Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Dataset size:** {len(dataset)} molecules",
        "",
        "## Summary",
        "",
        f"- **Validation passed:** {'✅ Yes' if result.passed else '❌ No'}",
        f"- **Duplicates found:** {len(result.duplicates)}",
        f"- **Missing InChIKey:** {result.missing_inchikey_count}",
        f"- **Outliers found:** {sum(len(v) for v in result.outliers.values())}",
        f"- **Label issues:** {len(result.label_issues)}",
        "",
        "## Missing Labels",
        "",
    ]

    for prop in property_names:
        count = result.missing_counts.get(prop, 0)
        pct = 100.0 * count / len(dataset) if len(dataset) > 0 else 0.0
        lines.append(f"- **{prop}:** {count} ({pct:.1f}%)")

    units = getattr(dataset, "property_units", None)
    if units is not None:
        lines.extend(["", "## Units", ""])
        for prop in property_names:
            unit = units.get(prop, "unknown") if isinstance(units, dict) else "unknown"
            lines.append(f"- **{prop}:** {unit}")

    lines.extend(["", "## Duplicates", ""])
    if result.duplicates:
        lines.append(f"Found {len(result.duplicates)} duplicate molecule(s):")
        lines.append("")
        for orig_idx, dup_idx in result.duplicates[:10]:  # Show first 10
            lines.append(f"- Indices {orig_idx} and {dup_idx}")
        if len(result.duplicates) > 10:
            lines.append(f"- ... and {len(result.duplicates) - 10} more")
    else:
        lines.append("No duplicates found.")

    lines.extend(["", "## Outliers", ""])
    total_outliers = sum(len(v) for v in result.outliers.values())
    if total_outliers > 0:
        lines.append(f"Found {total_outliers} outlier(s):")
        lines.append("")
        for prop, outlier_indices in result.outliers.items():
            if outlier_indices:
                lines.append(f"- **{prop}:** {len(outlier_indices)} outlier(s)")
                if len(outlier_indices) <= 5:
                    lines.append(f"  - Indices: {outlier_indices}")
                else:
                    lines.append(
                        f"  - Indices: {outlier_indices[:5]} ... and {len(outlier_indices) - 5} more"
                    )
    else:
        lines.append("No outliers detected.")

    lines.extend(["", "## Label Consistency Issues", ""])
    if result.label_issues:
        lines.append(f"Found {len(result.label_issues)} label issue(s):")
        lines.append("")
        for issue in result.label_issues[:10]:  # Show first 10
            lines.append(f"- {issue}")
        if len(result.label_issues) > 10:
            lines.append(f"- ... and {len(result.label_issues) - 10} more")
    else:
        lines.append("All labels within expected ranges.")

    # Label distribution statistics
    lines.extend(["", "## Label Distribution Statistics", ""])
    for prop_idx, prop_name in enumerate(property_names):
        values = []
        for i in range(len(dataset)):
            val = _get_label_scalar(dataset[i].y, prop_idx)
            if not np.isnan(val):
                values.append(float(val))

        if values:
            lines.append(f"### {prop_name}")
            lines.append("")
            lines.append(f"- **Count:** {len(values)}")
            lines.append(f"- **Mean:** {np.mean(values):.2f}")
            lines.append(f"- **Std:** {np.std(values):.2f}")
            lines.append(f"- **Min:** {np.min(values):.2f}")
            lines.append(f"- **Max:** {np.max(values):.2f}")
            lines.append("")

    report = "\n".join(lines)
    Path(output_path).write_text(report)
    logger.info("Report saved to %s", output_path)

    return report
