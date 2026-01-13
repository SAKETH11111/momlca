"""Tests for data validation module."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from gnn.data.validation import (
    DataValidator,
    ValidationResult,
    generate_report,
)


class MockDataset:
    """Mock dataset for testing validation."""

    def __init__(
        self,
        data_list: list[Data],
        property_names: list[str] | None = None,
        property_units: dict[str, str] | None = None,
    ):
        self.data_list = data_list
        self.property_names = property_names or ["logS", "logP", "pKa"]
        self.property_units = property_units

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Data:
        return self.data_list[idx]


def create_mock_data(
    inchikey: str,
    y_values: list[float],
) -> Data:
    """Create a mock Data object for testing."""
    data = Data()
    data.inchikey = inchikey
    data.y = torch.tensor(y_values, dtype=torch.float32)
    return data


class TestDataValidator:
    """Test DataValidator class."""

    def test_validator_initialization(self):
        """Test validator initializes with correct defaults."""
        validator = DataValidator()
        assert validator.mode == "warn"
        assert validator.outlier_threshold == 3.0
        assert "logS" in validator.expected_units
        assert "logP" in validator.expected_units
        assert "pKa" in validator.expected_units
        assert "logS" in validator.expected_ranges
        assert "logP" in validator.expected_ranges
        assert "pKa" in validator.expected_ranges

    def test_validator_custom_mode(self):
        """Test validator accepts custom mode."""
        validator = DataValidator(mode="fail")
        assert validator.mode == "fail"

    def test_validator_custom_threshold(self):
        """Test validator accepts custom outlier threshold."""
        validator = DataValidator(outlier_threshold=2.5)
        assert validator.outlier_threshold == 2.5

    def test_validator_custom_ranges(self):
        """Test validator accepts custom expected ranges."""
        custom_ranges = {"logS": (-5.0, 3.0)}
        validator = DataValidator(expected_ranges=custom_ranges)
        assert validator.expected_ranges == custom_ranges


class TestDuplicateDetection:
    """Test duplicate detection functionality."""

    def test_check_duplicates_finds_duplicates(self):
        """Test duplicate detection finds duplicate molecules."""
        data_list = [
            create_mock_data("INCHIKEY1", [1.0, 2.0, 3.0]),
            create_mock_data("INCHIKEY2", [4.0, 5.0, 6.0]),
            create_mock_data("INCHIKEY1", [7.0, 8.0, 9.0]),  # Duplicate
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        duplicates = validator.check_duplicates(dataset)

        assert len(duplicates) == 1
        assert duplicates[0] == (0, 2)

    def test_check_duplicates_no_duplicates(self):
        """Test duplicate detection with unique molecules."""
        data_list = [
            create_mock_data("INCHIKEY1", [1.0, 2.0, 3.0]),
            create_mock_data("INCHIKEY2", [4.0, 5.0, 6.0]),
            create_mock_data("INCHIKEY3", [7.0, 8.0, 9.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        duplicates = validator.check_duplicates(dataset)

        assert len(duplicates) == 0

    def test_check_duplicates_multiple_duplicates(self):
        """Test duplicate detection with multiple duplicate pairs."""
        data_list = [
            create_mock_data("INCHIKEY1", [1.0, 2.0, 3.0]),
            create_mock_data("INCHIKEY2", [4.0, 5.0, 6.0]),
            create_mock_data("INCHIKEY1", [7.0, 8.0, 9.0]),  # Duplicate of 0
            create_mock_data("INCHIKEY1", [10.0, 11.0, 12.0]),  # Another duplicate of 0
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        duplicates = validator.check_duplicates(dataset)

        assert len(duplicates) == 2
        assert (0, 2) in duplicates
        assert (0, 3) in duplicates


class TestInchiKeyHandling:
    """Test InChIKey edge cases."""

    def test_check_duplicates_ignores_missing_inchikey(self):
        """Missing/blank InChIKeys should not produce false duplicate pairs."""
        data_list = [
            create_mock_data("", [1.0, 2.0, 3.0]),
            create_mock_data("INCHIKEY1", [4.0, 5.0, 6.0]),
            create_mock_data("", [7.0, 8.0, 9.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        result = validator.check_duplicates_detailed(dataset)

        assert result.duplicates == []
        assert result.missing_inchikey_count == 2


class TestLabelConsistency:
    """Test label consistency checking."""

    def test_check_label_consistency_valid_labels(self):
        """Test label consistency with all valid labels."""
        data_list = [
            create_mock_data("KEY1", [-5.0, 2.0, 10.0]),
            create_mock_data("KEY2", [-2.0, 5.0, 8.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        issues = validator.check_label_consistency(dataset)

        assert len(issues) == 0

    def test_check_label_consistency_out_of_range(self):
        """Test label consistency detects out-of-range values."""
        data_list = [
            create_mock_data("KEY1", [-15.0, 2.0, 10.0]),  # logS too low
            create_mock_data("KEY2", [-2.0, 15.0, 8.0]),  # logP too high
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        issues = validator.check_label_consistency(dataset)

        assert len(issues) == 2
        assert any("logS" in issue for issue in issues)
        assert any("logP" in issue for issue in issues)

    def test_check_label_consistency_ignores_nan(self):
        """Test label consistency ignores NaN values."""
        data_list = [
            create_mock_data("KEY1", [float("nan"), 2.0, 10.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        issues = validator.check_label_consistency(dataset)

        assert len(issues) == 0


class TestUnitConsistency:
    """Test label unit consistency checking."""

    def test_check_unit_consistency_matches_defaults(self):
        """Unit check should pass when dataset units match expected defaults."""
        data_list = [
            create_mock_data("KEY1", [-5.0, 2.0, 10.0]),
            create_mock_data("KEY2", [-2.0, 5.0, 8.0]),
        ]
        dataset = MockDataset(
            data_list,
            property_units={
                "logS": "log(mol/L)",
                "logP": "unitless",
                "pKa": "pH",
            },
        )
        validator = DataValidator()

        issues = validator.check_unit_consistency(dataset)

        assert issues == []

    def test_check_unit_consistency_detects_mismatch(self):
        """Unit check should flag mismatched units."""
        data_list = [
            create_mock_data("KEY1", [-5.0, 2.0, 10.0]),
        ]
        dataset = MockDataset(
            data_list,
            property_units={
                "logS": "mol/L",
                "logP": "unitless",
                "pKa": "pH",
            },
        )
        validator = DataValidator()

        issues = validator.check_unit_consistency(dataset)

        assert len(issues) == 1
        assert "logS" in issues[0]


class TestOutlierDetection:
    """Test outlier detection functionality."""

    def test_check_outliers_detects_outliers(self):
        """Test outlier detection finds outliers beyond threshold."""
        # Create dataset with clear outlier (need enough data points for z-score)
        # With 20 points clustered around 1.0 and one at 50.0, the outlier will have z-score > 3
        base_values = [
            [1.0, 0.0, 0.0],
            [1.1, 1.0, 1.0],
            [0.9, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.2, 0.5, 0.5],
            [0.8, 1.0, 1.0],
            [1.1, 0.5, 0.5],
            [0.9, 1.0, 1.0],
            [1.0, 0.5, 0.5],
            [1.1, 1.0, 1.0],
            [0.9, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [1.2, 0.5, 0.5],
            [0.8, 1.0, 1.0],
            [1.1, 0.5, 0.5],
            [0.9, 1.0, 1.0],
            [1.0, 0.5, 0.5],
            [1.1, 1.0, 1.0],
            [0.95, 0.5, 0.5],
        ]
        data_list = [create_mock_data(f"KEY{i}", vals) for i, vals in enumerate(base_values)]
        data_list.append(create_mock_data("OUTLIER", [50.0, 1.0, 1.0]))  # Clear outlier in logS

        dataset = MockDataset(data_list)
        validator = DataValidator(outlier_threshold=3.0)

        outliers = validator.check_outliers(dataset)

        assert "logS" in outliers
        assert len(outliers["logS"]) > 0
        assert 19 in outliers["logS"]  # Index of the outlier

    def test_check_outliers_no_outliers(self):
        """Test outlier detection with no outliers."""
        data_list = [
            create_mock_data("KEY1", [0.0, 0.0, 0.0]),
            create_mock_data("KEY2", [1.0, 1.0, 1.0]),
            create_mock_data("KEY3", [0.5, 0.5, 0.5]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        outliers = validator.check_outliers(dataset)

        assert all(len(v) == 0 for v in outliers.values())

    def test_check_outliers_ignores_nan(self):
        """Test outlier detection ignores NaN values."""
        data_list = [
            create_mock_data("KEY1", [0.0, float("nan"), 0.0]),
            create_mock_data("KEY2", [1.0, float("nan"), 1.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        outliers = validator.check_outliers(dataset)

        # Should not raise error and logP should have no outliers
        assert "logP" in outliers
        assert len(outliers["logP"]) == 0


class TestMissingLabelCounting:
    """Test missing label counting."""

    def test_count_missing_labels_no_missing(self):
        """Test missing label counting with no missing values."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY2", [4.0, 5.0, 6.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        missing = validator.count_missing_labels(dataset)

        assert missing["logS"] == 0
        assert missing["logP"] == 0
        assert missing["pKa"] == 0

    def test_count_missing_labels_with_missing(self):
        """Test missing label counting with missing values."""
        data_list = [
            create_mock_data("KEY1", [float("nan"), 2.0, 3.0]),
            create_mock_data("KEY2", [4.0, float("nan"), 6.0]),
            create_mock_data("KEY3", [float("nan"), 8.0, float("nan")]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        missing = validator.count_missing_labels(dataset)

        assert missing["logS"] == 2
        assert missing["logP"] == 1
        assert missing["pKa"] == 1

    def test_count_missing_labels_all_missing(self):
        """Test missing label counting with all missing."""
        data_list = [
            create_mock_data("KEY1", [float("nan"), float("nan"), float("nan")]),
            create_mock_data("KEY2", [float("nan"), float("nan"), float("nan")]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        missing = validator.count_missing_labels(dataset)

        assert missing["logS"] == 2
        assert missing["logP"] == 2
        assert missing["pKa"] == 2


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self):
        """Test ValidationResult can be created."""
        result = ValidationResult(
            duplicates=[(0, 1)],
            missing_inchikey_count=0,
            outliers={"logS": [2]},
            missing_counts={"logS": 5, "logP": 3, "pKa": 0},
            label_issues=["Issue 1"],
            passed=False,
        )

        assert len(result.duplicates) == 1
        assert len(result.outliers) == 1
        assert result.missing_counts["logS"] == 5
        assert len(result.label_issues) == 1
        assert result.passed is False


class TestValidateMethod:
    """Test the main validate() method."""

    def test_validate_returns_result(self):
        """Test validate returns ValidationResult."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()

        result = validator.validate(dataset)

        assert isinstance(result, ValidationResult)

    def test_validate_passes_clean_data(self):
        """Test validate passes with clean data."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY2", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator(mode="warn")

        result = validator.validate(dataset)

        assert result.passed is True

    def test_validate_fails_with_duplicates(self):
        """Test validate fails with duplicates."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY1", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator(mode="warn")

        result = validator.validate(dataset)

        assert result.passed is False
        assert len(result.duplicates) > 0

    def test_validate_warn_mode_no_exception(self):
        """Test validate in warn mode does not raise exception."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY1", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator(mode="warn")

        # Should not raise
        result = validator.validate(dataset)
        assert result.passed is False

    def test_validate_fail_mode_raises_exception(self):
        """Test validate in fail mode raises exception on failure."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY1", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator(mode="fail")

        with pytest.raises(ValueError, match="Data validation failed"):
            validator.validate(dataset)

    def test_validate_fails_with_missing_inchikey(self):
        """Missing InChIKey should fail validation (cannot verify duplicates)."""
        data_list = [
            create_mock_data("", [1.0, 2.0, 3.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator(mode="warn")

        result = validator.validate(dataset)

        assert result.passed is False
        assert result.missing_inchikey_count == 1

    def test_validate_fail_mode_raises_on_missing_inchikey(self):
        """Fail mode should raise if InChIKey coverage is insufficient."""
        data_list = [
            create_mock_data("", [1.0, 2.0, 3.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator(mode="fail")

        with pytest.raises(ValueError, match="Data validation failed"):
            validator.validate(dataset)


class TestReportGeneration:
    """Test report generation."""

    def test_generate_report_creates_file(self, tmp_path):
        """Test generate_report creates markdown file."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY2", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()
        output_path = tmp_path / "report.md"

        report = generate_report(dataset, validator, str(output_path))

        assert output_path.exists()
        assert "Data Quality Report" in report

    def test_generate_report_includes_summary(self, tmp_path):
        """Test generate_report includes summary section."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY2", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()
        output_path = tmp_path / "report.md"

        report = generate_report(dataset, validator, str(output_path))

        assert "Summary" in report
        assert "Dataset size" in report
        assert "Validation passed" in report
        assert "Missing InChIKey" in report

    def test_generate_report_includes_missing_labels(self, tmp_path):
        """Test generate_report includes missing label counts."""
        data_list = [
            create_mock_data("KEY1", [float("nan"), 2.0, 3.0]),
            create_mock_data("KEY2", [2.0, float("nan"), 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()
        output_path = tmp_path / "report.md"

        report = generate_report(dataset, validator, str(output_path))

        assert "Missing Labels" in report
        assert "logS" in report
        assert "logP" in report

    def test_generate_report_includes_duplicates(self, tmp_path):
        """Test generate_report includes duplicate information."""
        data_list = [
            create_mock_data("KEY1", [1.0, 2.0, 3.0]),
            create_mock_data("KEY1", [2.0, 3.0, 4.0]),
        ]
        dataset = MockDataset(data_list)
        validator = DataValidator()
        output_path = tmp_path / "report.md"

        report = generate_report(dataset, validator, str(output_path))

        assert "Duplicates" in report or "duplicates" in report.lower()
