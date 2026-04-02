"""Tests for the PFASBench data curation script.

These tests verify the data acquisition and cleaning pipeline for
curating PFAS molecular data from EPA CompTox Dashboard exports.
"""

from __future__ import annotations

import json

# Add scripts to path
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from curate_pfasbench import (
    CompToxAPIClient,
    CurationStats,
    MoleculeProcessor,
    compute_esol_logs,
    compute_logp,
    compute_properties_for_smiles,
    deduplicate_by_inchikey,
    extract_property_value,
    find_column,
    load_comptox_export,
    process_comptox_dataframe,
    save_dataset,
    save_report,
)


# Test fixtures
@pytest.fixture
def sample_comptox_csv(tmp_path: Path) -> Path:
    """Create a sample CompTox-style CSV for testing."""
    data = {
        "DTXSID": ["DTXSID001", "DTXSID002", "DTXSID003", "DTXSID004"],
        "PREFERRED_NAME": ["TFA", "PFOA", "PFBS", "Invalid"],
        "SMILES": [
            "C(=O)(C(F)(F)F)O",  # TFA - valid
            "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA - valid
            "C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)(F)F",  # PFBS - valid
            "not_a_valid_smiles",  # Invalid
        ],
        "INCHIKEY": [
            "DTBXFDXVGNOQPV-UHFFFAOYSA-N",
            "SNGREZUHAYWORS-UHFFFAOYSA-N",
            "YZXBAPSDXZZRGB-UHFFFAOYSA-M",
            "",
        ],
        "OPERA_LOGP_PRED": [0.5, 4.8, 1.5, None],
        "OPERA_WS_PRED": [-0.5, -3.5, -1.0, None],
        "OPERA_pKa_acidic": [0.5, 0.5, None, None],
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_comptox.csv"
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_comptox_with_salts(tmp_path: Path) -> Path:
    """Create a CSV with salt/mixture SMILES for testing salt stripping."""
    data = {
        "SMILES": [
            "C(=O)(C(F)(F)F)O.[Na]",  # TFA sodium salt
            "CC.CCC",  # Ethane + propane mixture
            "C(=O)(C(F)(F)F)O",  # Regular TFA
        ],
        "PREFERRED_NAME": ["TFA-Na", "mixture", "TFA"],
        "OPERA_LOGP_PRED": [0.5, 1.0, 0.5],
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_salts.csv"
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def sample_comptox_with_duplicates(tmp_path: Path) -> Path:
    """Create a CSV with duplicate molecules for deduplication testing."""
    data = {
        "SMILES": [
            "C(=O)(C(F)(F)F)O",  # TFA
            "FC(F)(F)C(=O)O",  # TFA - same molecule, different SMILES
            "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
        ],
        "PREFERRED_NAME": ["TFA-1", "TFA-2", "PFOA"],
        "OPERA_LOGP_PRED": [0.5, 0.6, 4.8],
        "OPERA_WS_PRED": [-0.5, None, -3.5],
        "OPERA_pKa_acidic": [0.5, 0.6, None],
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "test_duplicates.csv"
    df.to_csv(filepath, index=False)
    return filepath


class TestMoleculeProcessor:
    """Tests for the MoleculeProcessor class."""

    def test_parse_smiles_valid(self) -> None:
        """Test parsing valid SMILES strings."""
        processor = MoleculeProcessor()
        mol = processor.parse_smiles("C(=O)(C(F)(F)F)O")
        assert mol is not None
        assert mol.GetNumAtoms() > 0

    def test_parse_smiles_invalid(self) -> None:
        """Test parsing invalid SMILES returns None."""
        processor = MoleculeProcessor()
        assert processor.parse_smiles("not_valid_smiles") is None
        assert processor.parse_smiles("") is None
        assert processor.parse_smiles(None) is None  # type: ignore

    def test_strip_salts_removes_counterion(self) -> None:
        """Test that salt stripping removes counterions."""
        processor = MoleculeProcessor()
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O.[Na]")
        assert mol is not None
        stripped = processor.strip_salts(mol)
        # Should keep only the TFA, not the sodium
        assert stripped.GetNumAtoms() < mol.GetNumAtoms()
        # Check it's the carboxylic acid, not sodium
        assert any(atom.GetAtomicNum() == 6 for atom in stripped.GetAtoms())

    def test_strip_salts_keeps_largest_fragment(self) -> None:
        """Test that largest fragment is kept for mixtures."""
        processor = MoleculeProcessor()
        mol = Chem.MolFromSmiles("CC.CCCCCC")  # Ethane + hexane
        assert mol is not None
        stripped = processor.strip_salts(mol)
        # Should keep hexane (6 carbons), not ethane (2 carbons)
        assert stripped.GetNumHeavyAtoms() == 6

    def test_canonicalize_produces_consistent_smiles(self) -> None:
        """Test canonicalization produces consistent output."""
        processor = MoleculeProcessor()
        mol1 = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        mol2 = Chem.MolFromSmiles("FC(F)(F)C(=O)O")
        assert mol1 is not None and mol2 is not None
        canon1 = processor.canonicalize(mol1)
        canon2 = processor.canonicalize(mol2)
        assert canon1 == canon2

    def test_generate_inchikey(self) -> None:
        """Test InChIKey generation."""
        processor = MoleculeProcessor()
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        assert mol is not None
        inchikey = processor.generate_inchikey(mol)
        assert inchikey.startswith("DTQVDTLACAAQTR")  # TFA InChIKey prefix

    def test_process_molecule_complete(self) -> None:
        """Test complete molecule processing pipeline."""
        processor = MoleculeProcessor()
        result = processor.process_molecule(
            smiles="C(=O)(C(F)(F)F)O",
            name="TFA",
        )
        assert result is not None
        assert "smiles" in result
        assert "inchikey" in result
        assert result["inchikey"].startswith("DTQVDTLACAAQTR")
        assert result["name"] == "TFA"

    def test_process_molecule_invalid_returns_none(self) -> None:
        """Test that invalid molecules return None."""
        processor = MoleculeProcessor()
        result = processor.process_molecule(smiles="invalid_smiles")
        assert result is None
        assert processor.stats.invalid_smiles == 1


class TestPropertyExtraction:
    """Tests for property value extraction."""

    def test_find_column_exact_match(self) -> None:
        """Test finding exact column match."""
        df = pd.DataFrame({"SMILES": [], "OPERA_LOGP_PRED": []})
        assert find_column(df, ["SMILES", "smiles"]) == "SMILES"
        assert find_column(df, ["OPERA_LOGP_PRED"]) == "OPERA_LOGP_PRED"

    def test_find_column_not_found(self) -> None:
        """Test behavior when column not found."""
        df = pd.DataFrame({"col_a": [], "col_b": []})
        assert find_column(df, ["col_c", "col_d"]) is None

    def test_extract_property_value_logp(self) -> None:
        """Test extracting logP property value."""
        row = pd.Series({"OPERA_LOGP_PRED": 4.5, "other_col": 1.0})
        value = extract_property_value(row, "logP")
        assert value == 4.5

    def test_extract_property_value_missing(self) -> None:
        """Test extracting missing property returns NaN."""
        row = pd.Series({"other_col": 1.0})
        value = extract_property_value(row, "logP")
        assert np.isnan(value)

    def test_extract_property_value_nan_in_data(self) -> None:
        """Test extracting property with NaN value."""
        row = pd.Series({"OPERA_LOGP_PRED": float("nan")})
        value = extract_property_value(row, "logP")
        assert np.isnan(value)


class TestDataProcessing:
    """Tests for DataFrame processing."""

    def test_load_comptox_csv(self, sample_comptox_csv: Path) -> None:
        """Test loading CompTox CSV export."""
        df = load_comptox_export(sample_comptox_csv)
        assert len(df) == 4
        assert "SMILES" in df.columns
        assert "PREFERRED_NAME" in df.columns

    def test_process_comptox_dataframe(self, sample_comptox_csv: Path) -> None:
        """Test processing CompTox DataFrame."""
        df = load_comptox_export(sample_comptox_csv)
        processor = MoleculeProcessor()
        records = process_comptox_dataframe(df, processor, source_name="test.csv")

        # Should have 3 valid molecules (excluding invalid SMILES)
        assert len(records) == 3
        assert processor.stats.valid_smiles == 3
        assert processor.stats.invalid_smiles == 1

        # Check properties extracted
        tfa_record = next(r for r in records if "TFA" in r.get("name", ""))
        assert tfa_record["logP"] == 0.5
        assert tfa_record["logS"] == -0.5
        assert tfa_record["pKa"] == 0.5

    def test_salt_stripping_during_processing(self, sample_comptox_with_salts: Path) -> None:
        """Test that salts are stripped during processing."""
        df = load_comptox_export(sample_comptox_with_salts)
        processor = MoleculeProcessor()
        records = process_comptox_dataframe(df, processor, source_name="salts.csv")

        # Should have processed all 3 molecules
        assert len(records) == 3
        # At least one salt should have been stripped
        assert processor.stats.salt_stripped >= 1


class TestDeduplication:
    """Tests for deduplication by InChIKey."""

    def test_deduplicate_removes_duplicates(self, sample_comptox_with_duplicates: Path) -> None:
        """Test that duplicate molecules are merged."""
        df = load_comptox_export(sample_comptox_with_duplicates)
        processor = MoleculeProcessor()
        records = process_comptox_dataframe(df, processor, source_name="dups.csv")

        stats = CurationStats()
        deduplicated = deduplicate_by_inchikey(records, stats)

        # TFA appears twice with same InChIKey, should be deduplicated to 1
        # PFOA appears once
        assert len(deduplicated) == 2
        assert stats.duplicates_removed == 1

    def test_deduplicate_merges_properties(self) -> None:
        """Test that properties are merged for duplicates."""
        records = [
            {
                "smiles": "FC(F)(F)C(=O)O",
                "inchikey": "DTBXFDXVGNOQPV-UHFFFAOYSA-N",
                "name": "TFA-1",
                "logP": 0.5,
                "logS": float("nan"),
                "pKa": 0.4,
                "source": "source1",
            },
            {
                "smiles": "FC(F)(F)C(=O)O",
                "inchikey": "DTBXFDXVGNOQPV-UHFFFAOYSA-N",
                "name": "TFA-2",
                "logP": 0.6,
                "logS": -0.5,
                "pKa": 0.6,
                "source": "source2",
            },
        ]
        stats = CurationStats()
        deduplicated = deduplicate_by_inchikey(records, stats)

        assert len(deduplicated) == 1
        merged = deduplicated[0]
        # Properties should be averaged
        assert merged["logP"] == 0.55  # (0.5 + 0.6) / 2
        assert merged["logS"] == -0.5  # Only one non-NaN value
        assert merged["pKa"] == 0.5  # (0.4 + 0.6) / 2

    def test_deduplicate_no_duplicates(self) -> None:
        """Test deduplication with no duplicates."""
        records = [
            {
                "smiles": "A",
                "inchikey": "KEY1",
                "name": "mol1",
                "logP": 1.0,
                "logS": 1.0,
                "pKa": 1.0,
                "source": "test",
            },
            {
                "smiles": "B",
                "inchikey": "KEY2",
                "name": "mol2",
                "logP": 2.0,
                "logS": 2.0,
                "pKa": 2.0,
                "source": "test",
            },
        ]
        stats = CurationStats()
        deduplicated = deduplicate_by_inchikey(records, stats)

        assert len(deduplicated) == 2
        assert stats.duplicates_removed == 0


class TestOutputGeneration:
    """Tests for output file generation."""

    def test_save_dataset_creates_csv(self, tmp_path: Path) -> None:
        """Test saving dataset creates valid CSV."""
        records = [
            {
                "smiles": "FC(F)(F)C(=O)O",
                "name": "TFA",
                "inchikey": "DTBXFDXVGNOQPV-UHFFFAOYSA-N",
                "logS": -0.5,
                "logP": 0.5,
                "pKa": 0.5,
                "source": "test",
            },
        ]
        output_path = tmp_path / "test_output.csv"
        save_dataset(records, output_path)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert df.iloc[0]["smiles"] == "FC(F)(F)C(=O)O"
        assert df.iloc[0]["logP"] == 0.5

    def test_save_dataset_column_order(self, tmp_path: Path) -> None:
        """Test that CSV has correct column order."""
        records = [
            {
                "smiles": "FC(F)(F)C(=O)O",
                "name": "TFA",
                "inchikey": "KEY",
                "logS": -0.5,
                "logP": 0.5,
                "pKa": 0.5,
                "source": "test",
            },
        ]
        output_path = tmp_path / "test_output.csv"
        save_dataset(records, output_path)

        df = pd.read_csv(output_path)
        expected_columns = ["smiles", "name", "inchikey", "logS", "logP", "pKa", "source"]
        assert list(df.columns) == expected_columns

    def test_save_report_creates_json(self, tmp_path: Path) -> None:
        """Test saving curation report creates valid JSON."""
        stats = CurationStats(
            input_count=100,
            valid_smiles=90,
            invalid_smiles=10,
            salt_stripped=5,
            duplicates_removed=3,
            final_count=87,
            logS_available=50,
            logP_available=80,
            pKa_available=30,
            all_props_available=10,
            sources=["file1.csv", "file2.csv"],
            errors=["Test error"],
            processing_time_sec=5.5,
        )
        report_path = tmp_path / "test_report.json"
        save_report(stats, report_path)

        assert report_path.exists()
        with open(report_path) as f:
            report = json.load(f)

        assert report["input"]["total_rows"] == 100
        assert report["processing"]["valid_smiles"] == 90
        assert report["output"]["final_count"] == 87
        assert len(report["input"]["sources"]) == 2


class TestStatisticsComputation:
    """Tests for statistics computation."""

    def test_curation_stats_initialization(self) -> None:
        """Test CurationStats default values."""
        stats = CurationStats()
        assert stats.input_count == 0
        assert stats.final_count == 0
        assert stats.sources == []
        assert stats.errors == []

    def test_curation_stats_tracking(self) -> None:
        """Test statistics are tracked during processing."""
        processor = MoleculeProcessor()

        # Process valid and invalid molecules
        processor.process_molecule("C(=O)(C(F)(F)F)O")  # Valid
        processor.process_molecule("invalid")  # Invalid

        assert processor.stats.valid_smiles == 1
        assert processor.stats.invalid_smiles == 1


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self, sample_comptox_csv: Path, tmp_path: Path) -> None:
        """Test the complete curation pipeline."""
        # Load and process
        df = load_comptox_export(sample_comptox_csv)
        processor = MoleculeProcessor()
        records = process_comptox_dataframe(df, processor, source_name="test.csv")

        # Deduplicate
        stats = processor.stats
        deduplicated = deduplicate_by_inchikey(records, stats)

        # Save
        output_path = tmp_path / "output.csv"
        save_dataset(deduplicated, output_path)

        # Verify output
        result_df = pd.read_csv(output_path)
        assert len(result_df) == 3  # 3 valid unique molecules
        assert "smiles" in result_df.columns
        assert "inchikey" in result_df.columns

        # Verify no duplicate InChIKeys
        assert result_df["inchikey"].nunique() == len(result_df)

    def test_pipeline_handles_empty_file(self, tmp_path: Path) -> None:
        """Test pipeline handles empty input gracefully."""
        empty_csv = tmp_path / "empty.csv"
        pd.DataFrame(columns=["SMILES", "PREFERRED_NAME"]).to_csv(empty_csv, index=False)

        df = load_comptox_export(empty_csv)
        processor = MoleculeProcessor()
        records = process_comptox_dataframe(df, processor, source_name="empty.csv")

        assert len(records) == 0
        assert processor.stats.input_count == 0

    def test_pipeline_preserves_nan_properties(
        self, sample_comptox_csv: Path, tmp_path: Path
    ) -> None:
        """Test that NaN properties are preserved through pipeline."""
        df = load_comptox_export(sample_comptox_csv)
        processor = MoleculeProcessor()
        records = process_comptox_dataframe(df, processor, source_name="test.csv")

        # PFBS has no pKa value
        pfbs_record = next(r for r in records if "PFBS" in r.get("name", ""))
        assert np.isnan(pfbs_record["pKa"])

        # Save and reload
        output_path = tmp_path / "output.csv"
        save_dataset(records, output_path)
        result_df = pd.read_csv(output_path)

        pfbs_row = result_df[result_df["name"] == "PFBS"].iloc[0]
        assert pd.isna(pfbs_row["pKa"])


class TestAPIClient:
    """Tests for the CompToxAPIClient."""

    @patch("requests.post")
    def test_fetch_physchem_properties_success(self, mock_post: MagicMock) -> None:
        """Test successful property fetching from API."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "dtxsid": "DTXSID001",
                "predictions": [
                    {"propertyName": "Water solubility", "value": -1.5},
                    {"propertyName": "LogP", "value": 2.5},
                    {"propertyName": "pKa", "value": 4.5},
                ],
            }
        ]
        mock_post.return_value = mock_response

        client = CompToxAPIClient(api_key="test_key")
        results = client.fetch_physchem_properties(["DTXSID001"])

        assert "DTXSID001" in results
        assert results["DTXSID001"]["logS"] == -1.5
        assert results["DTXSID001"]["logP"] == 2.5
        assert results["DTXSID001"]["pKa"] == 4.5
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_fetch_physchem_properties_failure(self, mock_post: MagicMock) -> None:
        """Test behavior when API request fails."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        client = CompToxAPIClient(api_key="test_key")
        results = client.fetch_physchem_properties(["DTXSID001"])

        assert results == {}
        mock_post.assert_called_once()


class TestPropertyComputation:
    """Tests for RDKit-based property computation."""

    def test_compute_logp_tfa(self) -> None:
        """Test logP computation for TFA (trifluoroacetic acid)."""
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        logp = compute_logp(mol)

        # TFA should have logP around 0.5-1.0 (small, polar, fluorinated)
        assert not np.isnan(logp)
        assert -2.0 < logp < 3.0  # Reasonable range for TFA

    def test_compute_logp_pfoa(self) -> None:
        """Test logP computation for PFOA (perfluorooctanoic acid)."""
        mol = Chem.MolFromSmiles("C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O")
        logp = compute_logp(mol)

        # PFOA should have higher logP due to longer chain
        assert not np.isnan(logp)
        assert logp > 0  # Should be positive (lipophilic)

    def test_compute_logp_invalid_mol(self) -> None:
        """Test logP returns NaN for None mol."""
        logp = compute_logp(None)  # type: ignore
        assert np.isnan(logp)

    def test_compute_esol_logs_tfa(self) -> None:
        """Test ESOL logS computation for TFA."""
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        logs = compute_esol_logs(mol)

        # TFA is quite soluble, expect negative but not too negative logS
        assert not np.isnan(logs)
        assert -5.0 < logs < 2.0  # Reasonable solubility range

    def test_compute_esol_logs_pfoa(self) -> None:
        """Test ESOL logS computation for PFOA."""
        mol = Chem.MolFromSmiles("C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O")
        logs = compute_esol_logs(mol)

        # PFOA should be less soluble than TFA
        assert not np.isnan(logs)
        # PFOA should have more negative logS (less soluble)
        tfa_mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        tfa_logs = compute_esol_logs(tfa_mol)
        assert logs < tfa_logs

    def test_compute_esol_logs_invalid_mol(self) -> None:
        """Test ESOL returns NaN for None mol."""
        logs = compute_esol_logs(None)  # type: ignore
        assert np.isnan(logs)

    def test_compute_properties_for_smiles_valid(self) -> None:
        """Test property computation from SMILES string."""
        props = compute_properties_for_smiles("C(=O)(C(F)(F)F)O")

        assert "logP" in props
        assert "logS" in props
        assert not np.isnan(props["logP"])
        assert not np.isnan(props["logS"])

    def test_compute_properties_for_smiles_invalid(self) -> None:
        """Test property computation for invalid SMILES."""
        props = compute_properties_for_smiles("not_a_valid_smiles")

        assert np.isnan(props["logP"])
        assert np.isnan(props["logS"])

    def test_compute_properties_for_smiles_empty(self) -> None:
        """Test property computation for empty SMILES.

        Note: RDKit treats empty SMILES as a valid empty molecule,
        which returns 0.0 for logP (no atoms to contribute).
        """
        props = compute_properties_for_smiles("")

        # Empty SMILES creates valid but empty molecule with logP=0
        assert "logP" in props
        assert "logS" in props

    def test_esol_formula_components(self) -> None:
        """Test that ESOL formula produces expected relative values.

        The ESOL equation is:
        logS = 0.16 - 0.63*logP - 0.0062*MW + 0.066*RB - 0.74*AP

        Higher MW and logP should decrease solubility.
        """
        # Simple molecules with increasing MW
        methane = Chem.MolFromSmiles("C")
        ethane = Chem.MolFromSmiles("CC")
        propane = Chem.MolFromSmiles("CCC")

        logs_methane = compute_esol_logs(methane)
        logs_ethane = compute_esol_logs(ethane)
        logs_propane = compute_esol_logs(propane)

        # Larger molecules should be less soluble (more negative logS)
        assert logs_methane > logs_ethane > logs_propane

    def test_logp_increases_with_chain_length(self) -> None:
        """Test that logP increases with carbon chain length."""
        logp_c2 = compute_logp(Chem.MolFromSmiles("CC"))
        logp_c4 = compute_logp(Chem.MolFromSmiles("CCCC"))
        logp_c8 = compute_logp(Chem.MolFromSmiles("CCCCCCCC"))

        # Longer chains should be more lipophilic
        assert logp_c2 < logp_c4 < logp_c8
