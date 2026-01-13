"""Tests for PyG Data conversion utilities."""

import pytest
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data

from gnn.data.transforms.to_pyg import mol_to_pyg_data, mols_to_pyg_batch

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_mol() -> Chem.Mol:
    """Create a simple ethanol molecule for testing."""
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None
    return mol


@pytest.fixture
def pfas_mol() -> Chem.Mol:
    """Create a PFAS molecule (PFOA) for testing."""
    # PFOA: Perfluorooctanoic acid
    mol = Chem.MolFromSmiles("C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O")
    assert mol is not None
    return mol


@pytest.fixture
def benzene_mol() -> Chem.Mol:
    """Create a benzene molecule for testing aromatic features."""
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    return mol


# ============================================================================
# Test: mol_to_pyg_data - Basic Conversion
# ============================================================================


class TestMolToPygData:
    """Tests for mol_to_pyg_data function."""

    def test_returns_data_object(self, simple_mol: Chem.Mol) -> None:
        """Test that conversion returns a PyG Data object."""
        data = mol_to_pyg_data(simple_mol)
        assert isinstance(data, Data)

    def test_data_has_atom_features(self, simple_mol: Chem.Mol) -> None:
        """Test data.x contains atom features tensor."""
        data = mol_to_pyg_data(simple_mol)
        assert hasattr(data, "x")
        assert isinstance(data.x, torch.Tensor)
        # Ethanol has 3 atoms (C, C, O)
        assert data.x.shape == (3, 22)
        assert data.x.dtype == torch.float32

    def test_data_has_edge_index(self, simple_mol: Chem.Mol) -> None:
        """Test data.edge_index contains bond connectivity (bidirectional)."""
        data = mol_to_pyg_data(simple_mol)
        assert hasattr(data, "edge_index")
        assert isinstance(data.edge_index, torch.Tensor)
        # Ethanol has 2 bonds * 2 directions = 4 edges
        assert data.edge_index.shape == (2, 4)
        assert data.edge_index.dtype == torch.long

    def test_data_has_edge_attr(self, simple_mol: Chem.Mol) -> None:
        """Test data.edge_attr contains bond features tensor."""
        data = mol_to_pyg_data(simple_mol)
        assert hasattr(data, "edge_attr")
        assert isinstance(data.edge_attr, torch.Tensor)
        # 4 edges (bidirectional), 12 bond features
        assert data.edge_attr.shape == (4, 12)
        assert data.edge_attr.dtype == torch.float32

    def test_data_has_num_nodes(self, simple_mol: Chem.Mol) -> None:
        """Test data.num_nodes equals number of atoms."""
        data = mol_to_pyg_data(simple_mol)
        assert data.num_nodes == 3

    def test_data_has_smiles(self, simple_mol: Chem.Mol) -> None:
        """Test data.smiles contains canonical SMILES string."""
        data = mol_to_pyg_data(simple_mol)
        assert hasattr(data, "smiles")
        assert isinstance(data.smiles, str)
        # Canonical SMILES for ethanol
        assert data.smiles == "CCO"


# ============================================================================
# Test: mol_to_pyg_data - 3D Positions
# ============================================================================


class TestMolToPygDataWith3D:
    """Tests for mol_to_pyg_data with 3D positions."""

    def test_pos_not_present_by_default(self, simple_mol: Chem.Mol) -> None:
        """Test that pos is not included when include_pos=False."""
        data = mol_to_pyg_data(simple_mol, include_pos=False)
        assert not hasattr(data, "pos") or data.pos is None

    def test_pos_included_when_requested(self, simple_mol: Chem.Mol) -> None:
        """Test data.pos contains 3D positions when include_pos=True."""
        data = mol_to_pyg_data(simple_mol, include_pos=True, pos_random_seed=0)
        assert hasattr(data, "pos")
        assert isinstance(data.pos, torch.Tensor)
        # 3 atoms, 3D positions
        assert data.pos.shape == (3, 3)
        assert data.pos.dtype == torch.float32

    def test_pos_matches_atom_count(self, benzene_mol: Chem.Mol) -> None:
        """Test pos tensor has correct number of atoms."""
        data = mol_to_pyg_data(benzene_mol, include_pos=True, pos_random_seed=0)
        assert data.pos.shape[0] == benzene_mol.GetNumAtoms()
        assert data.pos.shape[1] == 3


# ============================================================================
# Test: mol_to_pyg_data - Target Values
# ============================================================================


class TestMolToPygDataWithTarget:
    """Tests for mol_to_pyg_data with target values."""

    def test_y_not_present_by_default(self, simple_mol: Chem.Mol) -> None:
        """Test that y is not included when not provided."""
        data = mol_to_pyg_data(simple_mol)
        assert not hasattr(data, "y") or data.y is None

    def test_y_from_float(self, simple_mol: Chem.Mol) -> None:
        """Test y is set correctly from float value."""
        data = mol_to_pyg_data(simple_mol, y=1.5)
        assert hasattr(data, "y")
        assert isinstance(data.y, torch.Tensor)
        assert data.y.shape == (1,)
        assert data.y.dtype == torch.float32
        assert data.y.item() == pytest.approx(1.5)

    def test_y_from_int(self, simple_mol: Chem.Mol) -> None:
        """Test y is set correctly from int value."""
        data = mol_to_pyg_data(simple_mol, y=2)
        assert data.y.shape == (1,)
        assert data.y.item() == pytest.approx(2.0)

    def test_y_from_tensor(self, simple_mol: Chem.Mol) -> None:
        """Test y is set correctly from tensor."""
        target = torch.tensor([1.0, 2.0, 3.0])
        data = mol_to_pyg_data(simple_mol, y=target)
        assert hasattr(data, "y")
        assert data.y.shape == (1, 3)
        assert torch.allclose(data.y, target.unsqueeze(0))


# ============================================================================
# Test: mol_to_pyg_data - PFAS Molecules
# ============================================================================


class TestMolToPygDataPFAS:
    """Tests for mol_to_pyg_data with PFAS molecules."""

    def test_pfas_conversion(self, pfas_mol: Chem.Mol) -> None:
        """Test conversion works for PFAS molecule."""
        data = mol_to_pyg_data(pfas_mol)
        assert isinstance(data, Data)
        assert data.x.shape[0] == pfas_mol.GetNumAtoms()
        assert data.x.shape[1] == 22

    def test_pfas_with_3d(self, pfas_mol: Chem.Mol) -> None:
        """Test PFAS conversion with 3D positions."""
        data = mol_to_pyg_data(pfas_mol, include_pos=True, pos_random_seed=0)
        assert data.pos.shape == (pfas_mol.GetNumAtoms(), 3)


# ============================================================================
# Test: mols_to_pyg_batch - Batch Conversion
# ============================================================================


class TestMolsToPygBatch:
    """Tests for mols_to_pyg_batch function."""

    def test_returns_batch_object(self, simple_mol: Chem.Mol) -> None:
        """Test that batch conversion returns PyG Batch object."""
        mols = [simple_mol, simple_mol]
        batch = mols_to_pyg_batch(mols)
        assert isinstance(batch, Batch)

    def test_batch_has_concatenated_features(
        self, simple_mol: Chem.Mol, benzene_mol: Chem.Mol
    ) -> None:
        """Test batch.x contains concatenated atom features."""
        mols = [simple_mol, benzene_mol]
        batch = mols_to_pyg_batch(mols)

        # Ethanol: 3 atoms, Benzene: 6 atoms = 9 total
        expected_atoms = simple_mol.GetNumAtoms() + benzene_mol.GetNumAtoms()
        assert batch.x.shape == (expected_atoms, 22)

    def test_batch_has_batch_vector(self, simple_mol: Chem.Mol, benzene_mol: Chem.Mol) -> None:
        """Test batch.batch contains atom-to-graph assignment."""
        mols = [simple_mol, benzene_mol]
        batch = mols_to_pyg_batch(mols)

        assert hasattr(batch, "batch")
        # First 3 atoms belong to graph 0, next 6 to graph 1
        expected_batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
        assert torch.equal(batch.batch, expected_batch)

    def test_batch_has_ptr(self, simple_mol: Chem.Mol, benzene_mol: Chem.Mol) -> None:
        """Test batch.ptr contains graph boundaries."""
        mols = [simple_mol, benzene_mol]
        batch = mols_to_pyg_batch(mols)

        assert hasattr(batch, "ptr")
        # ptr = [0, 3, 9] for 2 graphs with 3 and 6 atoms
        expected_ptr = torch.tensor([0, 3, 9])
        assert torch.equal(batch.ptr, expected_ptr)

    def test_batch_with_targets(self, simple_mol: Chem.Mol) -> None:
        """Test batch conversion with target values."""
        mols = [simple_mol, simple_mol, simple_mol]
        ys = [1.0, 2.0, 3.0]
        batch = mols_to_pyg_batch(mols, ys=ys)

        assert hasattr(batch, "y")
        # PyG Batch concatenates y tensors: 3 molecules with y shape (1,) -> (3,)
        expected_y = torch.tensor([1.0, 2.0, 3.0])
        assert torch.allclose(batch.y, expected_y)

    def test_batch_with_3d_positions(self, simple_mol: Chem.Mol, benzene_mol: Chem.Mol) -> None:
        """Test batch conversion with 3D positions."""
        mols = [simple_mol, benzene_mol]
        batch = mols_to_pyg_batch(mols, include_pos=True, pos_random_seed=0)

        expected_atoms = simple_mol.GetNumAtoms() + benzene_mol.GetNumAtoms()
        assert batch.pos.shape == (expected_atoms, 3)


# ============================================================================
# Test: Batch Compatibility
# ============================================================================


class TestBatchCompatibility:
    """Tests verifying Data objects work with Batch.from_data_list()."""

    def test_data_can_be_batched_manually(
        self, simple_mol: Chem.Mol, benzene_mol: Chem.Mol
    ) -> None:
        """Test that individual Data objects can be batched with PyG."""
        data1 = mol_to_pyg_data(simple_mol)
        data2 = mol_to_pyg_data(benzene_mol)

        # This should work without errors
        batch = Batch.from_data_list([data1, data2])

        assert isinstance(batch, Batch)
        assert batch.num_graphs == 2

    def test_batched_data_preserves_smiles(
        self, simple_mol: Chem.Mol, benzene_mol: Chem.Mol
    ) -> None:
        """Test that smiles metadata is preserved in batch."""
        data1 = mol_to_pyg_data(simple_mol)
        data2 = mol_to_pyg_data(benzene_mol)

        batch = Batch.from_data_list([data1, data2])

        # smiles should be preserved as a list in the batch
        assert hasattr(batch, "smiles")
        assert len(batch.smiles) == 2

    def test_empty_list_handling(self) -> None:
        """Test handling of empty molecule list."""
        with pytest.raises(ValueError):
            mols_to_pyg_batch([])
