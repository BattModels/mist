"""SCScore fingerprints remain identical without NumPy's unsafe conversion."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


@pytest.fixture
def scorer():
    source = Path(__file__).resolve().parents[1] / "vendor/scscore/scscore.py"
    spec = importlib.util.spec_from_file_location("scscore_fingerprint", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SCScorer()


@pytest.mark.parametrize("smiles", ["CCO", "c1ccccc1", "N[C@@H](C)C(=O)O"])
def test_bit_fingerprint_matches_rdkit(scorer, smiles):
    molecule = Chem.MolFromSmiles(smiles)
    expected = AllChem.GetMorganFingerprintAsBitVect(
        molecule, scorer.FP_rad, nBits=scorer.FP_len, useChirality=True
    )
    actual = scorer.mol_to_fp(molecule)
    assert actual.dtype == np.dtype(bool)
    assert actual.shape == (scorer.FP_len,)
    assert np.flatnonzero(actual).tolist() == list(expected.GetOnBits())


def test_fingerprint_can_be_scored_without_downloading_weights(scorer):
    scorer.vars = [np.zeros((scorer.FP_len, 1)), np.zeros(1)]
    scorer._restored = True
    smiles, score = scorer.get_score_from_smi("CCO")
    assert smiles == "CCO"
    assert score == 3.0
