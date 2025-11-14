import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import pytest
import torch
import numpy as np
from transformers import AutoModel
from smirk import SmirkTokenizerFast

logger = logging.getLogger(__name__)


def validate_predictions(
    predictions: Union[torch.Tensor, np.ndarray, Dict], name: str = "predictions"
):
    """
    Validate that predictions are not NaN and not all zeros.
    """
    if isinstance(predictions, dict):
        for key, value in predictions.items():
            validate_predictions(value, name=f"{name}[{key}]")
        return

    if isinstance(predictions, dict) and "value" in predictions:
        predictions = predictions["value"]

    pred_array = np.array(predictions)

    if np.isnan(pred_array).any():
        raise AssertionError(f"{name} contains NaN values")

    if np.all(pred_array == 0.0):
        raise AssertionError(f"{name} are all zeros")

    logger.info(f"{name} validation passed: no NaN, not all zeros")


def pytest_addoption(parser):
    parser.addoption(
        "--hf-models-dir",
        action="store",
        default="hf-models",
        help="Directory containing packaged models",
    )


@pytest.fixture
def single_molecule_smiles() -> List[str]:
    return [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "O=C1OCCO1",  # Ethylene carbonate
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",  # Ibuprofen
    ]


@pytest.fixture
def conductivity_test_data() -> List[Dict[str, Any]]:
    return [
        {
            "solvent_composition": [{"CC1COC(=O)O1": 0.9}],
            "cation": "[Li+]",
            "anion": "F[P-](F)(F)(F)(F)F",
            "temperature": 298.15,
        },
        {
            "solvent_composition": [
                {
                    "CC1COC(=O)O1": 0.3,
                    "O=C1OCC(F)O1": 0.6,
                }
            ],
            "cation": "[Li+]",
            "anion": "F[P-](F)(F)(F)(F)F",
            "temperature": 313.0,
        },
    ]


@pytest.fixture
def excess_test_data() -> List[Dict[str, Any]]:
    return [
        {
            "smiles_list": [["CCO", "CCOC(=O)C"]],  # Binary mixture
            "composition": [[0.5, 0.5]],
            "temperature": [298.15],
        }
    ]


@pytest.fixture
def hf_models_base() -> Path:
    import os

    return Path(
        os.environ.get(
            "HF_MODELS_DIR", "/nfs/turbo/coe-venkvis/abhutani/electrolyte-fm/hf-models"
        )
    )


@pytest.fixture
def all_model_paths(hf_models_base) -> List[Path]:
    if not hf_models_base.exists():
        pytest.skip(f"Models directory not found: {hf_models_base}")
    paths = [p for p in hf_models_base.glob("mist-*-*") if p.is_dir()]
    if not paths:
        pytest.skip(f"No models found in {hf_models_base}")
    return sorted(paths)


def get_model_type_from_path(path: Path) -> str:
    name = path.name.lower()
    if "conductivity" in name or "cond" in name:
        return "conductivity"
    elif "excess" in name or "mix" in name:
        return "excess"
    else:
        return "single"


class TestSingleMoleculeModels:
    def test_model_loads(self, all_model_paths):
        single_paths = [
            p for p in all_model_paths if get_model_type_from_path(p) == "single"
        ]
        if not single_paths:
            pytest.skip("No single-molecule models found")

        for model_path in single_paths:
            logger.info(f"Loading model from {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            assert model is not None

    def test_predict_single_molecules(self, all_model_paths, single_molecule_smiles):
        single_paths = [
            p for p in all_model_paths if get_model_type_from_path(p) == "single"
        ]
        if not single_paths:
            pytest.skip("No single-molecule models found")

        for model_path in single_paths:
            logger.info(f"Testing predictions for {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            model_type = type(model).__name__

            # Skip encoder-only models
            if "RobertaPreLayerNormModel" in model_type:
                logger.info(f"Skipping encoder-only model {model_type}")
                continue

            predictions = model.predict(single_molecule_smiles)
            assert predictions is not None

            if isinstance(predictions, dict):
                assert len(predictions) > 0
                for task_name, task_data in predictions.items():
                    assert task_data is not None
                    if isinstance(task_data, dict) and "value" in task_data:
                        values = task_data["value"]
                        assert len(values) == len(single_molecule_smiles)
                        validate_predictions(
                            values, name=f"{model_path.name}:{task_name}"
                        )
            else:
                assert len(predictions) == len(single_molecule_smiles)
                validate_predictions(predictions, name=f"{model_path.name}")


class TestConductivityModels:
    def test_model_loads(self, all_model_paths):
        conductivity_paths = [
            p for p in all_model_paths if get_model_type_from_path(p) == "conductivity"
        ]
        if not conductivity_paths:
            pytest.skip("No conductivity models found")

        for model_path in conductivity_paths:
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            assert model is not None
            assert (
                "Conductivity" in type(model).__name__
                or "conductivity" in model_path.name.lower()
            )

    @pytest.mark.parametrize("test_case_idx", [0, 1])
    def test_predict_mixtures(
        self, all_model_paths, conductivity_test_data, test_case_idx
    ):
        conductivity_paths = [
            p for p in all_model_paths if get_model_type_from_path(p) == "conductivity"
        ]
        if not conductivity_paths:
            pytest.skip("No conductivity models found")

        test_case = conductivity_test_data[test_case_idx]

        for model_path in conductivity_paths:
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

            predictions = model.predict(
                solvent_composition=test_case["solvent_composition"],
                cation=test_case["cation"],
                anion=test_case["anion"],
                temperature=test_case["temperature"],
            )

            assert predictions is not None

            if isinstance(predictions, dict):
                assert len(predictions) > 0
                for key, value in predictions.items():
                    assert value is not None
                    validate_predictions(value, name=f"{model_path.name}:{key}")
            else:
                if hasattr(predictions, "shape"):
                    logger.info(f"Prediction shape: {predictions.shape}")
                assert predictions is not None
                validate_predictions(predictions, name=f"{model_path.name}")


class TestExcessPhysicsModels:
    def test_model_loads(self, all_model_paths):
        excess_paths = [
            p for p in all_model_paths if get_model_type_from_path(p) == "excess"
        ]
        if not excess_paths:
            pytest.skip("No excess physics models found")

        for model_path in excess_paths:
            logger.info(f"Loading excess physics model from {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            assert model is not None
            logger.info(f"Model loaded: {type(model).__name__}")

    def test_predict_binary_mixture(self, all_model_paths, excess_test_data):
        excess_paths = [
            p for p in all_model_paths if get_model_type_from_path(p) == "excess"
        ]
        if not excess_paths:
            pytest.skip("No excess physics models found")

        test_case = excess_test_data[0]

        for model_path in excess_paths:
            logger.info(f"Testing binary mixture for {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

            predictions = model.predict(
                smiles_list=test_case["smiles_list"],
                composition=test_case["composition"],
                temperature=test_case["temperature"],
            )

            assert predictions is not None
            logger.info(f"Prediction type: {type(predictions)}")

            if isinstance(predictions, dict):
                assert len(predictions) > 0
                for key, value in predictions.items():
                    logger.info(f"Output {key}: {type(value)}")
                    if hasattr(value, "shape"):
                        logger.info(f"Shape: {value.shape}")
                    assert value is not None
                    validate_predictions(value, name=f"{model_path.name}:{key}")


class TestEncoderModels:
    def test_encoder_forward_pass(self, all_model_paths, single_molecule_smiles):
        encoder_found = False

        for model_path in all_model_paths:
            logger.info(f"Testing encoder model {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            model_type = type(model).__name__

            if "RobertaPreLayerNormModel" not in model_type:
                continue

            encoder_found = True
            tok = SmirkTokenizerFast()
            inputs = tok(single_molecule_smiles, padding="longest", return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            assert outputs is not None
            assert hasattr(outputs, "last_hidden_state")
            logger.info(f"last_hidden_state shape: {outputs.last_hidden_state.shape}")
            assert outputs.last_hidden_state.shape[0] == len(single_molecule_smiles)
            validate_predictions(
                outputs.last_hidden_state, name=f"{model_path.name}:last_hidden_state"
            )

        if not encoder_found:
            pytest.skip("No encoder-only models found")


class TestModelIntegrity:
    def test_model_has_config(self, all_model_paths):
        for model_path in all_model_paths:
            logger.info(f"Testing config for {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

            assert hasattr(model, "config")
            assert model.config is not None
            logger.info(f"Model config type: {type(model.config).__name__}")

    def test_model_has_tokenizer(self, all_model_paths):
        for model_path in all_model_paths:
            logger.info(f"Testing tokenizer for {model_path.name}")
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)

            # Some models store tokenizer in config, others as attribute
            has_tokenizer = hasattr(model, "tokenizer") or hasattr(
                model.config, "tokenizer_class"
            )
            assert has_tokenizer

    def test_model_device_compatibility(self, all_model_paths):
        for model_path in all_model_paths:
            model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
            model = model.to("cpu")
            assert model.device.type == "cpu"
            if torch.cuda.is_available():
                model = model.to("cuda")
                assert model.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
