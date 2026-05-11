"""Tests for models in the mist-models HuggingFace organization."""

from contextlib import contextmanager
import gc
import logging
import os
from pathlib import Path
import tempfile
from typing import Iterator

import pytest
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModel

from .test_inference import (
    single_molecule_smiles,
    conductivity_test_data,
    excess_test_data,
    validate_predictions,
    get_model_type_from_path,
    check_multi_channel_labels,
)

logger = logging.getLogger(__name__)


@contextmanager
def loaded_hf_model(model_id: str, hf_token: str | None) -> Iterator[object]:
    """Load one HF model in an isolated cache that is removed after the check."""
    with tempfile.TemporaryDirectory(
        prefix="hf-model-cache-", dir=os.getenv("RUNNER_TEMP") or None
    ) as cache_dir:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
            cache_dir=cache_dir,
        )
        try:
            yield model
        finally:
            del model
            gc.collect()


def load_hf_config(model_id: str, hf_token: str | None):
    with tempfile.TemporaryDirectory(
        prefix="hf-config-cache-", dir=os.getenv("RUNNER_TEMP") or None
    ) as cache_dir:
        return AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
            cache_dir=cache_dir,
        )


@pytest.fixture
def hf_token():
    # Not testing private models for now
    # (storage limits will cause errors)
    return os.getenv("HF_TOKEN")


@pytest.fixture
def hf_org_models(hf_token):
    api = HfApi(token=hf_token)
    models = list(api.list_models(author="mist-models"))

    if not models:
        pytest.skip("No models found in mist-models organization")

    model_ids = [m.id for m in models]
    logger.info(f"Found {len(model_ids)} models in mist-models organization")
    return model_ids


class TestHFOrgSingleMoleculeModels:
    def test_predict_single_molecules(
        self, hf_org_models, hf_token, single_molecule_smiles
    ):
        single_models = [
            m for m in hf_org_models if get_model_type_from_path(Path(m)) == "single"
        ]

        for model_id in single_models:
            logger.info(f"Testing {model_id}")
            with loaded_hf_model(model_id, hf_token) as model:
                if "RobertaPreLayerNormModel" in type(model).__name__:
                    logger.info("Skipping encoder-only model")
                    continue

                predictions = model.predict(single_molecule_smiles)
                assert predictions is not None

                if isinstance(predictions, dict):
                    assert len(predictions) > 0
                    for task_name, task_data in predictions.items():
                        if isinstance(task_data, dict) and "value" in task_data:
                            values = task_data["value"]
                            assert len(values) == len(single_molecule_smiles)
                            validate_predictions(values, name=f"{model_id}:{task_name}")
                else:
                    assert len(predictions) == len(single_molecule_smiles)
                    validate_predictions(predictions, name=model_id)


class TestHFOrgConductivityModels:
    def test_predict_mixtures(self, hf_org_models, hf_token, conductivity_test_data):
        cond_models = [
            m
            for m in hf_org_models
            if get_model_type_from_path(Path(m)) == "conductivity"
        ]
        for model_id in cond_models:
            logger.info(f"Testing {model_id}")
            with loaded_hf_model(model_id, hf_token) as model:
                predictions = model.predict(conductivity_test_data)
                assert predictions is not None

                if isinstance(predictions, dict):
                    for key, value in predictions.items():
                        validate_predictions(value, name=f"{model_id}:{key}")
                else:
                    validate_predictions(predictions, name=model_id)


class TestHFOrgExcessPhysicsModels:
    def test_predict_binary_mixture(self, hf_org_models, hf_token, excess_test_data):
        excess_models = [
            m for m in hf_org_models if get_model_type_from_path(Path(m)) == "excess"
        ]
        test_case = excess_test_data[0]

        for model_id in excess_models:
            logger.info(f"Testing {model_id}")
            with loaded_hf_model(model_id, hf_token) as model:
                predictions = model.predict(
                    smiles_list=test_case["smiles_list"],
                    composition=test_case["composition"],
                    temperature=test_case["temperature"],
                )

                assert predictions is not None

                if isinstance(predictions, dict):
                    for key, value in predictions.items():
                        validate_predictions(value, name=f"{model_id}:{key}")


class TestHFOrgModelIntegrity:
    def test_all_models_config(self, hf_org_models, hf_token):
        for model_id in hf_org_models:
            logger.info(f"Checking config for {model_id}")
            config = load_hf_config(model_id, hf_token)
            assert config is not None

    def test_models_required_files(self, hf_org_models, hf_token):
        api = HfApi(token=hf_token)

        for model_id in hf_org_models:
            model_info = api.model_info(model_id)
            siblings = {f.rfilename for f in model_info.siblings}

            assert "config.json" in siblings, f"{model_id} missing config.json"
            assert "README.md" in siblings, f"{model_id} missing README.md"

            has_weights = any(
                "safetensors" in f or "pytorch_model.bin" in f for f in siblings
            )
            assert has_weights, f"{model_id} missing model weights"

    def test_multi_channel_model_labels(self, hf_org_models, hf_token):
        for model_id in hf_org_models:
            logger.info(f"Checking channels for {model_id}")
            config = load_hf_config(model_id, hf_token)
            check_multi_channel_labels(config, model_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
