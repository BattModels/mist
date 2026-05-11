"""Tests for models in the mist-models HuggingFace organization."""

from contextlib import contextmanager
from functools import lru_cache
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
HF_ORG = "mist-models"


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


@lru_cache
def list_hf_org_model_ids(hf_token: str | None) -> tuple[str, ...]:
    api = HfApi(token=hf_token)
    model_ids = tuple(m.id for m in api.list_models(author=HF_ORG))
    logger.info("Found %d models in %s organization", len(model_ids), HF_ORG)
    return model_ids


def parametrize_model_ids(metafunc, fixture_name: str, model_ids: tuple[str, ...]):
    if model_ids:
        metafunc.parametrize(
            fixture_name, model_ids, ids=lambda m: m.rsplit("/", 1)[-1]
        )
        return

    metafunc.parametrize(
        fixture_name,
        [
            pytest.param(
                None,
                marks=pytest.mark.skip(reason=f"No {fixture_name} models found"),
            )
        ],
        ids=["no-models"],
    )


def pytest_generate_tests(metafunc):
    model_ids = list_hf_org_model_ids(os.getenv("HF_TOKEN"))

    if "hf_model_id" in metafunc.fixturenames:
        parametrize_model_ids(metafunc, "hf_model_id", model_ids)

    if "single_model_id" in metafunc.fixturenames:
        parametrize_model_ids(
            metafunc,
            "single_model_id",
            tuple(
                m for m in model_ids if get_model_type_from_path(Path(m)) == "single"
            ),
        )

    if "conductivity_model_id" in metafunc.fixturenames:
        parametrize_model_ids(
            metafunc,
            "conductivity_model_id",
            tuple(
                m
                for m in model_ids
                if get_model_type_from_path(Path(m)) == "conductivity"
            ),
        )

    if "excess_model_id" in metafunc.fixturenames:
        parametrize_model_ids(
            metafunc,
            "excess_model_id",
            tuple(
                m for m in model_ids if get_model_type_from_path(Path(m)) == "excess"
            ),
        )


@pytest.fixture
def hf_token():
    # Not testing private models for now
    # (storage limits will cause errors)
    return os.getenv("HF_TOKEN")


class TestHFOrgSingleMoleculeModels:
    def test_predict_single_molecules(
        self, single_model_id, hf_token, single_molecule_smiles
    ):
        logger.info(f"Testing {single_model_id}")
        with loaded_hf_model(single_model_id, hf_token) as model:
            if "RobertaPreLayerNormModel" in type(model).__name__:
                pytest.skip("Skipping encoder-only model")

            predictions = model.predict(single_molecule_smiles)
            assert predictions is not None

            if isinstance(predictions, dict):
                assert len(predictions) > 0
                for task_name, task_data in predictions.items():
                    if isinstance(task_data, dict) and "value" in task_data:
                        values = task_data["value"]
                        assert len(values) == len(single_molecule_smiles)
                        validate_predictions(
                            values, name=f"{single_model_id}:{task_name}"
                        )
            else:
                assert len(predictions) == len(single_molecule_smiles)
                validate_predictions(predictions, name=single_model_id)


class TestHFOrgConductivityModels:
    def test_predict_mixtures(
        self, conductivity_model_id, hf_token, conductivity_test_data
    ):
        logger.info(f"Testing {conductivity_model_id}")
        with loaded_hf_model(conductivity_model_id, hf_token) as model:
            predictions = model.predict(conductivity_test_data)
            assert predictions is not None

            if isinstance(predictions, dict):
                for key, value in predictions.items():
                    validate_predictions(value, name=f"{conductivity_model_id}:{key}")
            else:
                validate_predictions(predictions, name=conductivity_model_id)


class TestHFOrgExcessPhysicsModels:
    def test_predict_binary_mixture(self, excess_model_id, hf_token, excess_test_data):
        test_case = excess_test_data[0]
        logger.info(f"Testing {excess_model_id}")
        with loaded_hf_model(excess_model_id, hf_token) as model:
            predictions = model.predict(
                smiles_list=test_case["smiles_list"],
                composition=test_case["composition"],
                temperature=test_case["temperature"],
            )

            assert predictions is not None

            if isinstance(predictions, dict):
                for key, value in predictions.items():
                    validate_predictions(value, name=f"{excess_model_id}:{key}")


class TestHFOrgModelIntegrity:
    def test_all_models_config(self, hf_model_id, hf_token):
        logger.info(f"Checking config for {hf_model_id}")
        config = load_hf_config(hf_model_id, hf_token)
        assert config is not None

    def test_models_required_files(self, hf_model_id, hf_token):
        api = HfApi(token=hf_token)
        model_info = api.model_info(hf_model_id)
        siblings = {f.rfilename for f in model_info.siblings}

        assert "config.json" in siblings, f"{hf_model_id} missing config.json"
        assert "README.md" in siblings, f"{hf_model_id} missing README.md"

        has_weights = any(
            "safetensors" in f or "pytorch_model.bin" in f for f in siblings
        )
        assert has_weights, f"{hf_model_id} missing model weights"

    def test_multi_channel_model_labels(self, hf_model_id, hf_token):
        logger.info(f"Checking channels for {hf_model_id}")
        config = load_hf_config(hf_model_id, hf_token)
        check_multi_channel_labels(config, hf_model_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--log-cli-level=INFO"])
