"""Production mixture checkpoints remain loadable without remote tokenizers."""

import json
import pytest
import torch
from transformers import RobertaPreLayerNormConfig

from electrolyte_fm.models.prod_mixture import (
    MISTExcessPhysics,
    MISTExcessPhysicsConfig,
    MISTIonicConductivity,
    MISTIonicConductivityConfig,
    MISTMixtures,
    MISTMixturesConfig,
)


@pytest.mark.parametrize(
    "model_class,config_class,kwargs",
    [
        (MISTExcessPhysics, MISTExcessPhysicsConfig, {}),
        (
            MISTIonicConductivity,
            MISTIonicConductivityConfig,
            {"n_components": 5, "task_network": {"kwargs": {"embed_dim": 8}}},
        ),
        (
            MISTMixtures,
            MISTMixturesConfig,
            {
                "task_network": {"embed_dim": 8, "output_size": 2, "dropout": 0.0},
                "temperature_condition": "concat",
            },
        ),
    ],
)
def test_mixture_model_loads_legacy_release_without_tokenizer_files(
    tmp_path, monkeypatch, model_class, config_class, kwargs
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    encoder = RobertaPreLayerNormConfig(
        vocab_size=165,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=128,
    )
    model = model_class(config_class(encoder=encoder.to_dict(), **kwargs)).eval()
    model.save_pretrained(tmp_path)
    # Older published excess-property models omit all tokenizer artifacts.
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        (tmp_path / filename).unlink(missing_ok=True)
    config = json.loads((tmp_path / "config.json").read_text())
    config["tokenizer_class"] = "SmirkTokenizer"
    (tmp_path / "config.json").write_text(json.dumps(config))
    loaded = model_class.from_pretrained(tmp_path, local_files_only=True).eval()
    assert loaded.tokenizer("CCO")["input_ids"] == model.tokenizer("CCO")["input_ids"]
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
