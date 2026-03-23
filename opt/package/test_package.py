import logging
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoModel, DataCollatorWithPadding

from electrolyte_fm.models.prod_finetune import MISTFinetuned, MISTMultiTask
from electrolyte_fm.models.prod_mixture import MISTIonicConductivity

# These should be outputs from running the packaging commands:
#   python -m opt.package finetuned <checkpoint_path>
#   python -m opt.package multitask <encoder_ckpt> --task-ckpt <task1> --task-ckpt <task2>
#   python -m opt.package conductivity <checkpoint_path>
FINETUNED_CKPT = Path(__file__).parent.joinpath()
MULTITASK_CKPT = Path(__file__).parent.joinpath()
ENCODER_CKPT = Path("mist-ti624ev1").resolve()


@pytest.fixture()
def finetuned_model():
    return MISTFinetuned.from_pretrained(
        str(FINETUNED_CKPT), trust_remote_code=True
    ).eval()


@pytest.fixture()
def multitask_model():
    return MISTMultiTask.from_pretrained(
        str(MULTITASK_CKPT), trust_remote_code=True
    ).eval()


@pytest.fixture()
def encoder_model():
    return AutoModel.from_pretrained(
        str(ENCODER_CKPT),
        trust_remote_code=True,
        add_pooling_layer=False,
    )


@pytest.mark.parametrize("model", ["finetuned_model", "multitask_model"])
def test_predict(model, request):
    model = request.getfixturevalue(model)
    assert isinstance(model.channels, list)
    out = model.predict(["CCCCC", "CC"])
    assert isinstance(out, dict)
    for chn in model.channels:
        assert out[chn["name"]]["value"].shape == (2,)


def check_weights_match(model1, model2):
    matches = True
    for (n1, w1), (n2, w2) in zip(model1, model2):
        if not w1.isclose(w2).all():
            matches = False
            assert n1 == n2
            logging.warning(
                f"Parameter {n1} do not match, shape: {w1.shape} vs. {w2.shape}, type: {w1.dtype} vs. {w2.dtype}"
            )

    assert matches


def test_encoder_weights(finetuned_model, multitask_model):
    check_weights_match(
        finetuned_model.encoder.named_parameters(),
        multitask_model.encoder.named_parameters(),
    )


def test_task_weights(finetuned_model, multitask_model):
    check_weights_match(
        finetuned_model.task_network.named_parameters(),
        multitask_model.task_networks[0].named_parameters(),
    )


def test_encoder_hs(finetuned_model, multitask_model):
    batch = ["Clc1ccccc1C2(NC)CCCCC2=O", "O(c1cc(cc(OC)c1OC)CCN)C"]
    hs_ft = finetuned_model.embed(batch)
    hs_mt = multitask_model.embed(batch)
    assert hs_ft.isclose(hs_mt).all()


@pytest.mark.parametrize("model", ["finetuned_model", "multitask_model"])
def test_ref_encoder(model, encoder_model, request):
    model = request.getfixturevalue(model)
    encoder_model = encoder_model.to(model.encoder.device)

    # Prepare batch
    batch = ["Clc1ccccc1C2(NC)CCCCC2=O", "O(c1cc(cc(OC)c1OC)CCN)C"]
    batch = model.tokenizer(batch)
    collate_fn = DataCollatorWithPadding(model.tokenizer)
    batch = collate_fn(batch)
    input_ids = batch["input_ids"].to(model.encoder.device)
    attention_mask = batch["attention_mask"].to(model.encoder.device)

    with torch.inference_mode():
        hs = model.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        hs_ref = encoder_model(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state

    assert hs.isclose(hs_ref).all()


def test_task_networks(finetuned_model, multitask_model):
    hidden_size = finetuned_model.encoder.config.hidden_size
    hs = torch.rand(4, 3, hidden_size).to(finetuned_model.encoder.device)
    with torch.inference_mode():
        y_ft = finetuned_model.task_network(hs.detach())
        y_mt = multitask_model.task_networks[0](hs.detach())

    # Check if stochastic
    with torch.inference_mode():
        y_ft_2 = finetuned_model.task_network(hs)
        y_mt_2 = multitask_model.task_networks[0](hs)
    assert y_ft_2.isclose(y_ft).all()
    assert y_mt_2.isclose(y_mt).all()

    assert y_ft.isclose(y_mt).all()


def get_weights(state, prefix):
    for k, v in state.items():
        if k.startswith(prefix):
            yield k, v


def test_safetensors_match():
    st_ft = load_file(FINETUNED_CKPT.joinpath("model.safetensors"))
    st_mt = load_file(MULTITASK_CKPT.joinpath("model.safetensors"))

    check_weights_match(get_weights(st_ft, "encoder"), get_weights(st_mt, "encoder"))
    check_weights_match(
        get_weights(st_ft, "task_network"), get_weights(st_mt, "task_networks.0.")
    )


@pytest.mark.parametrize(
    "model,pkgdir",
    [("finetuned_model", FINETUNED_CKPT), ("multitask_model", MULTITASK_CKPT)],
)
def test_weights_loaded(model, pkgdir, request):
    model = request.getfixturevalue(model)
    st = load_file(pkgdir.joinpath("model.safetensors"))
    matches = True
    for name, param in model.named_parameters():
        w_ckpt = st[name]
        if not w_ckpt.isclose(param).all():
            matches = False
            logging.warning(
                f"Parameter {name} do not match, shape: {param.shape} vs. {w_ckpt.shape}, type: {param.dtype} vs. {w_ckpt.dtype}"
            )

    assert matches


@pytest.mark.parametrize("model", ["finetuned_model", "multitask_model"])
def test_encoder_config(model, request):
    model = request.getfixturevalue(model)
    config = model.encoder.config
    if hasattr(config, "add_pooling_layer"):
        assert config.add_pooling_layer is False


def test_export_multitask_preserves_tasks():
    """Test that export_multitask preserves task_networks and transforms
    when loading an already-packaged MISTMultiTask model."""
    from opt.package.__main__ import export_multitask

    # Load an already-packaged multitask model with empty task_ckpt
    model = export_multitask(MULTITASK_CKPT, task_ckpt=[])

    # Verify task_networks and transforms are preserved, not empty
    assert len(model.task_networks) > 0, "task_networks should not be empty"
    assert len(model.transforms) > 0, "transforms should not be empty"
    assert len(model.task_networks) == len(
        model.transforms
    ), "task_networks and transforms should align"
