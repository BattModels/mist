"""Small, offline regression tests for the production finetuning interfaces."""

import ast
import importlib.util
from pathlib import Path

import lightning
import pytest
import torch
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
    RobertaPreLayerNormConfig,
    RobertaPreLayerNormModel,
)

from electrolyte_fm.models import prod_finetune as production
from electrolyte_fm.models.lm_finetuning import LMFinetuning
from electrolyte_fm.models.normalize import Standardize
from electrolyte_fm.models.prediction_task_head import PredictionTaskHead

ARCHITECTURES = [
    (RobertaConfig, RobertaModel),
    (RobertaPreLayerNormConfig, RobertaPreLayerNormModel),
]
SMILES = ["CCO", "O", "OCC", "NCC(O)O"]
CHANNELS = [{"name": "homo", "unit": "eV"}, {"name": "gap", "unit": "eV"}]


@pytest.fixture(autouse=True)
def offline_and_single_thread(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tokenizer():
    vocabulary = {
        "<s>": 0,
        "unused": 1,
        "</s>": 2,
        "<unk>": 3,
        "C": 4,
        "<pad>": 5,
        "O": 6,
        "N": 7,
        "<mask>": 8,
        "(": 9,
        ")": 10,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Split(Regex("."), behavior="isolated")
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    result = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        model_max_length=32,
    )
    result.model_input_names = ["input_ids", "attention_mask"]
    return result


def make_encoder(architecture, tokenizer, updated, pooler=False):
    config_class, model_class = architecture
    config = config_class(
        vocab_size=len(tokenizer),
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=32,
        pad_token_id=tokenizer.pad_token_id if updated else 1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )
    if updated:
        config.position_padding_idx = 1
    encoder = model_class(config, add_pooling_layer=pooler).eval()
    if updated:
        encoder.embeddings.padding_idx = 1
        encoder.embeddings.position_embeddings.padding_idx = 1
    return encoder


def make_components(architecture, tokenizer, updated, multitask):
    torch.manual_seed(73)
    encoder = make_encoder(architecture, tokenizer, updated)
    widths = [1, 1] if multitask else [2]
    heads = [PredictionTaskHead(8, width, dropout=0).eval() for width in widths]
    transforms = [Standardize(width).eval() for width in widths]
    for index, transform in enumerate(transforms):
        transform.mean.copy_(torch.arange(transform.num_outputs) + 0.7 + index)
        transform.std.copy_(torch.arange(transform.num_outputs) + 0.3 + index)
    return encoder, heads, transforms


def wrap_components(encoder, heads, transforms, tokenizer, multitask, **kwargs):
    if multitask:
        return production.MISTMultiTask.from_components(
            encoder, heads, transforms, tokenizer, CHANNELS, **kwargs
        )
    return production.MISTFinetuned.from_components(
        encoder, heads[0], transforms[0], tokenizer, CHANNELS, **kwargs
    )


def expected_positions(encoder, input_ids):
    token_pad = encoder.config.pad_token_id
    offset = getattr(encoder.config, "position_padding_idx", token_pad)
    mask = input_ids.ne(token_pad).long()
    return mask.cumsum(-1) * mask + offset


def expected_output(encoder, heads, transforms, batch):
    kwargs = {}
    if hasattr(encoder.config, "position_padding_idx"):
        kwargs["position_ids"] = expected_positions(encoder, batch["input_ids"])
    with torch.inference_mode():
        hidden = encoder(**batch, **kwargs).last_hidden_state
        prediction = torch.cat(
            [transform(head(hidden)) for head, transform in zip(heads, transforms)],
            dim=-1,
        )
    return prediction, hidden[:, 0]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("updated", [False, True], ids=["legacy", "new-token-ids"])
@pytest.mark.parametrize("multitask", [False, True], ids=["single", "multitask"])
def test_forward_embed_predict_preserve_trained_positions(
    tokenizer, architecture, updated, multitask
):
    encoder, heads, transforms = make_components(
        architecture, tokenizer, updated, multitask
    )
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask).eval()
    batch = tokenizer(SMILES, padding=True, return_tensors="pt")
    expected, embedding = expected_output(encoder, heads, transforms, batch)
    captured = []
    hook = model.encoder.embeddings.position_embeddings.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    try:
        with torch.inference_mode():
            actual = model(**batch)
        assert torch.equal(captured.pop(), expected_positions(encoder, batch.input_ids))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(model.embed(SMILES, batch_size=2), embedding)
        named = model.predict(SMILES, batch_size=2)
        for index, channel in enumerate(CHANNELS):
            values = named[channel["name"]]["value"]
            assert values.device.type == "cpu"
            assert named[channel["name"]]["unit"] == "eV"
            torch.testing.assert_close(values, expected[:, index])
        for positions in captured:
            assert positions.min() >= 1
            assert positions.max() < model.encoder.config.max_position_embeddings
    finally:
        hook.remove()
    assert model.encoder.config.pad_token_id == encoder.config.pad_token_id
    assert model.encoder.embeddings.position_embeddings.padding_idx == 1
    if updated:
        assert model.encoder.config.position_padding_idx == 1
        assert (
            model.encoder.embeddings.word_embeddings.padding_idx
            == tokenizer.pad_token_id
        )
    else:
        assert not hasattr(model.encoder.config, "position_padding_idx")


@pytest.mark.parametrize("multitask", [False, True])
def test_forward_honors_explicit_position_ids(tokenizer, multitask):
    encoder, heads, transforms = make_components(
        ARCHITECTURES[0], tokenizer, True, multitask
    )
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask).eval()
    batch = tokenizer(SMILES, padding=True, return_tensors="pt")
    explicit = torch.full_like(batch.input_ids, 3)
    observed = []
    hook = model.encoder.embeddings.position_embeddings.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach().clone())
    )
    try:
        model(**batch, position_ids=explicit)
    finally:
        hook.remove()
    assert torch.equal(observed[0], explicit)


@pytest.mark.parametrize("multitask", [False, True])
def test_from_components_can_reuse_without_reinitializing(
    tokenizer, monkeypatch, multitask
):
    encoder, heads, transforms = make_components(
        ARCHITECTURES[0], tokenizer, True, multitask
    )
    monkeypatch.setattr(
        production,
        "build_encoder_from_dict",
        lambda *_: pytest.fail("copy=False must not allocate another encoder"),
    )
    model = wrap_components(
        encoder, heads, transforms, tokenizer, multitask, copy=False
    ).eval()
    assert model.encoder is encoder
    if multitask:
        assert all(a is b for a, b in zip(model.task_networks, heads))
        assert all(a is b for a, b in zip(model.transforms, transforms))
    else:
        assert model.task_network is heads[0]
        assert model.transform is transforms[0]
    assert model.tokenizer is tokenizer


@pytest.mark.parametrize("multitask", [False, True])
@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("pooler", [False, True])
def test_from_components_copies_by_default(tokenizer, multitask, architecture, pooler):
    encoder, heads, transforms = make_components(
        architecture, tokenizer, True, multitask
    )
    encoder = make_encoder(architecture, tokenizer, True, pooler=pooler)
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask)
    assert model.encoder is not encoder
    assert model.encoder.get_input_embeddings().weight.data_ptr() != (
        encoder.get_input_embeddings().weight.data_ptr()
    )
    actual_heads = model.task_networks if multitask else [model.task_network]
    actual_transforms = model.transforms if multitask else [model.transform]
    assert all(a is not b for a, b in zip(actual_heads, heads))
    assert all(a is not b for a, b in zip(actual_transforms, transforms))


@pytest.mark.parametrize("multitask", [False, True])
def test_prediction_api_empty_batched_override_and_bfloat16(
    tokenizer, monkeypatch, multitask
):
    encoder, heads, transforms = make_components(
        ARCHITECTURES[0], tokenizer, True, multitask
    )
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask).eval()
    # The explicit tokenizer must be used, including the historical positional
    # second tokenizer argument on the multitask interface.
    model.tokenizer = None
    seen = []
    original_call = type(tokenizer).__call__

    def record_smiles(tokenizer_self, smiles, *args, **kwargs):
        if tokenizer_self is tokenizer:
            seen.extend(smiles)
        return original_call(tokenizer_self, smiles, *args, **kwargs)

    monkeypatch.setattr(type(tokenizer), "__call__", record_smiles)
    if multitask:
        named = model.predict(SMILES, tokenizer, batch_size=2)
    else:
        named = model.predict(SMILES, tokenizer=tokenizer, batch_size=2)
    # Canonicalizing CCO/OCC here would destroy the permutation experiment.
    assert seen == SMILES
    raw = model.predict(SMILES, tokenizer=tokenizer, return_dict=False, batch_size=1)
    assert raw.shape == (len(SMILES), 2)
    for index, channel in enumerate(CHANNELS):
        torch.testing.assert_close(named[channel["name"]]["value"], raw[:, index])
    assert model.embed([], tokenizer=tokenizer, batch_size=2).shape == (0, 8)
    empty = model.predict([], tokenizer=tokenizer, batch_size=2)
    assert set(empty) == {channel["name"] for channel in CHANNELS}
    assert all(len(record["value"]) == 0 for record in empty.values())
    model.to(torch.bfloat16)
    output = model.predict(SMILES[:2], tokenizer=tokenizer, return_dict=False)
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"
    assert output.numpy().shape == (2, 2)


@pytest.mark.parametrize("multitask", [False, True])
def test_string_channels(tokenizer, multitask):
    encoder, heads, transforms = make_components(
        ARCHITECTURES[0], tokenizer, True, multitask
    )
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask).eval()
    model.channels = ["homo", "gap"]
    assert set(model.predict(SMILES[:1])) == {"homo", "gap"}


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("updated", [False, True], ids=["legacy", "new-token-ids"])
@pytest.mark.parametrize("multitask", [False, True], ids=["single", "multitask"])
def test_offline_hf_round_trip_uses_local_production_class(
    tmp_path, tokenizer, architecture, updated, multitask
):
    encoder, heads, transforms = make_components(
        architecture, tokenizer, updated, multitask
    )
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask).eval()
    expected = model.predict(SMILES, return_dict=False)
    # Releases have auto_map entries: loading them must not execute an old
    # embedded production implementation instead of the locally fixed class.
    model.config.auto_map = {
        "AutoConfig": "not_present.UnusedConfig",
        "AutoModel": "not_present.UnusedModel",
    }
    snapshot = tmp_path / "release"
    model.save_pretrained(snapshot)
    restored = production.load_finetuned(snapshot, local_files_only=True).eval()
    expected_class = production.MISTMultiTask if multitask else production.MISTFinetuned
    assert type(restored) is expected_class
    torch.testing.assert_close(restored.predict(SMILES, return_dict=False), expected)
    torch.testing.assert_close(restored.embed(SMILES), model.embed(SMILES))
    assert restored.encoder.config.pad_token_id == encoder.config.pad_token_id
    assert restored.encoder.embeddings.position_embeddings.padding_idx == 1
    assert restored.tokenizer.get_vocab() == tokenizer.get_vocab()


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_lightning_checkpoint_conversion_preserves_predictions(
    tmp_path, tokenizer, architecture
):
    snapshot = tmp_path / "base"
    encoder = make_encoder(architecture, tokenizer, updated=False)
    encoder.save_pretrained(snapshot)
    tokenizer.save_pretrained(snapshot)
    trained = LMFinetuning(
        output_size=2,
        encoder_ckpt=str(snapshot),
        tokenizer=str(snapshot),
        task="regression",
        target_columns=["homo", "gap"],
        metrics=["mae"],
        transform="standardize",
        track_oov=False,
    )
    # The fix-pos-ids branch's earlier constructor records this only in hparams.
    if hasattr(trained, "tokenizer"):
        del trained.tokenizer
    trained.configure_model()
    trained.transform.mean.copy_(torch.tensor([-0.2, 0.3]))
    trained.transform.std.copy_(torch.tensor([0.05, 0.1]))
    trained.eval()
    checkpoint = tmp_path / "best.ckpt"
    torch.save(
        {
            "state_dict": trained.state_dict(),
            "hyper_parameters": dict(trained.hparams),
            "pytorch-lightning_version": lightning.__version__,
        },
        checkpoint,
    )
    batch = tokenizer(SMILES, padding=True, return_tensors="pt")
    with torch.inference_mode():
        expected = trained(batch)
    restored = production.MISTFinetuned.from_lightning_checkpoint(
        checkpoint, tokenizer=str(snapshot), local_files_only=True
    ).eval()
    torch.testing.assert_close(restored.predict(SMILES, return_dict=False), expected)
    dispatched = production.load_finetuned(checkpoint, local_files_only=True).eval()
    torch.testing.assert_close(dispatched.predict(SMILES, return_dict=False), expected)
    assert restored.encoder.config.pad_token_id == tokenizer.pad_token_id
    assert restored.encoder.config.position_padding_idx == 1
    assert set(restored.predict(SMILES)) == {"homo", "gap"}


@pytest.mark.parametrize("multitask", [False, True])
def test_packaged_production_code_keeps_position_helper_standalone(
    tmp_path, tokenizer, multitask
):
    from opt.package.write_model_class import write_modeling_module

    encoder, heads, transforms = make_components(
        ARCHITECTURES[0], tokenizer, True, multitask
    )
    model = wrap_components(encoder, heads, transforms, tokenizer, multitask).eval()
    root = Path(__file__).resolve().parents[1]
    generated = write_modeling_module(
        tmp_path,
        root / "opt/package/modeling_mist.py.j2",
        config_class=type(model.config),
        model_class=type(model),
        dep_classes=[PredictionTaskHead, Standardize],
        module_basename="modeling_test_finetuned",
    )
    source = generated.read_text()
    top_level_imports = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    for node in top_level_imports:
        names = (
            [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else [a.name for a in node.names]
        )
        assert not any(
            name.startswith(("lightning", "deepspeed", "electrolyte_fm"))
            for name in names
        )
        assert not isinstance(node, ast.ImportFrom) or node.level == 0
    assert "def create_position_ids(" in source
    spec = importlib.util.spec_from_file_location("modeling_test_finetuned", generated)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if multitask:
        packaged = module.MISTMultiTask.from_components(
            encoder, heads, transforms, tokenizer, CHANNELS
        ).eval()
    else:
        packaged = module.MISTFinetuned.from_components(
            encoder, heads[0], transforms[0], tokenizer, CHANNELS
        ).eval()
    torch.testing.assert_close(
        packaged.predict(SMILES, return_dict=False),
        model.predict(SMILES, return_dict=False),
    )
