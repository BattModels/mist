from types import SimpleNamespace

import pytest
import torch

from electrolyte_fm.models.roberta_base import RoBERTa
from electrolyte_fm.models.roberta_prelayernorm import RoBERTaPreLayerNorm
from electrolyte_fm.models.lm_finetuning import load_encoder
from electrolyte_fm.models.model_utils import create_position_ids


class FakeTokenizer(SimpleNamespace):
    def __len__(self):
        return self.length


@pytest.mark.parametrize("model_cls", [RoBERTa, RoBERTaPreLayerNorm])
def test_roberta_config_uses_tokenizer_token_ids(monkeypatch, model_cls):
    tokenizer = FakeTokenizer(
        length=101,
        vocab_size=101,
        pad_token_id=7,
        bos_token_id=0,
        eos_token_id=9,
    )
    monkeypatch.setattr(
        "electrolyte_fm.models.roberta_base.load_tokenizer",
        lambda name: tokenizer,
    )

    model = model_cls(vocab_size=13, tokenizer="test-tokenizer")

    assert model.config.vocab_size == tokenizer.vocab_size
    assert model.config.pad_token_id == tokenizer.pad_token_id
    assert model.config.bos_token_id == tokenizer.bos_token_id
    assert model.config.eos_token_id == tokenizer.eos_token_id


def test_load_encoder_applies_new_tokenizer(monkeypatch):
    model = RoBERTa(
        vocab_size=13,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_token_counter=False,
    )
    encoder = model.get_encoder()
    tokenizer = FakeTokenizer(
        length=17,
        vocab_size=17,
        pad_token_id=6,
        bos_token_id=0,
        eos_token_id=8,
    )
    monkeypatch.setattr(
        "electrolyte_fm.models.lm_finetuning.load_tokenizer",
        lambda name: tokenizer,
    )

    loaded = load_encoder(encoder, tokenizer="new-tokenizer")

    assert loaded is encoder
    assert loaded.config.vocab_size == tokenizer.vocab_size
    assert loaded.config.pad_token_id == tokenizer.pad_token_id
    assert loaded.config.bos_token_id == tokenizer.bos_token_id
    assert loaded.config.eos_token_id == tokenizer.eos_token_id
    assert loaded.get_input_embeddings().num_embeddings == tokenizer.vocab_size


def test_load_encoder_uses_new_pad_id_for_position_embeddings(monkeypatch):
    model = RoBERTa(
        vocab_size=13,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_token_counter=False,
    )
    encoder = model.get_encoder()
    tokenizer = FakeTokenizer(
        length=17,
        vocab_size=17,
        pad_token_id=6,
        bos_token_id=0,
        eos_token_id=8,
    )
    monkeypatch.setattr(
        "electrolyte_fm.models.lm_finetuning.load_tokenizer",
        lambda name: tokenizer,
    )
    loaded = load_encoder(encoder, tokenizer="new-tokenizer")
    captured_position_ids = []
    hook = loaded.embeddings.position_embeddings.register_forward_pre_hook(
        lambda module, args: captured_position_ids.append(args[0].detach().clone())
    )

    input_ids = torch.tensor(
        [[tokenizer.bos_token_id, 5, tokenizer.eos_token_id, 6, 6]]
    )
    position_ids = create_position_ids(
        input_ids,
        loaded.config.pad_token_id,
        loaded.config.position_padding_idx,
    )
    loaded.embeddings(input_ids=input_ids, position_ids=position_ids)
    hook.remove()

    assert loaded.config.pad_token_id == tokenizer.pad_token_id
    assert loaded.config.position_padding_idx == 1
    assert loaded.embeddings.padding_idx == 1
    assert loaded.embeddings.word_embeddings.padding_idx == tokenizer.pad_token_id
    assert loaded.embeddings.position_embeddings.padding_idx == 1
    assert torch.equal(captured_position_ids[0], torch.tensor([[2, 3, 4, 1, 1]]))


@pytest.mark.parametrize("model_cls", [RoBERTa, RoBERTaPreLayerNorm])
def test_load_encoder_without_new_tokenizer_preserves_trained_model(model_cls):
    model = model_cls(
        vocab_size=13,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_token_counter=False,
    )
    encoder = model.get_encoder()
    embeddings = encoder.get_input_embeddings()
    weights = embeddings.weight.detach().clone()
    token_config = {
        name: getattr(encoder.config, name)
        for name in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id")
    }

    loaded = load_encoder(encoder)

    assert loaded is encoder
    assert loaded.get_input_embeddings() is embeddings
    assert torch.equal(loaded.get_input_embeddings().weight, weights)
    assert {name: getattr(loaded.config, name) for name in token_config} == token_config


def test_load_encoder_can_skip_tokenizer_configuration(monkeypatch):
    model = RoBERTa(
        vocab_size=13,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        enable_token_counter=False,
    )
    encoder = model.get_encoder()
    embeddings = encoder.get_input_embeddings()
    token_config = {
        name: getattr(encoder.config, name)
        for name in ("vocab_size", "pad_token_id", "bos_token_id", "eos_token_id")
    }
    monkeypatch.setattr(
        "electrolyte_fm.models.lm_finetuning.load_tokenizer",
        lambda name: pytest.fail("tokenizer configuration should be skipped"),
    )

    loaded = load_encoder(
        encoder,
        tokenizer="new-tokenizer",
        configure_tokenizer=False,
    )

    assert loaded is encoder
    assert loaded.get_input_embeddings() is embeddings
    assert {name: getattr(loaded.config, name) for name in token_config} == token_config
