from typing import Optional, Union

from selfies import encoder as sf_encoder
from torch import tensor
from torch.nn import Module
from transformers import PreTrainedTokenizerFast

from electrolyte_fm.models import LMFinetuning
from electrolyte_fm.interpretibility.attention_maps import (
    maybe_load_model_and_tokenizer,
)


def get_embedding(
    checkpoint: Union[str, Module],
    smiles: str,
    tokenizer: Optional[Union[str, PreTrainedTokenizerFast]] = None,
):
    model, tok, tokenizer_name = maybe_load_model_and_tokenizer(checkpoint, tokenizer)

    if isinstance(model, LMFinetuning):
        encoder = model.encoder
    else:
        encoder = model.model

    # Get tokens and attention map
    seq = (
        sf_encoder(smiles)
        if (tokenizer_name and "selfies" in tokenizer_name)
        else smiles
    )
    encoding = tok(
        [
            seq,
        ]
    )
    if "special_tokens_mask" in encoding:
        encoding.pop("special_tokens_mask")
    encoding = {k: tensor(v).to(model.device) for k, v in encoding.items()}

    # `hidden_states` is a tuple with length = number of hidden layers
    # each element has shape [batch_size, sequence_length, hidden_size]
    embedding = encoder(**encoding, output_hidden_states=True).hidden_states[-1][
        :, 0, :
    ]
    return embedding
