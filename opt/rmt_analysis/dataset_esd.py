import numpy as np
import numpy.typing as npt

from electrolyte_fm.data_modules.utils import MolEncoding
from electrolyte_fm.data_modules import RobertaDataSet, PubChemQC
from electrolyte_fm.interpretibility.attention_maps import (
    maybe_load_model_and_tokenizer,
)


def dataset_embeddings(
    checkpoint: str, data_path: str, encoding: str = MolEncoding.KEKULE
):
    """
    Collate model embeddings for all molecules in dataset.

    Returns
    _______
    all_embds: np.NDArray
        Array of size [].
    """
    # Load MIST checkpoint
    tokenizer = None
    checkpoint, tokenizer, tokenizer_name = maybe_load_model_and_tokenizer(
        checkpoint, tokenizer
    )
    encoder = checkpoint.model

    # Load dataset
    if "pubchemqc" in data_path:
        ds = PubChemQC(path=data_path, encoding=encoding, mlm_probability=0.0)
    else:
        ds = RobertaDataSet(path=data_path, encoding=encoding, mlm_probability=0.0)
    ds.setup(stage="train")
    encoder.to("cpu")
    train_dl = ds.train_dataloader()

    # Extract embeddings
    all_embds = []
    for idx, batch in enumerate(train_dl):
        if "special_tokens_mask" in batch:
            batch.pop("special_tokens_mask")
        embds = encoder(**batch, output_hidden_states=True).hidden_states[-1][:, 0, :]
        all_embds.append(embds.detach().float().numpy())

    all_embds = np.vstack(all_embds)
    return all_embds


def emperical_spectral_density(embeds: npt.NDArray):
    return
