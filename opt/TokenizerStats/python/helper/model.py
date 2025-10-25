import os
import json
from pathlib import Path
import logging
import warnings
from rdkit import Chem
import torch
from smirk import SmirkTokenizerFast
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .tokenizer import load_tokenizer

warnings.filterwarnings("ignore", message=".*Unexpected or missing tensors.*")
logging.getLogger().setLevel(logging.ERROR)
AutoTokenizer.register("SmirkTokenizer", fast_tokenizer_class=SmirkTokenizerFast)


TOKENIZER_LABELS = {
    "meta-llama/Llama-3.2-1B": "Llama 3.2",
    "SmilesPE/SPE_ChEMBL": "SmilesPE",
    "3kdikco4": "ChemBERTa v1 (C)",
    "qdyxbwv3": "ChemBERTa v1",
    "llqb57c8": "SMI-TED (C)",
    "c1clszmm": "SMI-TED",
    "wv2dbdxf": "MoLFormer (C)",
    "20zu6xej": "MoLFormer",
    "p3xqpkrv": "SMILYAPE",
    "l7axquz1": "SMILYAPE (C)",
    "2scil3tk": "SmirkGPE (MB)",
    "cz8q161k": "Smirk (MB)",
    "ulxte55y": "SmirkGPE (C,NMB)",
    "c59ebnog": "SmirkGPE (NMB)",
    "ti624ev1": "Smirk",
    "u2vgi8dy": "Smirk (C)",
}


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles


def load_model(model, save_directory):
    if (file := Path(save_directory, "model.safetensors")).is_file():
        from safetensors.torch import load_model as st_load_model

        unexpected, missing = st_load_model(model, file, strict=False)
        if unexpected or missing:
            logging.warning(
                "Unexpected or missing tensors when loading %s, unexpected: %s missing: %s",
                file,
                unexpected,
                missing,
            )
    else:
        raise RuntimeError("No model found")


def resolve_tokenizer_name(tokenizer_name, encoder_path, tokenizer):
    if tokenizer_name in TOKENIZER_LABELS:
        return TOKENIZER_LABELS[tokenizer_name]

    for identifier, label in TOKENIZER_LABELS.items():
        if identifier in tokenizer_name or identifier in encoder_path:
            return label

    if os.path.isabs(tokenizer_name) or os.path.exists(tokenizer_name):
        return tokenizer.__class__.__name__

    raise ValueError(f"Unknown tokenizer: {tokenizer_name}")


class PackagedSmirkModel(torch.nn.Module):
    def __init__(self, encoder, encoder_path, tokenizer_name):
        super().__init__()
        self.encoder = encoder
        self.encoder_path = encoder_path
        self.tokenizer = load_tokenizer(tokenizer_name)
        self.tokenizer_name = self.resolve_tokenizer_name(tokenizer_name)

    def resolve_tokenizer_name(self, tokenizer_name):
        if tokenizer_name in TOKENIZER_LABELS:
            return TOKENIZER_LABELS[self._tokenizer_path]

        combined_path = tokenizer_name + self.encoder_path
        label = next(
            (
                label
                for identifier, label in TOKENIZER_LABELS.items()
                if identifier in combined_path
            ),
            None,
        )
        if label:
            return label

        raise ValueError(f"Unknown tokenizer: {self._tokenizer_path}")

    def attention_map(self, smi: list[str]):
        assert len(smi) == 1
        if "SMILYAPE" in self.tokenizer_name:
            encoding = self.tokenizer(smi[0])
            encoding["input_ids"] = torch.tensor(encoding["input_ids"]).unsqueeze(0)
            encoding["attention_mask"] = torch.tensor(
                encoding["attention_mask"]
            ).unsqueeze(0)
        else:
            encoding = self.tokenizer(smi)

        if "special_tokens_mask" in encoding:
            encoding.pop("special_tokens_mask")
        encoding = {
            k: torch.tensor(v).to(self.encoder.device) for k, v in encoding.items()
        }
        with torch.inference_mode():
            attentions = self.encoder(**encoding, output_attentions=True).attentions

        return attentions

    @classmethod
    def from_pretrained(cls, save_directory: str):
        config = json.loads(Path(save_directory, "config.json").read_text())
        try:
            encoder_config = AutoConfig.for_model(
                config["encoder"]["model_type"]
            ).from_dict(config["encoder"])
        except KeyError:
            encoder_config = AutoConfig.for_model("roberta-prelayernorm").from_dict(
                config
            )

        encoder = AutoModel.from_config(encoder_config, add_pooling_layer=False)
        if Path(f"{save_directory}/tokenizer_name").exists():
            tokenizer_name = (
                Path(f"{save_directory}/tokenizer_name").read_text().strip()
            )
        else:
            tokenizer_name = save_directory
        model = cls(encoder, save_directory, tokenizer_name)
        load_model(model, save_directory)
        return model


def get_attention_maps(model_path, smiles):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = PackagedSmirkModel.from_pretrained(model_path).eval().to(device)

    if "(C)" in model.tokenizer_name:
        smiles = canonicalize_smiles(smiles)

    tokens = model.tokenizer.convert_ids_to_tokens(
        model.tokenizer.encode(smiles, add_special_tokens=True)
    )

    if "APE" in model.tokenizer_name:
        tokens = [t for t in tokens if t not in ["<s>", "</s>"]]

    return {
        "tokenizer_name": model.tokenizer_name,
        "tokens": tokens,
        "attention_maps": model.attention_map([smiles]),
    }


if __name__ == "__main__":
    import glob

    for path in glob.glob(
        "/nfs/turbo/coe-venkvis/abhutani/electrolyte-fm/opt/TokenizerStats/smirk-models/**/pretrained"
    ):
        if "b61irf10" in path or "x8jqdruh" in path:
            continue
        attention_maps = get_attention_maps(path, "Cl[Pt@SP1](Cl)([NH3])[NH3]")
