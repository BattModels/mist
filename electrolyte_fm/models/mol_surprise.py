import torch
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding
from ..utils.tokenizer import load_tokenizer


class MolSurpriseFM(torch.nn.Module):
    def __init__(self, encoder, tokenizer):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.collate_fn = DataCollatorWithPadding(self.tokenizer)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        special_tokens_mask: torch.Tensor,
        per_token: bool = False,
    ):
        logits = self.encoder(input_ids, attention_mask).logits
        B = input_ids.shape[0]
        V = logits.shape[-1]

        labels = input_ids.detach().masked_fill(special_tokens_mask.bool(), -100)

        score = (
            F.cross_entropy(logits.view(-1, V), labels.view(-1), reduction="none")
            .reshape(B, -1)
            .sum(-1)
        )
        if per_token:
            score = score / attention_mask.sum(-1)
        return score

    def score(self, smiles: list[str], per_token: bool = False) -> list[float]:
        batch = self.tokenizer(smiles, return_special_tokens_mask=True)
        batch = self.collate_fn(batch).to(self.encoder.device)
        with torch.inference_mode():
            return self.forward(
                batch["input_ids"],
                batch["attention_mask"],
                batch["special_tokens_mask"],
                per_token,
            ).to("cpu")

    def embed(self, smiles: list[str]) -> torch.Tensor:
        batch = self.tokenizer(smiles, return_special_tokens_mask=True)
        batch = self.collate_fn(batch).to(self.encoder.device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if hasattr(self.encoder, "roberta_prelayernorm"):
            encoder = self.encoder.roberta_prelayernorm
        else:
            encoder = self.encoder.encoder
        with torch.inference_mode():
            return encoder(input_ids, attention_mask=attention_mask).last_hidden_state

    @classmethod
    def from_checkpoint(cls, ckpt: str, **kwargs):
        from ..utils.ckpt import DeepSpeedMixin

        encoder = DeepSpeedMixin.load(ckpt).model
        tokenizer = load_tokenizer(ckpt)
        return cls(encoder, tokenizer, **kwargs)

    @classmethod
    def from_pretrained(cls, name_or_path: str, dtype=None, **kwargs):
        encoder = AutoModelForMaskedLM.from_pretrained(
            name_or_path,
            trust_remote_code=True,
            # device_map="auto",
            # torch_dtype="auto",
        )
        tokenizer = load_tokenizer(name_or_path)
        return cls(encoder, tokenizer, **kwargs)

    @classmethod
    def from_untrained(cls, name_or_path: str, dtype=None, **kwargs):
        config = AutoConfig.from_pretrained(name_or_path, trust_remote_code=True)
        encoder = AutoModelForMaskedLM.from_config(config)
        tokenizer = load_tokenizer(name_or_path)
        return cls(encoder, tokenizer, **kwargs)
