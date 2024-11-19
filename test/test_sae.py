import pytest
import torch
from transformers import DataCollatorWithPadding

from electrolyte_fm.data_modules.sae_dataset import (
    HiddenStateDataModule,
    extract_hidden_state,
)
from electrolyte_fm.models.model_utils import load_encoder
from electrolyte_fm.models.sae import GatedSAE, TiedBiasSAE, avg_l0_norm
from electrolyte_fm.utils.tokenizer import load_tokenizer


@pytest.mark.parametrize("sae_cls", [GatedSAE, TiedBiasSAE])
class TestSAE:
    B = 3
    H = 4
    E = 2

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

    @property
    def feature_shape(self):
        return (self.B, self.H * self.E)

    def input(self):
        return torch.rand(self.B, self.H)

    def test_init(self, sae_cls):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        for p in sae.parameters():
            assert p.isfinite().all()
            assert not p.isnan().any()

    def test_forward(self, sae_cls):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        features = sae(self.input())
        assert features.shape == self.feature_shape
        assert features.isfinite().all()
        assert not features.isnan().any()

    def test_loss(self, sae_cls):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        x = self.input()
        out = sae.loss(x)
        for v in out.values():
            assert v.isfinite().all()
            assert not v.isnan().any()

        # Check output
        assert out["loss"].shape == ()
        assert out["features"].shape == self.feature_shape
        assert out["features"].isclose(sae(x)).all()
        # assert out["x_hat"].shape == x.shape
        # assert out["x_hat"].isclose(sae.reconstruct(sae(x))).all()


def test_avg_l0_norm():
    x = torch.tensor([[1, 0, 0], [0, 5, 0]])
    assert avg_l0_norm(x) == 1
    assert avg_l0_norm(x.T).isclose(torch.tensor(1 / 3))


def test_dataloader():
    ckpt_path = "ibm/MoLFormer-XL-both-10pct"
    path = "/lustre/fs0/awadell/realspace"
    dm = HiddenStateDataModule(ckpt_path, path)
    dm.prepare_data()
    dm.setup("fit")
    for batch in dm.train_dataloader():
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (dm.batch_size, 768)
        assert not batch.requires_grad
        break


def test_extract_hidden_state():
    ckpt_path = "ibm/MoLFormer-XL-both-10pct"
    encoder = load_encoder(ckpt_path)
    d_model = 768
    tok = load_tokenizer(ckpt_path)
    smiles = [
        "CCC(=O)OC1(C(CC2C1(CC(C3(C2CC(C4=CC(=O)C=CC43C)F)F)O)C)C)C(=O)SCF",
        "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F",
    ]
    tokens = [tok(smi) for smi in smiles]
    input_ids = [x["input_ids"] for x in tokens]
    attention_mask = [x["attention_mask"] for x in tokens]
    batch = extract_hidden_state(
        input_ids,
        attention_mask,
        encoder=encoder,
        collate=DataCollatorWithPadding(tok),
        layer=0.5,
    )
    assert "hidden_state" in batch
    assert batch["hidden_state"].shape == (len(smiles), d_model)
