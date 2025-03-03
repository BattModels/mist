from datasets import features
from numpy import zeros_like
import pytest
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaPreLayerNormConfig,
    RobertaPreLayerNormForMaskedLM,
    RobertaPreLayerNormModel,
)

# from electrolyte_fm.data_modules.sae_dataset import (
#     HiddenStateDataModule,
#     extract_hidden_state,
# )
from electrolyte_fm.models.model_utils import load_encoder
from electrolyte_fm.models.sae import (
    AbstractSAE,
    GatedSAE,
    InjectedCoder,
    SparsifiedModel,
    TiedBiasSAE,
    avg_l0_norm,
)
from electrolyte_fm.utils.tokenizer import load_tokenizer


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


DEVICE = get_default_device()


@pytest.fixture()
@pytest.mark.cuda
def roberta_tokenzier():
    tokenizer = load_tokenizer("smirk")
    config = RobertaPreLayerNormConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
    )
    return RobertaPreLayerNormForMaskedLM(config).to(DEVICE), tokenizer


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

    def test_forward(self, sae_cls: AbstractSAE):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        assert isinstance(sae, AbstractSAE)
        x = self.input()
        features = sae.forward(x)
        assert features.shape == x.shape
        assert features.isfinite().all()
        assert not features.isnan().any()

    def test_loss(self, sae_cls):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        assert isinstance(sae, AbstractSAE)
        x = self.input()
        y, loss = sae.forward_with_loss(x)
        assert loss.isfinite() and loss.shape == ()
        assert y.shape == x.shape
        assert y.isfinite().all()


# def test_avg_l0_norm():
#     x = torch.tensor([[1, 0, 0], [0, 5, 0]])
#     assert avg_l0_norm(x) == 1
#     assert avg_l0_norm(x.T).isclose(torch.tensor(1 / 3))


# def test_dataloader():
#     ckpt_path = "ibm/MoLFormer-XL-both-10pct"
#     path = "/lustre/fs0/awadell/realspace"
#     dm = HiddenStateDataModule(ckpt_path, path)
#     dm.prepare_data()
#     dm.setup("fit")
#     for batch in dm.train_dataloader():
#         assert isinstance(batch, torch.Tensor)
#         assert batch.shape == (dm.batch_size, 768)
#         assert not batch.requires_grad
#         break


def test_injected_coder(roberta_tokenzier):
    roberta, _ = roberta_tokenzier
    hidden_size = roberta.config.hidden_size
    sae = TiedBiasSAE(hidden_size=hidden_size, expansion=2)
    injected = InjectedCoder(roberta.base_model.encoder.layer[2], sae)
    injected.eval()
    assert isinstance(injected, InjectedCoder)
    dense_model = injected.dense_model
    assert not injected.training and not dense_model.training
    x = torch.rand(1, 5, hidden_size, device=DEVICE)

    # Check dense
    y_ref = dense_model(x)[0]
    injected.state = "dense"
    y_dense = injected(x)[0]
    print(y_ref, y_dense)
    assert y_dense.equal(y_ref)

    # Check null
    injected.state = "null"
    y_null = injected(x)[0]
    assert y_null.equal(x)

    # Check sparse
    injected.state = "sparse"
    y_sparse = injected(x)[0]
    assert y_sparse.shape == y_dense.shape


def test_instrumented(roberta_tokenzier):
    model = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    model, tokenizer = roberta_tokenzier
    hidden_size = model.config.hidden_size
    sae = TiedBiasSAE(hidden_size=hidden_size, expansion=2)
    sparse_model = SparsifiedModel.from_huggingface(model, sae, layer=2).to(DEVICE)
    model = model.to(DEVICE)
    sparse_model.eval()
    model.eval()

    batch = tokenizer("CNCCC")
    input_ids = torch.tensor(batch["input_ids"]).to(DEVICE)
    y = model(input_ids)[0]
    assert y.equal(model(input_ids)[0])
    print(y)

    with sparse_model.nullcoders() as sparse_model:
        y_null = sparse_model(input_ids)[0]
        print(y_null)

    with sparse_model.sparse(False) as sparse_model:
        y_dense = sparse_model(input_ids)[0]
        print(y_dense)
        assert y_dense.equal(y)

    assert False


def test_sparse_model(roberta_tokenzier):
    roberta, tokenizer = roberta_tokenzier
    sae = GatedSAE(hidden_size=roberta.config.hidden_size, expansion=2)
    sparse_model = SparsifiedModel.from_huggingface(roberta, sae, layer=2).to(DEVICE)
    robert = sparse_model.model.base_model
    assert isinstance(robert, RobertaPreLayerNormModel)
    assert isinstance(robert.encoder.layer[2], InjectedCoder)
    assert robert.encoder.layer[2] is sparse_model.coders[0]

    collate = DataCollatorWithPadding(tokenizer)
    batch = collate([tokenizer("CNCCC")])
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    y_sparse = sparse_model.forward(input_ids, attention_mask)
    robert.eval()
    y_dense = roberta(input_ids, attention_mask)[0]
    assert y_sparse.shape == y_dense.shape
    assert y_sparse.shape == (1, 5, len(tokenizer))

    # Check null features
    with sparse_model.nullcoders() as model:
        y_null = model.forward(input_ids, attention_mask)
        assert y_null.shape == y_sparse.shape

    # Check dense features
    with sparse_model.sparse(False) as model:
        # Run model in eval model to be deterministic
        model.eval()
        robert.eval()
        y_dense = roberta(input_ids, attention_mask)[0]
        y_dense_context = model.forward(input_ids, attention_mask)
        robert.train()
        model.train()
        assert y_dense_context.shape == y_dense.shape
        e = (y_dense_context.detach() - y_dense.detach()).abs()
        assert (e < 1e-6).all()

    # Check non-encoder parameters don't get gradients
    assert len(list(sparse_model.sparse_parameters())) > 0
    assert len(dict(sparse_model.sparse_named_parameters())) > 0

    # Check_gradient
    y, loss = sparse_model.forward_with_loss(input_ids, attention_mask)
    loss.backward()
    for k, v in sparse_model.sparse_named_parameters():
        assert v.grad is not None
        assert (v.grad != 0).any()

    # Check Recovered Loss
    rc = sparse_model.loss_recovered(
        input_ids, input_ids, attention_mask, sparse_output=y_sparse
    )
    print(rc)
    assert isinstance(rc, torch.Tensor) and rc.shape == ()
    assert rc.isfinite() and not rc.isnan()
    assert False


# def test_extract_hidden_state():
#     ckpt_path = "ibm/MoLFormer-XL-both-10pct"
#     encoder = load_encoder(ckpt_path)
#     d_model = 768
#     tok = load_tokenizer(ckpt_path)
#     smiles = [
#         "CCC(=O)OC1(C(CC2C1(CC(C3(C2CC(C4=CC(=O)C=CC43C)F)F)O)C)C)C(=O)SCF",
#         "CNCCC(c1ccccc1)Oc2ccc(cc2)C(F)(F)F",
#     ]
#     tokens = [tok(smi) for smi in smiles]
#     input_ids = [x["input_ids"] for x in tokens]
#     attention_mask = [x["attention_mask"] for x in tokens]
#     batch = extract_hidden_state(
#         input_ids,
#         attention_mask,
#         encoder=encoder,
#         collate=DataCollatorWithPadding(tok),
#         layer=0.5,
#     )
#     assert "hidden_state" in batch
#     assert batch["hidden_state"].shape == (len(smiles), d_model)


# def test_wrapped_sae():
#     encoder = load_encoder("ibm/MoLFormer-XL-both-10pct")
#     coder = GatedSAE(hidden_size=768, expansion=4)
#     model = WrappedSAE(encoder, coder, 2)
