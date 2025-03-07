import pytest
import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    RobertaPreLayerNormConfig,
    RobertaPreLayerNormForMaskedLM,
    RobertaPreLayerNormModel,
)

from electrolyte_fm.models.sae import (
    AbstractSAE,
    GatedSAE,
    InjectedCoder,
    SparsifiedModel,
    TiedBiasSAE,
    VanillaSAE,
    TopKSAE,
    topk,
)
from electrolyte_fm.utils.tokenizer import load_tokenizer


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_default_device()

SAE_CLASSES = [GatedSAE, TiedBiasSAE, VanillaSAE, TopKSAE]


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


def test_topk():
    x = torch.rand(8, 4, 20)
    assert ((x > 0).sum(-1) > 5).all()
    x_hat = topk(x, 5)
    assert ((x_hat > 0).sum(-1) == 5).all()


@pytest.mark.parametrize("sae_cls", SAE_CLASSES)
class TestSAE:
    B = 3
    H = 4
    T = 8
    E = 2

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

    @property
    def feature_shape(self):
        return (self.B, self.T, self.H * self.E)

    def input(self):
        return torch.rand(self.B, self.T, self.H)

    def test_init(self, sae_cls):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        for p in sae.parameters():
            assert p.isfinite().all()
            assert not p.isnan().any()

    def test_encode(self, sae_cls):
        sae = sae_cls(hidden_size=self.H, expansion=self.E)
        x = self.input()
        f = sae.encode(x)
        assert f.shape == self.feature_shape
        x_hat = sae.decode(f)
        assert x_hat.shape == x.shape
        assert sae.forward(x).equal(x_hat)

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


@pytest.mark.parametrize("sae_cls", SAE_CLASSES)
def test_injected_coder(sae_cls):
    hidden_size = 64
    model = torch.nn.Linear(hidden_size, hidden_size)
    sae = sae_cls(hidden_size=hidden_size, expansion=2)
    injected = InjectedCoder(model, sae).to(DEVICE)
    injected.eval()
    assert isinstance(injected, InjectedCoder)
    dense_model = injected.dense_model
    assert not injected.training and not dense_model.training
    x = torch.rand(2, 5, hidden_size, device=DEVICE)

    # Check dense
    y_ref = dense_model(x)
    injected.state = "dense"
    y_dense = injected(x)
    assert y_dense.shape == y_ref.shape
    assert y_dense.equal(y_ref)

    # Check null
    injected.state = "null"
    y_null = injected(x)
    assert y_null.shape == y_ref.shape
    assert y_null.equal(torch.zeros_like(y_dense))

    # Check sparse
    injected.state = "sparse"
    y_sparse = injected(x)
    assert y_sparse.shape == y_dense.shape


def test_instrumented():
    model = AutoModelForMaskedLM.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    hidden_size = model.config.hidden_size
    sae = TiedBiasSAE(hidden_size=hidden_size, expansion=2)
    sparse_model = SparsifiedModel.from_huggingface(model, sae, layer=0).to(DEVICE)
    model = model.to(DEVICE)
    sparse_model.eval()
    assert not model.training
    assert not sae.training
    assert not model.training

    batch = tokenizer(["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "C1=CC2=C(C=C1O)C(=CN2)CCN"])
    collate = DataCollatorWithPadding(tokenizer)
    batch = collate(batch)
    batch = {
        k: v.to(DEVICE)
        for k, v in batch.items()
        if k in ["input_ids", "attention_mask"]
    }
    batch["return_dict"] = True
    y = model(**batch).logits

    assert all([coder.state == "sparse" for coder in sparse_model.coders])
    with sparse_model.nullcoders() as sparse_model:
        assert all([coder.state == "null" for coder in sparse_model.coders])
        y_null = sparse_model(**batch).logits
        assert y_null.shape == y.shape
        assert y_null.device == y.device
        assert y_null.dtype == y.dtype

    with sparse_model.sparse(False) as sparse_model:
        assert all([coder.state == "dense" for coder in sparse_model.coders])
        y_dense = sparse_model(**batch).logits
        assert y_dense.shape == y.shape
        assert y_dense.device == y.device
        assert y_dense.dtype == y.dtype
        assert y_dense.equal(y)

    rc = sparse_model.loss_recovered(target=batch["input_ids"], **batch)
    assert isinstance(rc, torch.Tensor) and rc.shape == ()
    assert rc.isfinite() and not rc.isnan()
    assert rc <= 1


def test_sparse_model(roberta_tokenzier):
    roberta, tokenizer = roberta_tokenzier
    sae = TiedBiasSAE(hidden_size=roberta.config.hidden_size, expansion=2)
    sparse_model = SparsifiedModel.from_huggingface(roberta, sae, layer=2).to(DEVICE)
    robert = sparse_model.model.base_model
    assert isinstance(robert, RobertaPreLayerNormModel)
    assert isinstance(robert.encoder.layer[2].output.dense, InjectedCoder)
    assert robert.encoder.layer[2].output.dense is sparse_model.coders[0]

    collate = DataCollatorWithPadding(tokenizer)
    batch = collate([tokenizer("CNCCC")])
    batch = {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
        "return_dict": True,
    }
    y_sparse = sparse_model(**batch).logits
    robert.eval()
    y_dense = roberta(**batch).logits
    assert y_sparse.shape == y_dense.shape
    assert y_sparse.shape == (1, 5, len(tokenizer))

    # Check null features
    with sparse_model.nullcoders() as model:
        y_null = model(**batch).logits
        assert y_null.shape == y_sparse.shape

    # Check dense features
    with sparse_model.sparse(False) as model:
        # Run model in eval model to be deterministic
        model.eval()
        robert.eval()
        y_dense = roberta(**batch).logits
        y_dense_context = model(**batch).logits
        robert.train()
        model.train()
        assert y_dense_context.shape == y_dense.shape
        e = (y_dense_context.detach() - y_dense.detach()).abs()
        assert (e < 1e-6).all()

    # Check non-encoder parameters don't get gradients
    assert len(list(sparse_model.sparse_parameters())) > 0
    assert len(dict(sparse_model.sparse_named_parameters())) > 0

    # Check_gradient
    out, loss = sparse_model.forward_with_loss(**batch)
    # assert out.logits.equal(y_sparse)
    loss.backward()
    for _, v in sparse_model.sparse_named_parameters():
        assert v.grad is not None
        assert (v.grad != 0).any()

    # Check Recovered Loss
    rc = sparse_model.loss_recovered(
        **batch, target=batch["input_ids"], sparse_output=y_sparse
    )
    assert isinstance(rc, torch.Tensor) and rc.shape == ()
    assert rc.isfinite() and not rc.isnan()
