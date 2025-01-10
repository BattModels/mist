import torch
from sklearn.preprocessing import PowerTransformer as _PowerTransformer


def get_normalizer(transform: str, num_outputs: int) -> torch.nn.Module:
    if transform in ["standardize", Standardize.__name__]:
        return Standardize(num_outputs)
    elif transform in ["power_transform", PowerTransform.__name__]:
        return PowerTransform(num_outputs)
    else:
        return IdentityTransform()


class Standardize(torch.nn.Module):
    def __init__(self, num_outputs: int, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(num_outputs))
        self.register_buffer("std", torch.zeros(num_outputs))
        self.eps = float(eps)
        assert 0 <= self.eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.std * x) + self.mean

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def fit(self, ds) -> dict:
        target = torch.stack([torch.tensor(x) for x in ds["target"]])
        print(f"target: {target.shape}")
        self.mean = target.mean(0).to(self.mean)
        self.std = target.std(0).to(self.std) + self.eps
        return self.state_dict()


class PowerTransform(torch.nn.Module):
    """
    Apply a power transform (Yeo-Johnson) featurewise to make data more Gaussian-like.
    Followed by applying a zero-mean, unit-variance normalization to the
    transformed output to rescale targets to [-1, 1].
    """

    def __init__(self, num_outputs, eps: float = 1e-8):
        super().__init__()
        self.num_outputs = num_outputs
        self.register_buffer("lmbdas", torch.zeros(num_outputs))
        self.register_buffer("mean", torch.zeros(num_outputs))
        self.register_buffer("std", torch.zeros(num_outputs))
        self.eps = float(eps)
        assert 0 <= self.eps

    def _yeo_johnson_transform(self, x, lmbda):
        """
        Return transformed input x following Yeo-Johnson transform with
        parameter lambda.
        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/fbb32eae5/sklearn/preprocessing/_data.py#L3354
        """
        x_out = x.clone()
        eps = torch.finfo(x.dtype).eps
        pos = x >= 0  # binary mask

        # when x >= 0
        if abs(lmbda) < eps:
            x_out[pos] = torch.log1p(x[pos])
        else:  # lmbda != 0
            x_out[pos] = (torch.pow(x[pos] + 1, lmbda) - 1) / lmbda

        # when x < 0
        if abs(lmbda - 2) > eps:
            x_out[~pos] = -(torch.pow(-x[~pos] + 1, 2 - lmbda) - 1) / (2 - lmbda)
        else:  # lmbda == 2
            x_out[~pos] = -torch.log1p(-x[~pos])

        return x_out

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        """
        Return inverse-transformed input x following Yeo-Johnson inverse
        transform with parameter lambda.
        Adapted from
        https://github.com/scikit-learn/scikit-learn/blob/fbb32eae5/sklearn/preprocessing/_data.py#L3383
        """
        x_out = x.clone()
        pos = x >= 0
        eps = torch.finfo(x.dtype).eps

        # when x >= 0
        if abs(lmbda) < eps:  # lmbda == 0
            x_out[pos] = torch.exp(x[pos]) - 1
        else:  # lmbda != 0
            x_out[pos] = torch.pow(x[pos] * lmbda + 1, 1 / lmbda) - 1

        # when x < 0
        if abs(lmbda - 2) > eps:  # lmbda != 2
            x_out[~pos] = 1 - torch.pow(-(2 - lmbda) * x[~pos] + 1, 1 / (2 - lmbda))
        else:  # lmbda == 2
            x_out[~pos] = 1 - torch.exp(-x[~pos])
        return x_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Undo standardization
        x = (self.std * x) + self.mean
        x_out = torch.zeros_like(x)
        for i in range(self.num_outputs):
            x_out[:, i] = self._yeo_johnson_inverse_transform(x[:, i], self.lmbdas[i])
        return x_out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        x_out = torch.zeros_like(x)
        for i in range(self.num_outputs):
            x_out[:, i] = self._yeo_johnson_transform(x[:, i], self.lmbdas[i])
        # Standardization
        x_out = (x_out - self.mean) / self.std
        return x_out

    def fit(self, ds) -> dict:
        target = torch.stack([torch.tensor(x) for x in ds["target"]])
        # Fit Yeo-Johnson lambdas
        transformer = _PowerTransformer(method="yeo-johnson", standardize=False)
        target = torch.tensor(transformer.fit_transform(target))
        self.lmbdas = torch.tensor(transformer.lambdas_)
        # Fit standardization scaling
        self.mean = target.mean(0).to(self.mean)
        self.std = target.std(0).to(self.std) + self.eps
        return self.state_dict()


class IdentityTransform(torch.nn.Identity):
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def fit(self, ds) -> dict:
        return self.state_dict()
