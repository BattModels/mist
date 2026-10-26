import torch
from torch import nn


class ArrtheniusActivation(nn.Module):
    def forward(self, x: torch.Tensor, temperature: torch.Tensor):
        A, Ea = torch.chunk(x, 2, -1)
        return A * torch.exp(Ea / temperature)


class LinearExogenousEffect(nn.Module):
    def forward(self, x: torch.Tensor, temperature: torch.Tensor):
        a, b = torch.chunk(x, 2, -1)
        return a + b * (temperature / 293.15)


class ArrheniusTaskHead(nn.Module):
    """Predict one Arrhenius response per target, shaped [batch, output_size]."""

    def __init__(self, embed_dim: int, output_size: int = 1) -> None:
        super().__init__()
        self.desc_skip_connection = True
        self.output_size = output_size

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(embed_dim, int(0.5 * embed_dim))
        self.relu3 = nn.GELU()
        self.final = nn.Linear(int(0.5 * embed_dim), 2 * output_size)

    def forward(self, emb, temperature):
        x_out = self.fc1(emb)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.relu2(z)
        z = self.fc3(z)
        z = self.relu3(z)
        z = self.final(z)

        logA, Ea = z.reshape(emb.shape[0], self.output_size, 2).unbind(dim=-1)
        temperature = temperature.reshape(-1, 1)
        R = 8.63e-5

        e = torch.exp(torch.tensor([1], device=logA.device))
        C = torch.log10(e) / R

        cond = logA - C * Ea / temperature

        return cond


class VFTTaskHead(nn.Module):
    """Predict one VFT response per target, shaped [batch, output_size]."""

    def __init__(self, embed_dim: int, output_size: int = 1) -> None:
        super().__init__()
        self.desc_skip_connection = True
        self.output_size = output_size

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(embed_dim, int(0.5 * embed_dim))
        self.relu3 = nn.GELU()
        self.final = nn.Linear(int(0.5 * embed_dim), 3 * output_size)

    def forward(self, emb, temperature):
        x_out = self.fc1(emb)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.relu2(z)
        z = self.fc3(z)
        z = self.relu3(z)
        z = self.final(z)

        logA, Ea, T_g = z.reshape(emb.shape[0], self.output_size, 3).unbind(dim=-1)
        temperature = temperature.reshape(-1, 1)
        R = 8.63e-5

        e = torch.exp(torch.tensor([1], device=logA.device))
        C = torch.log10(e) / R

        cond = logA - C * Ea / (temperature - T_g)

        return cond


class VFTDecayTaskHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.desc_skip_connection = True

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(embed_dim, int(0.5 * embed_dim))
        self.relu3 = nn.GELU()
        self.final = nn.Linear(int(0.5 * embed_dim), 6)
        self.sigmoid = nn.Sigmoid()

    def forward(self, emb, temperature):
        x_out = self.fc1(emb)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.relu2(z)
        z = self.fc3(z)
        z = self.relu3(z)
        z = self.final(z)

        ln_A = z[:, 0]
        Ea = z[:, 1]
        Tg = z[:, 2]
        alpha = self.sigmoid(z[:, 3])
        beta = z[:, 4]
        lmbda = self.sigmoid(z[:, 5])
        params = {
            "conductivity": ln_A - Ea / (temperature - Tg),
            "ln_A": ln_A,
            "Ea": Ea,
            "Tg": Tg,
            "alpha": alpha,
            "beta": beta,
            "lmbda": lmbda,
        }
        return params
