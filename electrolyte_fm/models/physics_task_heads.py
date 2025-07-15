import torch
from torch import nn


class ArrtheniusActivation(nn.Module):
    def forward(self, x: torch.Tensor, temperature: torch.Tensor):
        A, Ea = torch.chunk(x, 2, -1)
        return A * torch.exp(Ea / temperature)


class ArrheniusTaskHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.desc_skip_connection = True

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(embed_dim, int(0.5 * embed_dim))
        self.relu3 = nn.GELU()
        self.final = nn.Linear(int(0.5 * embed_dim), 2)

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

        logA = z[:, 0]
        Ea = z[:, 1]
        R = 8.63e-5

        e = torch.exp(torch.tensor([1], device=logA.device))
        C = torch.log10(e) / R

        cond = logA - C * Ea / temperature

        return cond


class VFTTaskHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.desc_skip_connection = True

        self.fc1 = nn.Linear(embed_dim, embed_dim)
        self.relu1 = nn.GELU()
        self.fc2 = nn.Linear(embed_dim, embed_dim)
        self.relu2 = nn.GELU()
        self.fc3 = nn.Linear(embed_dim, int(0.5 * embed_dim))
        self.relu3 = nn.GELU()
        self.final = nn.Linear(int(0.5 * embed_dim), 3)

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

        logA = z[:, 0]
        Ea = z[:, 1]
        T_g = z[:, 2]
        R = 8.63e-5

        e = torch.exp(torch.tensor([1], device=logA.device))
        C = torch.log10(e) / R

        cond = logA - C * Ea / (temperature - T_g)

        return cond


class VFTTaskHeadWithDecay(nn.Module):
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

        return ln_A - Ea / (temperature - Tg), alpha, beta, lmbda
