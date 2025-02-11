import torch

from .prediction_task_head import PredictionTaskHead


class ArrheniusTaskHead(PredictionTaskHead):
    def __init__(
        self, embed_dim: int, output_size: int = 1, dropout: float = 0.2
    ) -> None:
        super().__init__(embed_dim=embed_dim, output_size=3)

    def forward(self, emb, temperature):
        emb = emb[:, 0, :]
        x_out = self.fc1(emb)
        x_out = self.dropout1(x_out)
        x_out = self.relu1(x_out)

        if self.desc_skip_connection is True:
            x_out = x_out + emb

        z = self.fc2(x_out)
        z = self.dropout2(z)
        z = self.relu2(z)
        if self.desc_skip_connection is True:
            z = self.final(z + x_out)
        else:
            z = self.final(z)

        R = 8.314  # ideal gas constant in J/K.mol
        A = z[:, 0]
        Ea = z[:, 1]
        return torch.mul(A, torch.exp(-1 * Ea / R * temperature))
