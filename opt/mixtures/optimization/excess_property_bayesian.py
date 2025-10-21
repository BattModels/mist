from dataclasses import dataclass
import torch
import time
import numpy as np
from tqdm import tqdm
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim.optimize import optimize_acqf_discrete
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

from smirk import SmirkTokenizerFast
from excess_property_optimization import ExcessOptimizer


@dataclass
class BayesOptConfig:
    num_iterations: int
    target_quantities: list[str]
    n_init: int
    n_iter_candidates: int
    num_comp: int
    inventory_size: int = 10_000
    precomputed_inventory: str | None = None


class ExcessBayesianOptimizer(ExcessOptimizer):
    def __init__(
        self, pretrained_ckpt: str, temperature: float, config: BayesOptConfig
    ) -> None:
        self.pretrained_ckpt = pretrained_ckpt
        self.model = self._load_model(self.pretrained_ckpt)
        self.tokenizer = SmirkTokenizerFast()
        self.temperature = temperature
        self.config = config
        self.n = self.config.num_comp
        self.target_trajectory = []

    def sample_pairs(self, num_samples: int):
        """
        Uniform random sampling without replacement.
        """
        inv_size = self.inventory_matrix.shape[0]
        # total possible pairs = n*(n-1)/2
        all_i, all_j = torch.triu_indices(
            inv_size, inv_size, offset=1
        )  # upper-triangular, no self-pairs
        num_pairs = all_i.numel()
        # random subset
        idx = torch.randperm(num_pairs)[:num_samples]
        pairs = torch.stack([all_i[idx], all_j[idx]], dim=1)
        return pairs

    def pair_feature(self, i_idx: torch.Tensor, j_idx: torch.Tensor) -> torch.Tensor:
        """
        Return z_ij = [E_i, E_j, |E_i - E_j|, E_i * E_j] on CPU (double).
        """
        E = self.inventory_matrix
        Ei = E[i_idx]
        Ej = E[j_idx]
        z = torch.cat([Ei + Ej, (Ei - Ej).abs(), Ei * Ej], dim=-1)  # (k, 3D)
        return z.detach().cpu().to(torch.double)

    def fit_gp(self, train_X: torch.Tensor, train_Y: torch.Tensor):
        d = train_X.shape[-1]
        gp = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=d),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        return gp

    @torch.no_grad()
    def score_pairs(
        self,
        pairs_ij: torch.Tensor,
    ):
        """
        Score many pairs by sweeping compositions and taking per-target maxima.
        Scalar score = sum_t max_c |y_excess_t / y_t|(c).
        """

        E = self.inventory_matrix
        n_pairs = pairs_ij.shape[0]
        # Build embs once per pair
        Ei = E[pairs_ij[:, 0]].unsqueeze(1)  # (n_samples, D)
        Ej = E[pairs_ij[:, 1]].unsqueeze(1)  # (n_samples, D)
        embs_pairs = torch.concat([Ei, Ej], dim=1)  # (n_samples, 2, D)
        # [[e_1a, e_1b], [e_2a, e_2b]]
        # --> [[e_1a, e_1b],  [e_1a, e_1b], ..., [e_2a, e_2b], [e_2a, e_2b]]
        embs_pairs = embs_pairs.repeat_interleave(
            repeats=self.n, dim=0
        )  # (n_composition*n_samples, 2, D)
        comp = torch.linspace(0, 1, steps=self.n)
        comp = torch.vstack((comp, 1 - comp)).T  # (n_composition*n_samples, 2)
        comp = comp.repeat(n_pairs, 1)
        y_rel = self.forward(comp, embs_pairs)
        mask = (
            torch.tensor(
                [
                    i in self.config.target_quantities
                    for i in self.model.config.target_columns
                ]
            )
            .unsqueeze(0)
            .repeat(n_pairs * self.n, 1)
        )
        y_masked = torch.masked_select(y_rel, mask)
        y_masked = y_masked.reshape(n_pairs, self.n, -1)
        if y_masked.ndim > 1:
            mx, _ = torch.max(torch.abs(y_masked), dim=1)
        else:
            mx = torch.abs(y_masked).max()
        return mx

    def run_optimization(self, batch_q: int = 1):
        self.runtime = time.time()
        # Initial design: random pairs, score = max over c (batched)
        init_pairs = self.sample_pairs(num_samples=self.config.n_init * 4)[
            : self.config.n_init
        ]

        # Build initial train set (X=z_ij, Y=score)
        train_X = self.pair_feature(init_pairs[:, 0], init_pairs[:, 1])  # (n_init, d)
        train_Y = self.score_pairs(init_pairs).to(torch.double)  # (n_init,)

        # Metadata to recover best solution and plot
        meta_pairs = init_pairs.clone()  # (n_init, 2)

        for _ in tqdm(range(self.config.num_iterations)):
            gp = self.fit_gp(train_X, train_Y)
            acqf = qLogExpectedImprovement(model=gp, best_f=train_Y.max())

            # candidate set of new pairs (P,2)
            cand_pairs = self.sample_pairs(num_samples=self.config.n_iter_candidates)
            if cand_pairs.numel() == 0:
                break

            # build candidate features (P,d)
            X_cand = self.pair_feature(cand_pairs[:, 0], cand_pairs[:, 1])

            # discrete acquisition argmax over candidate feature set
            X_sel, _ = optimize_acqf_discrete(
                acq_function=acqf, choices=X_cand, q=batch_q
            )
            # map back to indices
            with torch.no_grad():
                diffs = ((X_cand.unsqueeze(0) - X_sel.unsqueeze(1)) ** 2).sum(
                    dim=-1
                )  # (q,P)
                pick = torch.argmin(diffs, dim=1).cpu()  # (q,)

            chosen_pairs = cand_pairs[pick]  # (q,2)

            # Evaluate new pairs (score = max over c), append to train set
            y_new = self.score_pairs(chosen_pairs)  # (q,), (q,C)
            train_X = torch.cat([train_X, X_sel], dim=0)  # (N+q, d)
            train_Y = torch.cat([train_Y, y_new], dim=0)  # (N+q, 1)
            meta_pairs = torch.cat([meta_pairs, chosen_pairs], dim=0)  # (N+q, 2)

            # Track best-so-far
            self.target_trajectory.append(float(train_Y.max().item()))

        best_idx = int(torch.argmax(train_Y).item())
        best_pair = meta_pairs[best_idx].tolist()

        i_idx, j_idx = best_pair
        self.optimal_mixture = [
            {
                "compounds": [self.inventory[i_idx], self.inventory[j_idx]],
                "temperature": self.temperature,
            }
        ]

        self.target_trajectory = np.array(self.target_trajectory, dtype=float)
        self.runtime = time.time() - self.runtime


if __name__ == "__main__":
    # density target 0.279603
    name_or_path = "/Users/anoushka/VSCodeProjects/electrolyte-fm/data/models/mist-mixtures-17o0xho9"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = BayesOptConfig(
        num_iterations=10,
        target_quantities=["density [gram / centimeter ** 3]"],
        num_comp=21,
        n_init=64,
        n_iter_candidates=4000,
        inventory_size=10_000,
    )
    # ["molar volume [centimeter ** 3 / mole]"]
    # ["density [gram / centimeter ** 3]"]
    opt = ExcessBayesianOptimizer(name_or_path, temperature=298.15, config=config)
    opt.run_optimization()
    opt.save_run()
