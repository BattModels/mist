import math

import torch

kT = 0.0256  # kT at 298K in eV (k_B * T / e)
e = 1.0  # elementary charge
LN_HA_CONCENTRATION = 2.52  # ln([HA])


def f_HA(ETN: torch.Tensor) -> torch.Tensor:
    a0, a1, a2 = 1.605, -2.906, 1.626
    return a0 + a1 * ETN + a2 * ETN**2


def f_SHE_Li(beta: torch.Tensor) -> torch.Tensor:
    b0, b1, b2 = -4.011, 2.258, -1.032
    return b0 + b1 * beta + b2 * beta**2


def U_dis(beta: torch.Tensor) -> torch.Tensor:
    # TODO: specify the function for U_dis based on beta
    return


def h_abstraction_energy(
    pka_value: torch.Tensor, beta: torch.Tensor, ETN: torch.Tensor
) -> torch.Tensor:
    """
    Calculate hydrogen abstraction energy in eV.

    ΔG = (ΔG_H*Li2O2 - 1/2 ΔG_H2^0) - f_HA(ET^N)
         - kT/2 (ln[HA] - ln(10)·pKa) + e(U_dis(ET^N) + f^SHE_Li+||Li(β))
    """
    term1 = -1.5  # DELTA_G_H_LI2O2 - 0.5 * DELTA_G_H2_0
    term2 = f_HA(ETN)
    term3 = (kT / 2) * (LN_HA_CONCENTRATION - math.log(10) * pka_value)
    term4 = e * (U_dis(ETN) + f_SHE_Li(beta))

    return term1 - term2 - term3 + term4


def f_O2_O2minus(ETN: torch.Tensor) -> torch.Tensor:
    d0, d1, d2 = -0.948, 0.192, 0.427
    return d0 + d1 * ETN + d2 * ETN**2


def nucleophilic_attack_bound(
    pka_value: torch.Tensor, beta: torch.Tensor, ETN: torch.Tensor
) -> torch.Tensor:
    """
    Calculate nucleophilic attack bound.

    r_n ≤ exp((f_SHE_Li+||Li(β) - f_O2/O2-(ET^N)) / kT - ln(10)/2 × pKa)
    """
    numerator = f_SHE_Li(beta) - f_O2_O2minus(ETN)
    exponent = numerator / kT - (math.log(10) / 2) * pka_value
    return torch.exp(exponent)


def solution_mediated_reaction(beta: torch.Tensor, ETN: torch.Tensor) -> torch.Tensor:
    """
    Calculate solution mediated reaction free energy in eV.

    ΔG_HA^sol = f_SHE_Li+||Li(β) - f_O2/O2-(ET^N) ≤ -0.35 eV
    """
    return f_SHE_Li(beta) - f_O2_O2minus(ETN)


def eORR_eOER_bound(beta: torch.Tensor) -> torch.Tensor:
    """
    Calculate eORR/eOER bound.
    """
    return -0.785 * beta + 1.147


def eORR_bound(beta: torch.Tensor) -> torch.Tensor:
    """
    Calculate eORR bound.
    """
    return -2.849 * beta**2 + 0.107 * beta + 1.936
