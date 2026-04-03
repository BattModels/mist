import math

import torch

kT = 0.0256  # k_B T at 298.15 K, in eV
e = 1.0  # elementary charge
LN_HA_CONCENTRATION = 2.52  # ln([HA])


def f_HA(ETN: torch.Tensor) -> torch.Tensor:
    a0, a1, a2 = 1.605, -2.906, 1.626
    return a0 + a1 * ETN + a2 * ETN**2


def h_abstraction_energy(
    pka_value: torch.Tensor, beta: torch.Tensor, ETN: torch.Tensor
) -> torch.Tensor:
    """
    Calculate hydrogen abstraction energy in eV.

    ΔG = (ΔG_H*Li2O2 - 1/2 ΔG_H2^0) - f_HA(ET^N)
         - kT/2 (ln[HA] - ln(10)·pKa) + e(U_dis + f^SHE_Li+||Li(β))
    """
    term1 = -1.5  # DELTA_G_H_LI2O2 - 0.5 * DELTA_G_H2_0
    U_dis = 2.67  # V
    term2 = f_HA(ETN)
    term3 = (kT / 2) * (LN_HA_CONCENTRATION - math.log(10) * pka_value)
    term4 = e * (U_dis + f_SHE_Li(beta))

    return term1 - term2 - term3 + term4


def f_SHE_Li(beta: torch.Tensor) -> torch.Tensor:
    return -3.330 - 0.226 * beta - 1.075 * beta**2


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
    beta_ACN = 0.19
    ETN_ACN = 0.46
    numerator_ACN = f_SHE_Li(beta_ACN) - f_O2_O2minus(ETN_ACN)
    rn_ACN = numerator_ACN / kT - (math.log(10) / 2) * 30
    numerator = f_SHE_Li(beta) - f_O2_O2minus(ETN)
    exponent = numerator / kT - (math.log(10) / 2) * pka_value
    # Use exp(a - b) instead of exp(a)/exp(b) to avoid numerical underflow
    return torch.exp(exponent - rn_ACN)


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
    return 2.849 * beta**2 + 0.107 * beta + 1.936
