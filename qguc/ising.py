"""QUBO -> Ising conversion and coefficient normalisation for QAOA.

Spin convention (matches the project slides):

    x_i = (1 - z_i) / 2,   z_i in {-1, +1},   so x_i = 1  <->  z_i = -1.

Substituting into E(x) = offset + sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j:

    Q_ii x_i     = Q_ii/2 - (Q_ii/2) z_i
    Q_ij x_i x_j = (Q_ij/4) (1 - z_i - z_j + z_i z_j)

hence E = const + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j with

    h_i   = -Q_ii/2 - (1/4) sum_{j != i} Q_ij   (Q_ij read from the upper
                                                triangle, whichever order)
    J_ij  = Q_ij / 4                            (i < j)
    const = offset + (1/2) sum_i Q_ii + (1/4) sum_{i<j} Q_ij

Normalisation
-------------
QUBO coefficients here span ~1e6 to 1e9 INR. QAOA's cost layer applies
exp(-i gamma H_C); with coefficients that large any usable gamma wraps the
phase many times past 2 pi and the landscape in gamma becomes aliasing
noise. :func:`normalise` divides h and J by

    scale = max(max|h_i|, max|J_ij|)

so all coefficients lie in [-1, 1]. The physical energy is recovered as

    E_phys = const + scale * E_norm,   E_norm = sum h z + sum J z z

where h, J are the normalised coefficients. The map is a positive affine
transformation, so it preserves the ordering of states (and hence the
argmin), and any approximation ratio of the form
(E_max - E) / (E_max - E_min) is invariant to it; see
:func:`qguc.exact.approximation_ratio`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .qubo import QUBO


def bits_to_spins(x) -> np.ndarray:
    """x in {0,1} -> z in {+1,-1}, with x = 1 <-> z = -1."""
    return 1 - 2 * np.asarray(x, dtype=int)


def spins_to_bits(z) -> np.ndarray:
    return (1 - np.asarray(z, dtype=int)) // 2


@dataclass
class IsingModel:
    """E_phys(z) = const + scale * (sum_i h_i z_i + sum_{i<j} J_ij z_i z_j).

    ``h`` and ``J`` (strictly upper triangular) are the coefficients the
    circuit sees. ``scale`` is 1 for an unnormalised model; ``const`` is
    always in physical units (INR).
    """

    h: np.ndarray
    J: np.ndarray
    const: float
    scale: float = 1.0
    params: dict = field(default_factory=dict)

    @property
    def n_vars(self) -> int:
        return self.h.shape[0]

    @property
    def h_physical(self) -> np.ndarray:
        return self.h * self.scale

    @property
    def J_physical(self) -> np.ndarray:
        return self.J * self.scale

    def normalised_energy(self, z):
        """sum h z + sum J z z for spins z of shape (n,) or (..., n)."""
        z = np.asarray(z, dtype=float)
        return z @ self.h + np.einsum("...i,ij,...j->...", z, self.J, z)

    def energy(self, z):
        """Physical (INR) energy of spins z."""
        return self.to_physical(self.normalised_energy(z))

    def energy_from_bits(self, x):
        return self.energy(bits_to_spins(x))

    def to_physical(self, e_norm):
        return self.const + self.scale * np.asarray(e_norm, dtype=float)

    def to_normalised(self, e_phys):
        return (np.asarray(e_phys, dtype=float) - self.const) / self.scale


def qubo_to_ising(qubo: QUBO) -> IsingModel:
    """Exact, unnormalised (scale = 1) Ising form of ``qubo``."""
    Q = qubo.Q
    diag = np.diag(Q).copy()
    upper = np.triu(Q, k=1)
    row_coupling = upper.sum(axis=1) + upper.sum(axis=0)  # sum_{j != i} Q_ij
    h = -diag / 2.0 - row_coupling / 4.0
    J = upper / 4.0
    const = qubo.offset + diag.sum() / 2.0 + upper.sum() / 4.0
    return IsingModel(h=h, J=J, const=float(const), scale=1.0, params=dict(qubo.params))


def normalise(model: IsingModel) -> IsingModel:
    """Rescale so max(max|h|, max|J|) = 1. Physical energies are unchanged."""
    s = max(np.abs(model.h).max(initial=0.0), np.abs(model.J).max(initial=0.0))
    if s == 0.0:
        return replace(model)
    return replace(model, h=model.h / s, J=model.J / s, scale=model.scale * s)
