"""Warm-start QAOA machinery (Egger, Marecek & Woerner, Quantum 5, 479, 2021).

Everything is driven by a per-variable probability vector c (length n_vars,
c_i = desired P(x_i = 1)); where c comes from is what distinguishes arms.

Regularisation
--------------
c_i -> clip(c_i, eps, 1 - eps), default eps = 0.1. At c_i in {0, 1} the
mixer below is diagonal (+-Z) and can never flip qubit i, so the warm start
would be frozen in; this is a correctness requirement, not cosmetics.

Initial state
-------------
theta_i = 2 arcsin(sqrt(c_i)),   |psi_0> = prod_i RY(theta_i)|0>,
so qubit i is |1> with probability sin^2(theta_i / 2) = c_i.

Warm-start mixer
----------------
The X mixer's extremal eigenstate is |+>^n, so pairing it with a warm start
pulls the state back toward uniform. The warm mixer has |psi_0> as its
GROUND state:

    H_M^(i) = [[2c_i - 1,              -2 sqrt(c_i (1 - c_i))],
               [-2 sqrt(c_i (1 - c_i)), 1 - 2c_i            ]]
            = -RY(theta_i) Z RY(-theta_i) = -(cos theta_i Z + sin theta_i X)

    exp(-i beta H_M^(i)) = RY(theta_i) RZ(-2 beta) RY(-theta_i).

Both identities, the ground-state property and the c = 1/2 limit are checked
numerically in tests/test_warmstart.py. At c_i = 1/2, H_M^(i) = -X: the
mixer is exp(+i beta X), the standard X mixer exp(-i beta X) with beta ->
-beta (the same family of unitaries; arm A's beta init is symmetric about
0), and |psi_0> = |+>^n.
"""

from __future__ import annotations

import numpy as np

from .fastsim import ry_product_state

DEFAULT_EPS = 0.1


def regularise(c, eps: float = DEFAULT_EPS) -> np.ndarray:
    c = np.asarray(c, float)
    if not 0.0 <= eps <= 0.5:  # eps = 0.5 is the sanity limit: c = 1/2 -> uniform state, X mixer (beta -> -beta)
        raise ValueError("eps must be in [0, 0.5]")
    return np.clip(c, eps, 1.0 - eps)


def thetas_from_c(c) -> np.ndarray:
    return 2.0 * np.arcsin(np.sqrt(np.asarray(c, float)))


def warm_initial_state(c, dtype=np.complex128) -> np.ndarray:
    """prod_i RY(theta_i)|0> for an (already regularised) c."""
    return ry_product_state(thetas_from_c(c), dtype)


def mixer_hamiltonians(c) -> np.ndarray:
    """H_M^(i) for every qubit, shape (n, 2, 2)."""
    c = np.asarray(c, float)
    off = -2.0 * np.sqrt(c * (1.0 - c))
    H = np.empty((len(c), 2, 2))
    H[:, 0, 0], H[:, 1, 1] = 2 * c - 1, 1 - 2 * c
    H[:, 0, 1] = H[:, 1, 0] = off
    return H


class WarmStartMixer:
    """exp(-i beta H_M^(i)) on every qubit, for an (already regularised) c."""

    def __init__(self, c):
        self.c = np.asarray(c, float)
        self.n = len(self.c)
        th = thetas_from_c(self.c)
        self._cos, self._sin = np.cos(th), np.sin(th)  # cos th = 1 - 2c, sin th = 2 sqrt(c(1-c))

    def unitaries(self, beta) -> np.ndarray:
        # H_M = -(cos th Z + sin th X), so exp(-i beta H_M) = cos(beta) I + i sin(beta) (cos th Z + sin th X)
        beta = np.asarray(beta, float)
        cb, sb = np.cos(beta)[..., None], np.sin(beta)[..., None]
        U = np.empty(beta.shape + (self.n, 2, 2), dtype=np.complex128)
        U[..., 0, 0] = cb + 1j * sb * self._cos
        U[..., 1, 1] = cb - 1j * sb * self._cos
        U[..., 0, 1] = U[..., 1, 0] = 1j * sb * self._sin
        return U
