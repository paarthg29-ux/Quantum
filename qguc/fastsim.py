"""Fast statevector QAOA for a DIAGONAL cost Hamiltonian (pure NumPy).

Our cost Hamiltonian H_C is a sum of Z and ZZ terms, so it is diagonal in
the computational basis with eigenvalue E[k] on basis state k, and E is
exactly the energy vector brute force already enumerates. Hence

    cost layer   exp(-i gamma H_C)  = elementwise  psi * exp(-i gamma E)
    mixer layer  exp(-i beta H_M)   = n single-qubit 2x2 rotations
    <H_C>                           = sum_k |psi_k|^2 E[k]

No 2^n x 2^n (dense or sparse) matrix is ever formed; the largest arrays
are a few state-sized vectors. tests/test_fastsim.py checks every piece
against PennyLane, which stays in the repository only as that oracle.

Conventions
-----------
* Basis state k <-> bits x_j = (k >> j) & 1, the numbering of
  :mod:`qguc.exact`, so E from brute force can be used as is. Qubit j is
  variable j; spin z_j = 1 - 2 x_j is the Z eigenvalue (|0> -> +1).
  (PennyLane orders wire 0 as the MOST significant bit, so the oracle puts
  variable j on wire n-1-j.)
* A mixer is any object with ``unitaries(beta) -> (n, 2, 2)`` (or
  (B, n, 2, 2) for an array of B betas): one 2x2 unitary per qubit, which
  may depend on beta and on per-qubit angles fixed at construction. The
  initial state is any normalised 2^n vector. These two pieces are what the
  QAOA arms vary; everything else here is shared.
* E is in normalised Ising units (max|h, J| = 1) when built by
  :func:`energy_vector`; the approximation ratio is invariant to that.
* complex128 by default; ``dtype=np.complex64`` halves memory and traffic
  at ~1e-6 relative accuracy (scripts/s4_fastsim_timing.py reports the
  measured difference).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .exact import brute_force, iter_bitstrings
from .instance import UCInstance
from .ising import IsingModel, bits_to_spins, normalise, qubo_to_ising
from .qubo import QUBO, build_qubo

_REAL = {np.dtype(np.complex128): np.float64, np.dtype(np.complex64): np.float32}


def _n_qubits(size: int) -> int:
    n = int(size).bit_length() - 1
    if size != 1 << n:
        raise ValueError(f"length {size} is not a power of two")
    return n


# ---- energy vector -------------------------------------------------------


@dataclass
class EnergyVector:
    """Diagonal of the normalised cost Hamiltonian and its exact spectrum facts."""

    E: np.ndarray  # (2^n,) float64, normalised Ising units, state numbering of qguc.exact
    ising: IsingModel  # normalised; E[k] = ising.normalised_energy(z(k))
    qubo: QUBO
    e_min: float
    e_max: float
    ground: np.ndarray  # all (degenerate) ground-state indices
    elapsed_s: float

    @property
    def n(self) -> int:
        return _n_qubits(self.E.shape[0])

    @property
    def sigma(self) -> float:
        """Std of E over the uniform distribution, = sqrt(sum h^2 + sum J^2)."""
        return float(np.sqrt((self.ising.h**2).sum() + (self.ising.J**2).sum()))

    def ratio(self, energy):
        """Approximation ratio (E_max - E) / (E_max - E_min)."""
        return (self.e_max - np.asarray(energy)) / (self.e_max - self.e_min)


def energy_vector(instance: UCInstance, lambda_demand: float, lambda_switch: float = 0.0) -> EnergyVector:
    """Energy vector of ``build_qubo(instance, ...)``, computed once by brute
    force and cached on the instance (key: the QUBO parameters)."""
    key = ("energy_vector", float(lambda_demand), float(lambda_switch), instance.demand_bias_mw)
    if key not in instance.cache:
        t0 = time.perf_counter()
        qubo = build_qubo(instance, lambda_demand, lambda_switch=lambda_switch)
        bf = brute_force(qubo, return_energies=True)
        ising = normalise(qubo_to_ising(qubo))
        E = ising.to_normalised(bf.energies)
        instance.cache[key] = EnergyVector(
            E=E, ising=ising, qubo=qubo, e_min=float(E[bf.ground_indices[0]]), e_max=float(E.max()),
            ground=np.sort(bf.ground_indices), elapsed_s=time.perf_counter() - t0,
        )
    return instance.cache[key]


def ising_energy_vector(h, J, chunk_size: int = 1 << 16) -> np.ndarray:
    """E[k] = sum_j h_j z_j + sum_{i<j} J_ij z_i z_j for every basis state k."""
    h, J = np.asarray(h, float), np.triu(np.asarray(J, float), 1)
    model = IsingModel(h=h, J=J, const=0.0)
    return np.concatenate([model.normalised_energy(bits_to_spins(X)) for _, X in iter_bitstrings(len(h), chunk_size)])


# ---- states --------------------------------------------------------------


def uniform_state(n: int, dtype=np.complex128) -> np.ndarray:
    """|+>^n."""
    return np.full(1 << n, 2.0 ** (-n / 2), dtype=dtype)


def product_state(amplitudes, dtype=np.complex128) -> np.ndarray:
    """Tensor product of per-qubit states; ``amplitudes`` has shape (n, 2),
    row j = (amplitude of |0>, amplitude of |1>) of qubit j."""
    amps = np.asarray(amplitudes, dtype=np.complex128)
    psi = np.ones(1, dtype=np.complex128)
    for a in amps:  # qubit j becomes bit j
        psi = np.outer(a, psi).ravel()
    return psi.astype(dtype)


def ry_product_state(thetas, dtype=np.complex128) -> np.ndarray:
    """prod_j RY(theta_j)|0>: qubit j has P(x_j = 1) = sin^2(theta_j / 2)."""
    t = np.asarray(thetas, float)
    return product_state(np.stack([np.cos(t / 2), np.sin(t / 2)], axis=1), dtype)


# ---- mixers --------------------------------------------------------------


def _axis_rotation(beta, cos_t, sin_t) -> np.ndarray:
    """exp(-i beta (cos_t Z + sin_t X)) per qubit; beta scalar -> (n, 2, 2),
    beta of shape (B,) -> (B, n, 2, 2)."""
    beta = np.asarray(beta, float)
    c, s = np.cos(beta)[..., None], np.sin(beta)[..., None]
    U = np.empty(beta.shape + (len(cos_t), 2, 2), dtype=np.complex128)
    U[..., 0, 0] = c - 1j * s * cos_t
    U[..., 0, 1] = -1j * s * sin_t
    U[..., 1, 0] = -1j * s * sin_t
    U[..., 1, 1] = c + 1j * s * cos_t
    return U


class XMixer:
    """Standard mixer exp(-i beta sum_j X_j) = prod_j RX(2 beta)."""

    def __init__(self, n: int):
        self.n = n

    def unitaries(self, beta) -> np.ndarray:
        return _axis_rotation(beta, np.zeros(self.n), np.ones(self.n))


class RotatedMixer:
    """Per-qubit axis mixer exp(-i beta (cos t_j Z_j + sin t_j X_j)).

    Its per-qubit +1 eigenstate is RY(t_j)|0> (see :func:`ry_product_state`),
    so it is the natural mixer for a product-state warm start; t_j = pi/2
    for all j recovers :class:`XMixer` and |+>^n.
    """

    def __init__(self, thetas):
        self.thetas = np.asarray(thetas, float)
        self.n = len(self.thetas)

    def unitaries(self, beta) -> np.ndarray:
        return _axis_rotation(beta, np.cos(self.thetas), np.sin(self.thetas))


class CustomMixer:
    """Wraps ``fn(beta) -> (n, 2, 2)`` (any per-qubit unitaries)."""

    def __init__(self, n: int, fn):
        self.n, self.fn = n, fn

    def unitaries(self, beta) -> np.ndarray:
        beta = np.asarray(beta, float)
        if beta.ndim == 0:
            return np.asarray(self.fn(float(beta)), dtype=np.complex128)
        return np.stack([np.asarray(self.fn(float(b)), dtype=np.complex128) for b in beta])


# ---- primitive operations --------------------------------------------------


def apply_diagonal_phase(psi, E, gamma) -> np.ndarray:
    """psi * exp(-i gamma E). psi (N,) with scalar gamma, or (B, N) with
    gamma of shape (B,)."""
    psi = np.asarray(psi)
    out = np.array(psi, copy=True)
    _phase_inplace(out.reshape(-1, out.shape[-1]), np.asarray(E, _REAL[out.dtype]), np.atleast_1d(gamma))
    return out


def apply_single_qubit(psi, qubit: int, U) -> np.ndarray:
    """Apply the 2x2 ``U`` to ``qubit``: the state is viewed as
    (2^(n-1-q), 2, 2^q) and the middle axis is mixed. psi (N,) with U (2, 2),
    or (B, N) with U (2, 2) or (B, 2, 2)."""
    psi = np.asarray(psi)
    out = np.array(psi, copy=True)
    U = np.asarray(U, dtype=np.complex128)
    flat = out.reshape(-1, out.shape[-1])
    _rot_inplace(flat, qubit, np.broadcast_to(U, (flat.shape[0], 2, 2)))
    return out


def _phase_inplace(psi2: np.ndarray, E: np.ndarray, gammas: np.ndarray) -> None:
    """psi2 (B, N) *= exp(-i gamma_b E); E already in psi2's real dtype."""
    real = _REAL[psi2.dtype]
    for b in range(psi2.shape[0]):  # B is small; one fused pass per row
        ang = E * real(-gammas[b])
        f = np.empty(psi2.shape[1], dtype=psi2.dtype)
        np.cos(ang, out=f.real)
        np.sin(ang, out=f.imag)
        psi2[b] *= f


def _rot_inplace(psi2: np.ndarray, q: int, U: np.ndarray) -> None:
    """psi2 (B, N): apply U[b] (2x2) to qubit q of row b, in place."""
    B = psi2.shape[0]
    v = psi2.reshape(B, -1, 2, 1 << q)
    a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
    U = U.astype(psi2.dtype, copy=False)
    u00, u01, u10, u11 = (U[:, r, c].reshape(B, 1, 1) for r, c in ((0, 0), (0, 1), (1, 0), (1, 1)))
    t = a0.copy()
    a0 *= u00
    a0 += u01 * a1
    a1 *= u11
    a1 += u10 * t


def _group_kron(U: np.ndarray) -> np.ndarray:
    """U (B, k, 2, 2), qubits q0..q0+k-1 -> (B, 2^k, 2^k) = kron(U_{k-1}, ..., U_0)
    (higher qubit = more significant bit within the group)."""
    B = U.shape[0]
    M = np.ones((B, 1, 1), dtype=np.complex128)
    for i in range(U.shape[1]):
        m = M.shape[1]
        M = np.einsum("bij,bkl->bikjl", U[:, i], M).reshape(B, 2 * m, 2 * m)
    return M


def _apply_mixer(psi2: np.ndarray, U: np.ndarray, buf: np.ndarray, group: int) -> np.ndarray:
    """Apply the per-qubit unitaries U (B, n, 2, 2) to psi2 (B, N).

    Qubits are taken ``group`` at a time: their 2x2s are combined into one
    2^group x 2^group Kronecker product and applied with a single BLAS matmul
    to the state viewed as (B, 2^(n-q0-k), 2^k, 2^q0). This is the same
    operation as ``group`` calls of :func:`apply_single_qubit`, with ``group``
    times fewer passes over memory. ``buf`` is a second state-sized buffer;
    returns whichever of (psi2, buf) holds the result.
    """
    B = psi2.shape[0]
    n = U.shape[1]
    src, dst = psi2, buf
    for q0 in range(0, n, group):
        k = min(group, n - q0)
        K = 1 << k
        M = _group_kron(U[:, q0:q0 + k]).astype(src.dtype, copy=False)
        if q0 == 0:  # lowest qubits are contiguous: one (rows x K) @ (K x K)
            np.matmul(src.reshape(B, -1, K), np.swapaxes(M, -1, -2), out=dst.reshape(B, -1, K))
        else:
            shape = (B, -1, K, 1 << q0)
            np.matmul(M[:, None], src.reshape(shape), out=dst.reshape(shape))
        src, dst = dst, src
    return src


# ---- QAOA ----------------------------------------------------------------


def probabilities(psi) -> np.ndarray:
    psi = np.asarray(psi)
    return psi.real.astype(np.float64) ** 2 + psi.imag.astype(np.float64) ** 2


def expectation(psi, E) -> np.ndarray | float:
    """<H_C> = sum |psi|^2 E (accumulated in float64)."""
    out = probabilities(psi) @ np.asarray(E, np.float64)
    return float(out) if np.ndim(out) == 0 else out


def sample(psi, shots: int, rng) -> np.ndarray:
    """``shots`` basis-state indices drawn from |psi|^2."""
    cdf = np.cumsum(probabilities(psi))
    return np.minimum(np.searchsorted(cdf, rng.random(shots) * cdf[-1], side="right"), cdf.size - 1)


class QAOASimulator:
    """QAOA on a diagonal cost Hamiltonian E with injectable initial state
    and mixer:

        |psi> = prod_{k=1..p} [ M(beta_k) exp(-i gamma_k H_C) ] |psi_0>

    ``state``/``energy`` take gammas, betas of shape (p,), or (B, p) for a
    batch of B parameter sets evaluated together (rows of shape (B, 2^n)).
    """

    def __init__(self, E, initial_state=None, mixer=None, dtype=np.complex128, group: int = 4):
        self.group = group  # qubits per mixer matmul (see _apply_mixer); 1 = qubit by qubit
        self.dtype = np.dtype(dtype)
        if self.dtype not in _REAL:
            raise ValueError("dtype must be complex128 or complex64")
        self.E = np.asarray(E, np.float64)
        self.n = _n_qubits(self.E.shape[0])
        self._E = self.E.astype(_REAL[self.dtype])
        psi0 = uniform_state(self.n, self.dtype) if initial_state is None else np.asarray(initial_state)
        if psi0.shape != self.E.shape:
            raise ValueError(f"initial state has shape {psi0.shape}, expected {self.E.shape}")
        if not np.isclose(np.vdot(psi0, psi0).real, 1.0, atol=1e-6):
            raise ValueError("initial state is not normalised")
        self.initial_state = psi0.astype(self.dtype)
        self.mixer = XMixer(self.n) if mixer is None else mixer
        if self.mixer.n != self.n:
            raise ValueError(f"mixer acts on {self.mixer.n} qubits, state has {self.n}")

    def state(self, gammas, betas) -> np.ndarray:
        g, b = np.asarray(gammas, float), np.asarray(betas, float)
        if g.shape != b.shape or g.ndim not in (1, 2):
            raise ValueError("gammas and betas must both have shape (p,) or (B, p)")
        batched = g.ndim == 2
        g2, b2 = np.atleast_2d(g), np.atleast_2d(b)
        psi = np.repeat(self.initial_state[None, :], g2.shape[0], axis=0)
        buf = np.empty_like(psi)
        for k in range(g2.shape[1]):
            _phase_inplace(psi, self._E, g2[:, k])
            U = self.mixer.unitaries(b2[:, k])  # (B, n, 2, 2)
            if self.group <= 1:
                for q in range(self.n):
                    _rot_inplace(psi, q, U[:, q])
            else:
                out = _apply_mixer(psi, U, buf, self.group)
                if out is buf:
                    psi, buf = buf, psi
        return psi if batched else psi[0]

    def energy(self, gammas, betas):
        return expectation(self.state(gammas, betas), self.E)
