from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .sat_params import SatParams


def legendre_gll_nodes(n: int) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return Gauss–Lobatto–Legendre (GLL) nodes and quadrature weights.

    n = polynomial order N (GLL points count is N+1).
    """
    if n < 1:
        raise ValueError("The order n must be >= 1 (GLL points count is n+1).")

    p_n = np.polynomial.legendre.Legendre.basis(n)
    p_deriv = p_n.deriv(1)

    # Internal nodes are roots of P'_n
    internal_nodes = p_deriv.roots() if n > 1 else np.array([], dtype=float)

    nodes = np.r_[-1.0, internal_nodes, 1.0]
    nodes.sort()

    # GLL quadrature weights
    weights = 2.0 / (n * (n + 1) * (p_n(nodes) ** 2))
    return nodes.astype(float), weights.astype(float)


def build_D_LGL(N: int) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Build 1D differentiation matrix on GLL nodes (LGL collocation)."""
    x, weights = legendre_gll_nodes(N)
    p_N = np.polynomial.legendre.Legendre.basis(N)
    Lx = p_N(x)

    x_diff = x[:, None] - x[None, :]
    Lx_ratio = Lx[:, None] / Lx[None, :]

    np.fill_diagonal(x_diff, 1.0)  # avoid division by zero
    D = Lx_ratio / x_diff
    np.fill_diagonal(D, 0.0)

    # Diagonal endpoint entries
    D[0, 0] = -N * (N + 1) / 4.0
    D[N, N] = N * (N + 1) / 4.0
    return D.astype(float), x.astype(float), weights.astype(float)


def build_basic_operators(N: int, Ne: int) -> tuple[NDArray[np.floating], NDArray[np.floating], SatParams]:
    """Return (D, xi, sat_params) used by the cubed-sphere solver."""
    if Ne < 1:
        raise ValueError("Ne must be >= 1.")
    D, xi, w_gll = build_D_LGL(N)

    d_angle = (np.pi / 2.0) / Ne
    s_scale = 2.0 / d_angle

    tau_0 = s_scale / w_gll[0]
    tau_N = s_scale / w_gll[-1]

    sat_param = SatParams(s_scale=s_scale, tau_0=tau_0, tau_N=tau_N, w_gll=w_gll)
    return D, xi, sat_param