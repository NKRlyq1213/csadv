from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

# 5-stage low-storage Runge–Kutta coefficients (same as your notebook)
A_RK = np.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
], dtype=float)

B_RK = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
], dtype=float)


def compute_fixed_dt(CFL: float, R: float, u_max: float, N: int) -> float:
    """Match your notebook dt rule but make R explicit."""
    if CFL <= 0:
        raise ValueError("CFL must be positive.")
    if R <= 0:
        raise ValueError("R must be positive.")
    if u_max <= 0:
        raise ValueError("u_max must be positive.")
    if N < 1:
        raise ValueError("N must be >= 1.")
    return float(CFL * R / (u_max * (N**2)))


def lsrk5_step(
    state: NDArray[np.floating],
    du: NDArray[np.floating],
    rhs_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    dt: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """One time step using 5-stage low-storage RK.

    state, du: shape (6, Ng, Ng)
    rhs_func: RHS(state) -> same shape
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    for j in range(5):
        rhs = rhs_func(state)
        du = A_RK[j] * du + dt * rhs
        state = state + B_RK[j] * du
    return state, du


def integrate_fixed_dt(
    state0: NDArray[np.floating],
    rhs_func: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    dt_fixed: float,
    t_final: float,
) -> NDArray[np.floating]:
    """Integrate from t=0 to t_final with fixed dt (last step clipped)."""
    if t_final < 0:
        raise ValueError("t_final must be >= 0.")
    if dt_fixed <= 0:
        raise ValueError("dt_fixed must be positive.")

    state = state0.copy()
    du = np.zeros_like(state)

    t = 0.0
    while t < t_final - 1e-12:
        dt = dt_fixed if (t + dt_fixed <= t_final) else (t_final - t)
        state, du = lsrk5_step(state, du, rhs_func, dt)
        t += dt
    return state