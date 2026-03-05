from __future__ import annotations

import numpy as np

from csadv.config import FACE_ORDER
from csadv.geometry import build_cubed_sphere_equiangular
from csadv.initial_conditions import get_ic
from csadv.integrators import compute_fixed_dt, integrate_fixed_dt
from csadv.operators import build_basic_operators
from csadv.physics import get_velocity_field, stack_contravariant_on_cube
from csadv.boundary import get_boundary
from csadv.rhs import compute_global_rhs


def run_convergence_test(
    Ng_list: list[int],
    *,
    R: float = 1.0,
    CFL: float = 0.1,
    Ne: int = 1,
    u0: float = 1.0,
    alpha0: float = 0.0,
    use_colat: bool = False,
    ic_name: str = "gaussian",
    ic_kwargs: dict | None = None,
    boundary_name: str = "sat_inflow",
    boundary_backend: str = "numpy",   # "numpy" or "numba"
    rhs_backend: str = "numpy",        # "numpy" or "numba"
    n_periods: int = 1,
) -> list[dict]:
    """Run convergence sweep over Ng_list.

    Returns list of dict with keys: Ng, N, dt, T_final, L2, Linf, order_L2, order_Linf.
    """
    if ic_kwargs is None:
        ic_kwargs = {}

    if R <= 0:
        raise ValueError("R must be positive.")
    if Ne < 1:
        raise ValueError("Ne must be >= 1.")
    if u0 <= 0:
        raise ValueError("u0 must be positive.")
    if n_periods < 0:
        raise ValueError("n_periods must be >= 0.")

    a = R / np.sqrt(3.0)

    results: list[dict] = []
    for Ng in Ng_list:
        N = Ng - 1
        D, xi, sat_param = build_basic_operators(N, Ne)
        w = sat_param.w_gll
        wi = w[:, None]
        wj = w[None, :]

        cube = build_cubed_sphere_equiangular(Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=use_colat)

        # IC
        ic = get_ic(ic_name, **ic_kwargs)
        phi0 = ic(cube)
        state0 = phi0.copy()

        # Velocity field (first version: rigid rotation)
        field = get_velocity_field("rigid_rotation", u0=u0, alpha0=alpha0, use_colat=use_colat)
        u1, u2 = stack_contravariant_on_cube(cube, field)

        # Boundary scheme (supports backend="numpy"/"numba" if you enabled it)
        boundary = get_boundary(boundary_name, sat_param=sat_param, backend=boundary_backend)

        # Time horizon: 1 revolution along great circle => T = 2πR / u0
        T_period = 2.0 * np.pi * R / u0
        T_final = n_periods * T_period

        dt_fixed = compute_fixed_dt(CFL=CFL, R=R, u_max=u0, N=N)

        def rhs_func(s):
            return compute_global_rhs(
                s, cube, D, sat_param, u1, u2, boundary,
                backend=rhs_backend,
            )

        state = integrate_fixed_dt(state0, rhs_func, dt_fixed=dt_fixed, t_final=T_final)

        err = state - phi0
        Linf = float(np.max(np.abs(err)))

        # L2 weighted by sqrtg * wi * wj (same as your notebook)
        err2 = 0.0
        for k, fid in enumerate(FACE_ORDER):
            sqrtg = cube.faces[fid].sqrtg
            area_w = sqrtg * wi * wj
            err2 += float(np.sum((err[k] ** 2) * area_w))
        L2 = float(np.sqrt(err2))

        results.append({
            "Ng": Ng,
            "N": N,
            "dt": dt_fixed,
            "T_final": T_final,
            "L2": L2,
            "Linf": Linf,
            "order_L2": None,
            "order_Linf": None,
        })

    # empirical order using h ~ 1/N
    for i in range(1, len(results)):
        N0 = results[i - 1]["N"]
        N1 = results[i]["N"]
        h_ratio = (1.0 / N1) / (1.0 / N0)  # = N0/N1
        if h_ratio <= 0:
            continue

        e0 = results[i - 1]["L2"]
        e1 = results[i]["L2"]
        if e0 > 0 and e1 > 0:
            results[i]["order_L2"] = float(np.log(e1 / e0) / np.log(h_ratio))

        e0 = results[i - 1]["Linf"]
        e1 = results[i]["Linf"]
        if e0 > 0 and e1 > 0:
            results[i]["order_Linf"] = float(np.log(e1 / e0) / np.log(h_ratio))

    return results