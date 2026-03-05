import numpy as np
import pytest

from csadv.geometry import build_cubed_sphere_equiangular
from csadv.operators import build_basic_operators
from csadv.physics import get_velocity_field, stack_contravariant_on_cube
from csadv.boundary import SatInflowPenalty
from csadv.rhs import compute_global_rhs


def test_rhs_numpy_runs():
    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    Ne = 1

    D, xi, sat_param = build_basic_operators(N, Ne)
    cube = build_cubed_sphere_equiangular(Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    field = get_velocity_field("rigid_rotation", u0=1.0, alpha0=0.3, use_colat=False)
    u1, u2 = stack_contravariant_on_cube(cube, field)

    state = np.random.default_rng(0).normal(size=(6, Ng, Ng)).astype(float)
    bnd = SatInflowPenalty(sat_param=sat_param, backend="numpy")

    rhs = compute_global_rhs(state, cube, D, sat_param, u1, u2, bnd, backend="numpy")
    assert rhs.shape == state.shape
    assert np.all(np.isfinite(rhs))


def test_rhs_numpy_numba_match():
    numba = pytest.importorskip("numba")

    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    Ne = 1

    D, xi, sat_param = build_basic_operators(N, Ne)
    cube = build_cubed_sphere_equiangular(Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    field = get_velocity_field("rigid_rotation", u0=1.0, alpha0=0.3, use_colat=False)
    u1, u2 = stack_contravariant_on_cube(cube, field)

    state = np.random.default_rng(1).normal(size=(6, Ng, Ng)).astype(float)

    bnd_np = SatInflowPenalty(sat_param=sat_param, backend="numpy")
    bnd_nb = SatInflowPenalty(sat_param=sat_param, backend="numba")

    rhs_np = compute_global_rhs(state, cube, D, sat_param, u1, u2, bnd_np, backend="numpy")
    rhs_nb = compute_global_rhs(state, cube, D, sat_param, u1, u2, bnd_nb, backend="numba")

    # 浮點運算路徑不同，允許很小誤差
    assert np.allclose(rhs_np, rhs_nb, atol=1e-10, rtol=1e-10)