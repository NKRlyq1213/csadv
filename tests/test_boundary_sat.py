import numpy as np

from csadv.geometry import build_cubed_sphere_equiangular
from csadv.operators import build_basic_operators
from csadv.physics import get_velocity_field, evaluate_contravariant_on_face
from csadv.boundary import SatInflowPenalty


def _stack_u1u2(cube, field):
    u1 = np.zeros((6, cube.Ng, cube.Ng), dtype=float)
    u2 = np.zeros_like(u1)
    for k, fid in enumerate(["P1", "P2", "P3", "P4", "P5", "P6"]):
        face = cube.faces[fid]
        u1k, u2k = evaluate_contravariant_on_face(face.lam, face.lat, face.Ainv, field)
        u1[k] = u1k
        u2[k] = u2k
    return u1, u2


def test_sat_constant_field_zero_penalty():
    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    Ne = 1

    D, xi, sat_param = build_basic_operators(N, Ne)
    cube = build_cubed_sphere_equiangular(Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    field = get_velocity_field("rigid_rotation", u0=1.0, alpha0=0.3, use_colat=False)
    u1, u2 = _stack_u1u2(cube, field)

    state = np.ones((6, Ng, Ng), dtype=float) * 2.0
    bnd = SatInflowPenalty(sat_param=sat_param)

    pen = bnd.penalty(state, cube, u1, u2)
    assert np.allclose(pen, 0.0, atol=1e-12, rtol=0.0)


def test_sat_west_inflow_nonzero_on_p1_only():
    # Construct u1>0 everywhere, u2=0 => only west edge has inflow (Vn=-u1<0)
    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    Ne = 1

    D, xi, sat_param = build_basic_operators(N, Ne)
    cube = build_cubed_sphere_equiangular(Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    u1 = np.ones((6, Ng, Ng), dtype=float)
    u2 = np.zeros_like(u1)

    state = np.zeros((6, Ng, Ng), dtype=float)
    # P1 is index 0; give west edge q_in=1, neighbor (P4 east edge) q_out=0
    state[0, 0, :] = 1.0

    bnd = SatInflowPenalty(sat_param=sat_param)
    pen = bnd.penalty(state, cube, u1, u2)

    sqrtg_edge = cube.faces["P1"].sqrtg[0, :]
    expected = (-sat_param.tau_0)  # Vn=-1 => 0.5*(Vn-|Vn|)=-1

    assert np.allclose(pen[0, 0, :], expected, atol=1e-12, rtol=0.0)
    # Everywhere else should be zero (up to tiny float noise)
    pen[0, 0, :] = 0.0
    assert np.allclose(pen, 0.0, atol=1e-12, rtol=0.0)
