import numpy as np

from csadv.geometry import build_cubed_sphere_equiangular, contravariant_to_uv
from csadv.physics import get_velocity_field, evaluate_contravariant_on_face


def test_rigid_rotation_alpha0_zero_matches_u_cos_lat():
    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    cube = build_cubed_sphere_equiangular(Ne=1, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    face = cube.faces["P1"]
    u0 = 2.5
    field = get_velocity_field("rigid_rotation", u0=u0, alpha0=0.0, use_colat=False)

    u, v = field.uv(face.lam, face.lat)
    assert u.shape == (Ng, Ng) and v.shape == (Ng, Ng)
    assert np.max(np.abs(v)) < 1e-12

    expected = u0 * np.cos(face.lat)  # lat is latitude since use_colat=False
    assert np.allclose(u, expected, atol=1e-12, rtol=1e-12)


def test_contravariant_roundtrip_on_face():
    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    cube = build_cubed_sphere_equiangular(Ne=1, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    face = cube.faces["P1"]
    field = get_velocity_field("rigid_rotation", u0=1.0, alpha0=0.3, use_colat=False)

    u1, u2 = evaluate_contravariant_on_face(face.lam, face.lat, face.Ainv, field)
    u_back, v_back = contravariant_to_uv(u1, u2, face.A)

    u, v = field.uv(face.lam, face.lat)
    assert np.allclose(u_back, u, atol=1e-10, rtol=1e-10)
    assert np.allclose(v_back, v, atol=1e-10, rtol=1e-10)