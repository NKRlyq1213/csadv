import numpy as np

from csadv.geometry import build_cubed_sphere_equiangular
from csadv.initial_conditions import get_ic


def test_gaussian_ic_shape_and_range():
    R = 1.0
    a = R / np.sqrt(3.0)
    Ng = 5
    N = Ng - 1
    cube = build_cubed_sphere_equiangular(Ne=1, Ng=Ng, N=N, a=a, R=R, use_colat=False)

    ic = get_ic(
        "gaussian",
        lam0=0.0,
        lat0_or_colat0=0.0,   # latitude=0
        sigma_rad=0.4,        # relatively wide for tiny Ng
        amp=2.0,
        background=0.5,
        use_colat=False,
    )
    phi = ic(cube)

    assert phi.shape == (6, Ng, Ng)
    assert np.all(np.isfinite(phi))
    assert phi.min() >= 0.5 - 1e-12
    assert phi.max() <= 2.5 + 1e-12
    assert phi.max() > 0.5  # should have a bump