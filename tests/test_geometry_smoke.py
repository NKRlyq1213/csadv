import numpy as np

from csadv.geometry import build_cubed_sphere_equiangular


def test_build_cube_smoke():
    R = 1.0
    a = R / np.sqrt(3.0)
    Ne = 1
    Ng = 5
    N = Ng - 1

    cube = build_cubed_sphere_equiangular(Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=False)
    assert cube.Ng == Ng
    assert set(cube.faces.keys()) == {"P1", "P2", "P3", "P4", "P5", "P6"}

    f = cube.faces["P1"]
    assert f.X.shape == (Ng, Ng)
    assert f.A.shape == (Ng, Ng, 2, 2)
    assert np.all(np.isfinite(f.sqrtg))
    assert np.all(f.sqrtg > 0.0)

    # 檢查 Ainv 近似為 A 的逆（取一個內點）
    i, j = 2, 2
    A = f.A[i, j]
    Ainv = f.Ainv[i, j]
    I = A @ Ainv
    assert np.allclose(I, np.eye(2), atol=1e-10, rtol=1e-10)