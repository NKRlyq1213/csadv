import numpy as np
from csadv.operators import legendre_gll_nodes, build_D_LGL, build_basic_operators, SatParams


def test_gll_nodes_weights_basic():
    x, w = legendre_gll_nodes(3)
    assert x.shape == (4,)
    assert np.isclose(x[0], -1.0) and np.isclose(x[-1], 1.0)
    assert np.all(w > 0)
    assert np.isclose(np.sum(w), 2.0)


def test_D_shape_and_endpoints():
    N = 4
    D, x, w = build_D_LGL(N)
    assert D.shape == (N + 1, N + 1)
    assert np.isclose(D[0, 0], -N * (N + 1) / 4.0)
    assert np.isclose(D[N, N], N * (N + 1) / 4.0)


def test_basic_operators_returns_satparams():
    D, xi, sat = build_basic_operators(4, Ne=2)
    assert isinstance(sat, SatParams)
    assert sat.w_gll.shape == (5,)