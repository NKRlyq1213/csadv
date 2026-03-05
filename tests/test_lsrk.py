import numpy as np

from csadv.integrators import lsrk5_step


def test_lsrk_zero_rhs_keeps_state():
    Ng = 5
    state = np.random.default_rng(0).normal(size=(6, Ng, Ng)).astype(float)
    du = np.zeros_like(state)

    def rhs(_s):
        return np.zeros_like(_s)

    state2, du2 = lsrk5_step(state.copy(), du, rhs, dt=0.1)
    assert np.allclose(state2, state, atol=0.0, rtol=0.0)
    assert np.allclose(du2, 0.0, atol=0.0, rtol=0.0)