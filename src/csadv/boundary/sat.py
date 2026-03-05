from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from csadv.config import FACE_ORDER, FACE_IDX
from csadv.operators import SatParams
from csadv.geometry.cubed_sphere import CubeGeometry
from .base import BoundaryScheme, register_boundary

try:
    from numba import njit
except Exception:  # numba not installed
    njit = None


CONN_TABLE: dict[tuple[int, int], tuple[int, int, bool]] = {
    (0, 0): (3, 1, False), (0, 1): (1, 0, False), (0, 2): (5, 3, False), (0, 3): (4, 2, False),
    (1, 0): (0, 1, False), (1, 1): (2, 0, False), (1, 2): (5, 1, True),  (1, 3): (4, 1, False),
    (2, 0): (1, 1, False), (2, 1): (3, 0, False), (2, 2): (5, 2, True),  (2, 3): (4, 3, True),
    (3, 0): (2, 1, False), (3, 1): (0, 0, False), (3, 2): (5, 0, False), (3, 3): (4, 0, True),
    (4, 0): (3, 3, True),  (4, 1): (1, 3, False), (4, 2): (0, 3, False), (4, 3): (2, 3, True),
    (5, 0): (3, 2, False), (5, 1): (1, 2, True),  (5, 2): (2, 2, True),  (5, 3): (0, 2, False),
}


def extract_boundary_val(state: NDArray[np.floating], face_idx: int, side_idx: int) -> NDArray[np.floating]:
    nbr_face, nbr_side, reverse = CONN_TABLE[(face_idx, side_idx)]
    nbr = state[nbr_face]
    if nbr_side == 0:
        arr = nbr[0, :]
    elif nbr_side == 1:
        arr = nbr[-1, :]
    elif nbr_side == 2:
        arr = nbr[:, 0]
    else:
        arr = nbr[:, -1]
    if reverse:
        arr = arr[::-1]
    return arr.astype(float)


def _penalty_edge_numpy(Vn: NDArray[np.floating], q_in: NDArray[np.floating], q_out: NDArray[np.floating], tau: float) -> NDArray[np.floating]:
    flux_diff = 0.5 * (Vn - np.abs(Vn)) * (q_in - q_out)  # only inflow contributes
    return (flux_diff * tau).astype(float)


if njit is not None:
    @njit(cache=True)
    def _penalty_edge_numba(Vn, q_in, q_out, tau):
        n = Vn.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            vn = Vn[i]
            a = vn if vn >= 0.0 else -vn
            coef = 0.5 * (vn - a)  # = 0 when vn>=0, = vn when vn<0
            out[i] = (coef * (q_in[i] - q_out[i])) * tau
        return out
else:
    _penalty_edge_numba = None


@dataclass(frozen=True, slots=True)
class SatInflowPenalty(BoundaryScheme):
    sat_param: SatParams
    backend: str = "numpy"  # "numpy" or "numba"

    def penalty(
        self,
        state: NDArray[np.floating],
        cube: CubeGeometry,
        u1: NDArray[np.floating],
        u2: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        Ng = cube.Ng
        pen_global = np.zeros_like(state, dtype=float)

        tau_0 = float(self.sat_param.tau_0)
        tau_N = float(self.sat_param.tau_N)

        use_numba = (self.backend == "numba")
        if use_numba and _penalty_edge_numba is None:
            raise RuntimeError("backend='numba' requested but numba is not installed.")

        for fid in FACE_ORDER:
            idx = FACE_IDX[fid]
            sqrtg = cube.faces[fid].sqrtg
            phi = state[idx]
            u1f = u1[idx]
            u2f = u2[idx]
            pen = np.zeros((Ng, Ng), dtype=float)

            # West
            Vn = -u1f[0, :]
            q_out = extract_boundary_val(state, idx, 0)
            sqrtg_edge = sqrtg[0, :]
            edge = _penalty_edge_numba(Vn, phi[0, :], q_out, tau_0) if use_numba else _penalty_edge_numpy(Vn, phi[0, :], q_out, tau_0)
            pen[0, :] += edge * (1.0 / sqrtg_edge)

            # East
            Vn = u1f[-1, :]
            q_out = extract_boundary_val(state, idx, 1)
            sqrtg_edge = sqrtg[-1, :]
            edge = _penalty_edge_numba(Vn, phi[-1, :], q_out, tau_N) if use_numba else _penalty_edge_numpy(Vn, phi[-1, :], q_out, tau_N)
            pen[-1, :] += edge * (1.0 / sqrtg_edge)

            # South
            Vn = -u2f[:, 0]
            q_out = extract_boundary_val(state, idx, 2)
            sqrtg_edge = sqrtg[:, 0]
            edge = _penalty_edge_numba(Vn, phi[:, 0], q_out, tau_0) if use_numba else _penalty_edge_numpy(Vn, phi[:, 0], q_out, tau_0)
            pen[:, 0] += edge * (1.0 / sqrtg_edge)

            # North
            Vn = u2f[:, -1]
            q_out = extract_boundary_val(state, idx, 3)
            sqrtg_edge = sqrtg[:, -1]
            edge = _penalty_edge_numba(Vn, phi[:, -1], q_out, tau_N) if use_numba else _penalty_edge_numpy(Vn, phi[:, -1], q_out, tau_N)
            pen[:, -1] += edge * (1.0 / sqrtg_edge)

            pen_global[idx] = pen

        return pen_global


register_boundary("sat_inflow", lambda **kw: SatInflowPenalty(**kw))