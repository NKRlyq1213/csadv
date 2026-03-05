from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from csadv.config import FACE_ORDER, FACE_IDX
from csadv.geometry.cubed_sphere import CubeGeometry
from csadv.operators import SatParams
from csadv.boundary.base import BoundaryScheme

try:
    from numba import njit
except Exception:
    njit = None


def adv_kernel_numpy(
    phi: NDArray[np.floating],
    sqrtg: NDArray[np.floating],
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    D: NDArray[np.floating],
    s_scale: float,
) -> NDArray[np.floating]:
    # √g * φ * u1 / u2
    sg_phi_u1 = sqrtg * phi * u1
    sg_phi_u2 = sqrtg * phi * u2
    sqrtgu1 = sqrtg * u1
    sqrtgu2 = sqrtg * u2

    # α direction
    Dα_Fα = (s_scale / sqrtg) * (D @ sg_phi_u1)
    Aα_DαU = u1 * (s_scale * (D @ phi))
    U_DαAα = phi * ((s_scale / sqrtg) * (D @ sqrtgu1))

    # β direction
    Dβ_Fβ = (s_scale / sqrtg) * (sg_phi_u2 @ D.T)
    Aβ_DβU = u2 * (s_scale * (phi @ D.T))
    U_DβAβ = phi * ((s_scale / sqrtg) * (sqrtgu2 @ D.T))

    adv = 0.5 * (Dα_Fα + Aα_DαU - U_DαAα) + 0.5 * (Dβ_Fβ + Aβ_DβU - U_DβAβ)
    return adv.astype(float)


if njit is not None:
    @njit(cache=True)
    def adv_kernel_numba(phi, sqrtg, u1, u2, D, s_scale):
        Ng = phi.shape[0]
        # temp arrays
        sg_phi_u1 = np.empty((Ng, Ng), dtype=np.float64)
        sg_phi_u2 = np.empty((Ng, Ng), dtype=np.float64)
        sqrtgu1 = np.empty((Ng, Ng), dtype=np.float64)
        sqrtgu2 = np.empty((Ng, Ng), dtype=np.float64)

        for i in range(Ng):
            for j in range(Ng):
                sg = sqrtg[i, j]
                p = phi[i, j]
                u1ij = u1[i, j]
                u2ij = u2[i, j]
                sg_phi_u1[i, j] = sg * p * u1ij
                sg_phi_u2[i, j] = sg * p * u2ij
                sqrtgu1[i, j] = sg * u1ij
                sqrtgu2[i, j] = sg * u2ij

        # matrix multiplies via loops
        D_sgphi_u1 = np.empty((Ng, Ng), dtype=np.float64)
        D_phi = np.empty((Ng, Ng), dtype=np.float64)
        D_sqrtgu1 = np.empty((Ng, Ng), dtype=np.float64)

        for i in range(Ng):
            for j in range(Ng):
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for k in range(Ng):
                    Dik = D[i, k]
                    s1 += Dik * sg_phi_u1[k, j]
                    s2 += Dik * phi[k, j]
                    s3 += Dik * sqrtgu1[k, j]
                D_sgphi_u1[i, j] = s1
                D_phi[i, j] = s2
                D_sqrtgu1[i, j] = s3

        sgphi_u2_Dt = np.empty((Ng, Ng), dtype=np.float64)
        phi_Dt = np.empty((Ng, Ng), dtype=np.float64)
        sqrtgu2_Dt = np.empty((Ng, Ng), dtype=np.float64)

        # A @ D.T => sum_k A[i,k]*D[j,k]
        for i in range(Ng):
            for j in range(Ng):
                s1 = 0.0
                s2 = 0.0
                s3 = 0.0
                for k in range(Ng):
                    Djk = D[j, k]
                    s1 += sg_phi_u2[i, k] * Djk
                    s2 += phi[i, k] * Djk
                    s3 += sqrtgu2[i, k] * Djk
                sgphi_u2_Dt[i, j] = s1
                phi_Dt[i, j] = s2
                sqrtgu2_Dt[i, j] = s3

        adv = np.empty((Ng, Ng), dtype=np.float64)
        for i in range(Ng):
            for j in range(Ng):
                sg = sqrtg[i, j]
                inv_sg = 1.0 / sg
                p = phi[i, j]
                u1ij = u1[i, j]
                u2ij = u2[i, j]

                Dα_Fα = (s_scale * inv_sg) * D_sgphi_u1[i, j]
                Aα_DαU = u1ij * (s_scale * D_phi[i, j])
                U_DαAα = p * ((s_scale * inv_sg) * D_sqrtgu1[i, j])

                Dβ_Fβ = (s_scale * inv_sg) * sgphi_u2_Dt[i, j]
                Aβ_DβU = u2ij * (s_scale * phi_Dt[i, j])
                U_DβAβ = p * ((s_scale * inv_sg) * sqrtgu2_Dt[i, j])

                adv[i, j] = 0.5 * (Dα_Fα + Aα_DαU - U_DαAα) + 0.5 * (Dβ_Fβ + Aβ_DβU - U_DβAβ)
        return adv
else:
    adv_kernel_numba = None


def compute_global_rhs(
    state: NDArray[np.floating],
    cube: CubeGeometry,
    D: NDArray[np.floating],
    sat_param: SatParams,
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    boundary: BoundaryScheme,
    *,
    backend: str = "numpy",
    out: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """RHS = -adv + boundary_penalty.

    backend: "numpy" or "numba" (for adv kernel).
    """
    Ng = cube.Ng
    if state.shape != (6, Ng, Ng):
        raise ValueError(f"state must have shape {(6, Ng, Ng)}")
    if u1.shape != (6, Ng, Ng) or u2.shape != (6, Ng, Ng):
        raise ValueError(f"u1/u2 must have shape {(6, Ng, Ng)}")

    if out is None:
        out = np.empty_like(state, dtype=float)

    s_scale = float(sat_param.s_scale)
    pen = boundary.penalty(state, cube, u1, u2)

    use_numba = (backend == "numba")
    if use_numba and adv_kernel_numba is None:
        raise RuntimeError("backend='numba' requested but numba is not installed.")

    for fid in FACE_ORDER:
        idx = FACE_IDX[fid]
        face = cube.faces[fid]
        sqrtg = face.sqrtg
        phi = state[idx]
        u1f = u1[idx]
        u2f = u2[idx]

        if use_numba:
            adv = adv_kernel_numba(phi, sqrtg, u1f, u2f, D, s_scale)
        else:
            adv = adv_kernel_numpy(phi, sqrtg, u1f, u2f, D, s_scale)

        out[idx] = (-adv + pen[idx]).astype(float)

    return out