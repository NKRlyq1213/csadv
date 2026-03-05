from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def lati_and_longitude(
    X: NDArray[np.floating],
    Y: NDArray[np.floating],
    Z: NDArray[np.floating],
    return_colat: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return longitude λ and (latitude φ) or (colatitude θ).

    - return_colat=False: latitude φ ∈ [-π/2, π/2]
    - return_colat=True : colatitude θ ∈ [0, π]
    """
    R = np.sqrt(X * X + Y * Y + Z * Z)
    lam = np.arctan2(Y, X)  # (-π, π]

    z_over_r = np.clip(Z / R, -1.0, 1.0)
    if return_colat:
        lat_or_colat = np.arccos(z_over_r)   # θ
    else:
        lat_or_colat = np.arcsin(z_over_r)   # φ
    return lam.astype(float), lat_or_colat.astype(float)


def build_Atilde(
    face_id: str,
    lam: NDArray[np.floating],
    th: NDArray[np.floating],
    alpha: NDArray[np.floating],
    beta: NDArray[np.floating],
    R: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Build A, Ainv and sqrtg for a given face.

    A maps (u¹,u²) -> (u,v). Ainv maps (u,v) -> (u¹,u²).
    """
    Ng = lam.shape[0]
    A = np.empty((Ng, Ng, 2, 2), dtype=float)
    Ainv = np.empty_like(A)

    cos_th = np.cos(th)
    cos_lam = np.cos(lam)
    sin_th = np.sin(th)
    sin_lam = np.sin(lam)

    sin2th = sin_th**2
    sin2lam = sin_lam**2
    cos2th = cos_th**2
    cos2lam = cos_lam**2

    sec_alpha = 1.0 / np.cos(alpha)
    sec_beta = 1.0 / np.cos(beta)
    sec2a = sec_alpha**2
    sec2b = sec_beta**2

    if face_id == "P1":
        A[..., 0, 0] = R * cos2lam * cos_th * sec2a
        A[..., 0, 1] = 0.0
        A[..., 1, 0] = -R * cos_th * cos_lam * sin_th * sin_lam * sec2a
        A[..., 1, 1] = R * cos2th * cos_lam * sec2b
    elif face_id == "P2":
        A[..., 0, 0] = R * sin2lam * cos_th * sec2a
        A[..., 0, 1] = 0.0
        A[..., 1, 0] = R * cos_th * cos_lam * sin_th * sin_lam * sec2a
        A[..., 1, 1] = R * cos2th * sin_lam * sec2b
    elif face_id == "P3":
        A[..., 0, 0] = R * cos2lam * cos_th * sec2a
        A[..., 0, 1] = 0.0
        A[..., 1, 0] = -R * cos_th * cos_lam * sin_th * sin_lam * sec2a
        A[..., 1, 1] = -R * cos2th * cos_lam * sec2b
    elif face_id == "P4":
        A[..., 0, 0] = R * sin2lam * cos_th * sec2a
        A[..., 0, 1] = 0.0
        A[..., 1, 0] = R * cos_th * cos_lam * sin_th * sin_lam * sec2a
        A[..., 1, 1] = -R * cos2th * sin_lam * sec2b
    elif face_id == "P5":
        A[..., 0, 0] = R * sin_th * cos_lam * sec2a
        A[..., 0, 1] = R * sin_th * sin_lam * sec2b
        A[..., 1, 0] = -R * sin2th * sin_lam * sec2a
        A[..., 1, 1] = R * sin2th * cos_lam * sec2b
    elif face_id == "P6":
        A[..., 0, 0] = -R * sin_th * cos_lam * sec2a
        A[..., 0, 1] = R * sin_th * sin_lam * sec2b
        A[..., 1, 0] = R * sin2th * sin_lam * sec2a
        A[..., 1, 1] = R * sin2th * cos_lam * sec2b
    else:
        raise ValueError("face_id must be one of 'P1'..'P6'.")

    detA = A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    Ainv[..., 0, 0] = A[..., 1, 1] / detA
    Ainv[..., 0, 1] = -A[..., 0, 1] / detA
    Ainv[..., 1, 0] = -A[..., 1, 0] / detA
    Ainv[..., 1, 1] = A[..., 0, 0] / detA

    sqrtg = np.abs(detA)
    return A, Ainv, sqrtg


def uv_to_contravariant(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    Ainv: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """(u,v) -> (u¹,u²)."""
    u1 = Ainv[..., 0, 0] * u + Ainv[..., 0, 1] * v
    u2 = Ainv[..., 1, 0] * u + Ainv[..., 1, 1] * v
    return u1, u2


def contravariant_to_uv(
    u1: NDArray[np.floating],
    u2: NDArray[np.floating],
    A: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """(u¹,u²) -> (u,v)."""
    u = A[..., 0, 0] * u1 + A[..., 0, 1] * u2
    v = A[..., 1, 0] * u1 + A[..., 1, 1] * u2
    return u, v