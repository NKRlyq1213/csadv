from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from numpy.typing import NDArray

from csadv.operators import legendre_gll_nodes
from .transforms import lati_and_longitude, build_Atilde


@dataclass(frozen=True, slots=True)
class FaceGeometry:
    """Single face geometry (panel) on the cubed-sphere."""
    x: NDArray[np.floating]       # panel-plane x (Ng, Ng)
    y: NDArray[np.floating]       # panel-plane y (Ng, Ng)
    X: NDArray[np.floating]       # spherical X   (Ng, Ng)
    Y: NDArray[np.floating]       # spherical Y   (Ng, Ng)
    Z: NDArray[np.floating]       # spherical Z   (Ng, Ng)
    alpha: NDArray[np.floating]   # local alpha   (Ng, Ng)
    beta: NDArray[np.floating]    # local beta    (Ng, Ng)
    lam: NDArray[np.floating]     # longitude     (Ng, Ng)
    lat: NDArray[np.floating]     # latitude/colat (Ng, Ng)
    sqrtg: NDArray[np.floating]   # Jacobian √g   (Ng, Ng)
    A: NDArray[np.floating]       # (u¹,u²)->(u,v) (Ng, Ng,2,2)
    Ainv: NDArray[np.floating]    # (u,v)->(u¹,u²) (Ng, Ng,2,2)


@dataclass(frozen=True, slots=True)
class CubeGeometry:
    """Cubed-sphere geometry with six faces."""
    faces: Dict[str, FaceGeometry]  # keys: 'P1'..'P6'
    Ne: int
    Ng: int
    N: int
    a: float
    R: float
    use_colat: bool


def build_equiangular_face(
    face_id: str,
    Ne: int,
    Ng: int,
    N: int,
    a: float,
    R: float,
    use_colat: bool = False,
) -> FaceGeometry:
    """Build one equiangular face geometry.

    目前版本對 Ne 不做細分（和你 notebook 一樣），先保留參數以便之後擴充多元素。
    """
    if Ng != N + 1:
        raise ValueError("Require Ng = N + 1 (GLL points count).")
    if Ne < 1:
        raise ValueError("Ne must be >= 1.")

    g, _ = legendre_gll_nodes(N)  # (Ng,)
    S = (g + 1.0) * 0.5
    ones = np.ones(Ng)

    alphaL, alphaR = -np.pi / 4.0, np.pi / 4.0
    betaB, betaT = -np.pi / 4.0, np.pi / 4.0

    alpha = alphaL + np.outer(S, ones) * (alphaR - alphaL)
    beta = betaB + np.outer(ones, S) * (betaT - betaB)

    x = a * np.tan(alpha)
    y = a * np.tan(beta)
    r = np.sqrt(a * a + x * x + y * y)

    if face_id == "P1":       # +X
        X = R * a / r
        Y = R * x / r
        Z = R * y / r
    elif face_id == "P2":     # +Y
        X = -R * x / r
        Y = R * a / r
        Z = R * y / r
    elif face_id == "P3":     # -X
        X = -R * a / r
        Y = -R * x / r
        Z = R * y / r
    elif face_id == "P4":     # -Y
        X = R * x / r
        Y = -R * a / r
        Z = R * y / r
    elif face_id == "P5":     # +Z
        X = -R * y / r
        Y = R * x / r
        Z = R * a / r
    elif face_id == "P6":     # -Z
        X = R * y / r
        Y = R * x / r
        Z = -R * a / r
    else:
        raise ValueError("face_id must be one of 'P1'..'P6'.")

    lam, lat_or_colat = lati_and_longitude(X, Y, Z, return_colat=use_colat)
    # build_Atilde 的 th 參數在你 notebook 版本是「緯度/共緯度」都可切換，但請保持 use_colat 全程一致
    A, Ainv, sqrtg = build_Atilde(face_id, lam, lat_or_colat, alpha, beta, R)

    return FaceGeometry(
        x=x.astype(float),
        y=y.astype(float),
        X=X.astype(float),
        Y=Y.astype(float),
        Z=Z.astype(float),
        alpha=alpha.astype(float),
        beta=beta.astype(float),
        lam=lam.astype(float),
        lat=lat_or_colat.astype(float),
        sqrtg=sqrtg.astype(float),
        A=A.astype(float),
        Ainv=Ainv.astype(float),
    )


def build_cubed_sphere_equiangular(
    Ne: int,
    Ng: int,
    N: int,
    a: float,
    R: float,
    use_colat: bool = False,
) -> CubeGeometry:
    faces: Dict[str, FaceGeometry] = {}
    for fid in ["P1", "P2", "P3", "P4", "P5", "P6"]:
        faces[fid] = build_equiangular_face(fid, Ne, Ng, N, a, R, use_colat=use_colat)
    return CubeGeometry(faces=faces, Ne=Ne, Ng=Ng, N=N, a=a, R=R, use_colat=use_colat)