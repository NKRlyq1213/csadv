from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from csadv.geometry.cubed_sphere import CubeGeometry
from .base import stack_faces, register_ic


def _to_lat(lat_or_colat: NDArray[np.floating] | float, use_colat: bool) -> NDArray[np.floating] | float:
    """Convert (latitude or colatitude) -> latitude."""
    return (np.pi / 2.0 - lat_or_colat) if use_colat else lat_or_colat


def great_circle_distance(
    lam: NDArray[np.floating],
    lat_or_colat: NDArray[np.floating],
    lam0: float,
    lat0_or_colat0: float,
    R: float,
    use_colat: bool = False,
) -> NDArray[np.floating]:
    """Great-circle distance on a sphere."""
    lat = _to_lat(lat_or_colat, use_colat)
    lat0 = float(_to_lat(lat0_or_colat0, use_colat))

    dlam = lam - lam0
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)

    cos_c = sin_lat0 * sin_lat + cos_lat0 * cos_lat * np.cos(dlam)
    cos_c = np.clip(cos_c, -1.0, 1.0)
    central_angle = np.arccos(cos_c)
    return (R * central_angle).astype(float)


@dataclass(frozen=True, slots=True)
class GaussianIC:
    """Gaussian bump defined by spherical distance from (lam0, lat0)."""
    lam0: float
    lat0_or_colat0: float
    sigma_m: float | None = None
    sigma_rad: float | None = None
    amp: float = 1.0
    background: float = 0.0
    use_colat: bool = False

    def __call__(self, cube: CubeGeometry) -> NDArray[np.floating]:
        if (self.sigma_m is None) == (self.sigma_rad is None):
            raise ValueError("Provide exactly one of sigma_m or sigma_rad.")

        if self.sigma_rad is not None:
            sigma_dist = cube.R * float(self.sigma_rad)  # convert to distance
        else:
            sigma_dist = float(self.sigma_m)

        if sigma_dist <= 0.0:
            raise ValueError("sigma must be positive.")

        face_to_phi: dict[str, NDArray[np.floating]] = {}
        for fid, face in cube.faces.items():
            dist = great_circle_distance(
                face.lam, face.lat, self.lam0, self.lat0_or_colat0, cube.R, use_colat=self.use_colat
            )
            phi = self.background + self.amp * np.exp(-(dist * dist) / (2.0 * sigma_dist * sigma_dist))
            face_to_phi[fid] = phi.astype(float)

        return stack_faces(cube, face_to_phi)


# Register built-in IC
register_ic(
    "gaussian",
    lambda **kw: GaussianIC(**kw),
)