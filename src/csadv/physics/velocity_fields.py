from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from csadv.geometry.transforms import uv_to_contravariant


class VelocityField(Protocol):
    """Interface for velocity fields.

    The input (lam, lat_or_colat) are arrays on each face.
    """
    def uv(
        self,
        lam: NDArray[np.floating],
        lat_or_colat: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]: ...


@dataclass(frozen=True, slots=True)
class RigidRotationParams:
    """Solid-body rotation parameters (paper test case style)."""
    u0: float = 1.0
    alpha0: float = 0.0
    use_colat: bool = False  # if True, lat_or_colat is colatitude θ, else latitude φ


def paper_wind_uv(
    lam: NDArray[np.floating],
    lat_or_colat: NDArray[np.floating],
    u0: float = 1.0,
    alpha0: float = 0.0,
    use_colat: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Rigid rotation wind (u,v) on the sphere.

    Notes:
    - This matches your notebook's formula (solid-body rotation with tilt alpha0).
    - The formula expects latitude φ. If use_colat=True, we convert θ -> φ = π/2 - θ.
    """
    lat = (np.pi / 2.0 - lat_or_colat) if use_colat else lat_or_colat
    u = u0 * (np.cos(alpha0) * np.cos(lat) + np.sin(alpha0) * np.cos(lam) * np.sin(lat))
    v = -u0 * np.sin(alpha0) * np.sin(lam)
    return u.astype(float), v.astype(float)


class RigidRotationField:
    def __init__(self, params: RigidRotationParams):
        self.params = params

    def uv(
        self,
        lam: NDArray[np.floating],
        lat_or_colat: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        p = self.params
        return paper_wind_uv(lam, lat_or_colat, u0=p.u0, alpha0=p.alpha0, use_colat=p.use_colat)


# ----------------------------
# Registry (plug-in mechanism)
# ----------------------------

_VELOCITY_REGISTRY: dict[str, Callable[..., VelocityField]] = {}


def register_velocity_field(name: str, factory: Callable[..., VelocityField]) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("Velocity field name must be a non-empty string.")
    _VELOCITY_REGISTRY[name] = factory


def get_velocity_field(name: str, **kwargs) -> VelocityField:
    if name not in _VELOCITY_REGISTRY:
        raise KeyError(f"Unknown velocity field '{name}'. Available: {sorted(_VELOCITY_REGISTRY.keys())}")
    return _VELOCITY_REGISTRY[name](**kwargs)


def evaluate_contravariant_on_face(
    lam: NDArray[np.floating],
    lat_or_colat: NDArray[np.floating],
    Ainv: NDArray[np.floating],
    field: VelocityField,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute contravariant velocities (u1,u2) on a face using its Ainv."""
    u, v = field.uv(lam, lat_or_colat)
    return uv_to_contravariant(u, v, Ainv)


# Register built-ins
register_velocity_field(
    "rigid_rotation",
    lambda **kw: RigidRotationField(RigidRotationParams(**kw)),
)
def stack_contravariant_on_cube(cube, field):
    import numpy as np
    u1 = np.zeros((6, cube.Ng, cube.Ng), dtype=float)
    u2 = np.zeros_like(u1)
    for k, fid in enumerate(['P1','P2','P3','P4','P5','P6']):
        face = cube.faces[fid]
        u1k, u2k = evaluate_contravariant_on_face(face.lam, face.lat, face.Ainv, field)
        u1[k] = u1k
        u2[k] = u2k
    return u1, u2
