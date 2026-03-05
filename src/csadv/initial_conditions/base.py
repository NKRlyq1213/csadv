from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from csadv.config import FACE_ORDER
from csadv.geometry.cubed_sphere import CubeGeometry


class InitialCondition(Protocol):
    """Return scalar field on all 6 faces: shape (6, Ng, Ng) in FACE_ORDER."""
    def __call__(self, cube: CubeGeometry) -> NDArray[np.floating]: ...


_IC_REGISTRY: dict[str, Callable[..., InitialCondition]] = {}


def face_order() -> list[str]:
    return list(FACE_ORDER)


def register_ic(name: str, factory: Callable[..., InitialCondition]) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("IC name must be a non-empty string.")
    _IC_REGISTRY[name] = factory


def get_ic(name: str, **kwargs) -> InitialCondition:
    if name not in _IC_REGISTRY:
        raise KeyError(f"Unknown IC '{name}'. Available: {sorted(_IC_REGISTRY.keys())}")
    return _IC_REGISTRY[name](**kwargs)


def stack_faces(cube: CubeGeometry, face_to_field: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
    """Stack face fields into shape (6, Ng, Ng) using FACE_ORDER."""
    Ng = cube.Ng
    out = np.empty((6, Ng, Ng), dtype=float)
    for k, fid in enumerate(FACE_ORDER):
        arr = face_to_field[fid]
        if arr.shape != (Ng, Ng):
            raise ValueError(f"Face {fid} has shape {arr.shape}, expected {(Ng, Ng)}")
        out[k] = arr
    return out