from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from csadv.geometry.cubed_sphere import CubeGeometry


class BoundaryScheme(Protocol):
    """Boundary/face-coupling scheme interface.

    Return penalty term (same shape as state): (6, Ng, Ng).
    """
    def penalty(
        self,
        state: NDArray[np.floating],
        cube: CubeGeometry,
        u1: NDArray[np.floating],
        u2: NDArray[np.floating],
    ) -> NDArray[np.floating]: ...


_BND_REGISTRY: dict[str, Callable[..., BoundaryScheme]] = {}


def register_boundary(name: str, factory: Callable[..., BoundaryScheme]) -> None:
    if not name or not isinstance(name, str):
        raise ValueError("Boundary name must be a non-empty string.")
    _BND_REGISTRY[name] = factory


def get_boundary(name: str, **kwargs) -> BoundaryScheme:
    if name not in _BND_REGISTRY:
        raise KeyError(f"Unknown boundary '{name}'. Available: {sorted(_BND_REGISTRY.keys())}")
    return _BND_REGISTRY[name](**kwargs)