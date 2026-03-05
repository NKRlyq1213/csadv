from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SatParams:
    """SAT penalty parameters for 1D GLL operators (used on each element edge)."""

    s_scale: float
    tau_0: float
    tau_N: float
    w_gll: NDArray[np.floating]