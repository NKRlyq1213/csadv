csadv — Cubed-sphere advection (NumPy / Numba)

A small research/teaching codebase for scalar advection on a cubed-sphere grid (equiangular mapping), with:

CPU baseline (NumPy) and accelerated backend (Numba)

Pluggable initial conditions, velocity fields, and boundary coupling schemes

A simple CLI to run convergence experiments

This repository is designed to be extended: add new IC/boundary variants without rewriting the solver pipeline.

Features

Cubed-sphere equiangular geometry (geometry/)

GLL operators and SAT parameters (operators/)

Velocity fields registry (currently includes rigid rotation) (physics/)

Initial conditions registry (currently includes great-circle Gaussian) (initial_conditions/)

Boundary schemes registry (currently includes SAT inflow penalty) (boundary/)

Advection RHS with NumPy / Numba switch (rhs/)

LSRK5 time integrator (integrators/)

Convergence experiment runner (experiments/)

Tests (pytest) + CI-ready structure

Requirements

Python >= 3.10

NumPy

(Optional) Numba for acceleration

Installation
Create & activate a virtual environment (Windows PowerShell)

py -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install -U pip

Install (development mode)

NumPy-only:
pip install -e ".[dev]"

NumPy + Numba:
pip install -e ".[dev,numba]"

Quick start (CLI)

Run (NumPy backend):

csadv-conv --Ng 9 13 17 --R 1 --CFL 0.08 --u0 1 --alpha0 0.3 ^
--ic gaussian --lam0 0 --lat0 0 --sigma-m 0.2 ^
--boundary sat_inflow --bnd-backend numpy --rhs-backend numpy --periods 1

Run (Numba backend):

csadv-conv --Ng 9 13 17 --R 1 --CFL 0.08 --u0 1 --alpha0 0.3 ^
--ic gaussian --lam0 0 --lat0 0 --sigma-m 0.2 ^
--boundary sat_inflow --bnd-backend numba --rhs-backend numba --periods 1

Write results to JSON:

csadv-conv --Ng 9 13 17 --sigma-m 0.2 --periods 1 --json conv_result.json

Run via module:

python -m csadv --Ng 9 13 17 --sigma-m 0.2 --periods 1

Running tests

pytest -q

Extending the codebase
1) Add a new initial condition (IC)

Create: src/csadv/initial_conditions/my_ic.py

Example:

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from csadv.geometry.cubed_sphere import CubeGeometry
from csadv.initial_conditions.base import stack_faces, register_ic

@dataclass(frozen=True, slots=True)
class MyIC:
amp: float = 1.0

def __call__(self, cube: CubeGeometry) -> NDArray[np.floating]:  
    face_to_phi = {}  
    for fid, face in cube.faces.items():  
        face_to_phi[fid] = np.ones((cube.Ng, cube.Ng), dtype=float) * self.amp  
    return stack_faces(cube, face_to_phi)  

register_ic("my_ic", lambda **kw: MyIC(**kw))

2) Add a new boundary coupling scheme

Create: src/csadv/boundary/my_scheme.py

Example:

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

from csadv.geometry.cubed_sphere import CubeGeometry
from csadv.boundary.base import BoundaryScheme, register_boundary

@dataclass(frozen=True, slots=True)
class MyBoundary(BoundaryScheme):
def penalty(self, state: NDArray[np.floating], cube: CubeGeometry, u1: NDArray[np.floating], u2: NDArray[np.floating]) -> NDArray[np.floating]:
return np.zeros_like(state, dtype=float)

register_boundary("my_boundary", lambda **kw: MyBoundary(**kw))

3) Add a new velocity field

Add a new factory in physics/velocity_fields.py and register it via register_velocity_field(...).
Implement uv(lam, lat_or_colat) -> (u, v).

Project layout

src/csadv/
operators/ # GLL operators, SAT params
geometry/ # cubed-sphere mapping + transforms
physics/ # velocity fields registry
initial_conditions/ # IC registry (gaussian, etc.)
boundary/ # boundary/face coupling registry (SAT inflow, etc.)
rhs/ # advection RHS (numpy/numba)
integrators/ # LSRK integrator
experiments/ # convergence runner (no plotting)
cli.py # csadv-conv entrypoint
tests/ # pytest tests

Notes / current limitations

Ne is currently treated as a parameter but the face construction uses a single GLL grid per face (future work: multi-element per face).

The focus is correctness + extensibility first; performance improvements are mainly via Numba backend.

csadv — 立方球面對流（NumPy / Numba）

這是一個用於研究/教學的 cubed-sphere（equiangular 映射）標量對流程式，具備：

CPU 基準版（NumPy）與加速版（Numba）

可插拔的初始場（IC）、速度場、邊界/面耦合方案

提供簡單的 CLI 用來跑收斂測試

本專案設計重點是「可擴充」：你可以新增 IC/邊界方案版本，而不用重寫整個求解流程。

功能

cubed-sphere equiangular 幾何 (geometry/)

GLL 算子與 SAT 參數 (operators/)

速度場 registry（目前含剛體旋轉）(physics/)

初始場 registry（目前含大球距離 Gaussian）(initial_conditions/)

邊界方案 registry（目前含 SAT inflow penalty）(boundary/)

RHS 支援 NumPy / Numba 切換 (rhs/)

LSRK5 時間積分 (integrators/)

收斂測試 runner（不含畫圖）(experiments/)

pytest 測試、可直接接 CI

需求

Python >= 3.10

NumPy

（可選）Numba（加速）

安裝
建立與啟用虛擬環境（Windows PowerShell）

py -m venv .venv
..venv\Scripts\Activate.ps1
python -m pip install -U pip

安裝（開發模式）

只用 NumPy：
pip install -e ".[dev]"

NumPy + Numba：
pip install -e ".[dev,numba]"

快速開始（CLI）

NumPy 版本：

csadv-conv --Ng 9 13 17 --R 1 --CFL 0.08 --u0 1 --alpha0 0.3 ^
--ic gaussian --lam0 0 --lat0 0 --sigma-m 0.2 ^
--boundary sat_inflow --bnd-backend numpy --rhs-backend numpy --periods 1

Numba 版本：

csadv-conv --Ng 9 13 17 --R 1 --CFL 0.08 --u0 1 --alpha0 0.3 ^
--ic gaussian --lam0 0 --lat0 0 --sigma-m 0.2 ^
--boundary sat_inflow --bnd-backend numba --rhs-backend numba --periods 1

輸出 JSON：

csadv-conv --Ng 9 13 17 --sigma-m 0.2 --periods 1 --json conv_result.json

模組方式執行：

python -m csadv --Ng 9 13 17 --sigma-m 0.2 --periods 1

執行測試

pytest -q

如何擴充（新增版本）
1) 新增初始場（IC）

建立檔案 src/csadv/initial_conditions/my_ic.py，實作 __call__(cube) 回傳 (6, Ng, Ng)，並 register_ic("my_ic", ...) 註冊即可。

2) 新增邊界/面耦合方案

建立 src/csadv/boundary/my_scheme.py，實作 penalty(state, cube, u1, u2) 回傳 (6, Ng, Ng)，並 register_boundary("my_boundary", ...) 註冊即可。

3) 新增速度場

在 physics/velocity_fields.py 新增一個 factory，提供 uv(lam, lat_or_colat) -> (u, v)，再註冊即可。

專案結構

src/csadv/
operators/ # GLL 算子、SAT 參數
geometry/ # cubed-sphere 幾何與轉換
physics/ # 速度場 registry
initial_conditions/ # 初始場 registry
boundary/ # 邊界/面耦合 registry
rhs/ # RHS（numpy/numba）
integrators/ # LSRK 積分器
experiments/ # 收斂測試（不畫圖）
cli.py # csadv-conv 入口
tests/ # pytest 測試

備註 / 目前限制

Ne 目前保留為參數，但面內網格目前是每個 face 一張 GLL 網格（未做多元素切分；後續可擴充）。

目前以正確性＋可擴充為主；效能主要靠 Numba backend 提升。
