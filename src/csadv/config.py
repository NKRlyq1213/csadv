from __future__ import annotations

# Canonical face order used everywhere
FACE_ORDER = ["P1", "P2", "P3", "P4", "P5", "P6"]
FACE_IDX = {name: i for i, name in enumerate(FACE_ORDER)}

# Side index convention (matches your notebook)
# 0: West  (i=0)   normal = -e1
# 1: East  (i=N)   normal = +e1
# 2: South (j=0)   normal = -e2
# 3: North (j=N)   normal = +e2