# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Small deterministic fixtures shared by RKPM weight tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
TOOL_ROOT = TEST_DIR.parent
IAMREX_ROOT = TOOL_ROOT.parents[1]
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from src.weight_solver import GridSpec  # noqa: E402


def build_small_fixture():
    """Return a nonzero-origin, anisotropic grid with three 27-point stencils."""
    prob_lo = np.asarray([0.25, -0.75, 1.5])
    dx = np.asarray([0.2, 0.3, 0.4])
    n_cell = np.asarray([24, 20, 18])
    prob_hi = prob_lo + n_cell * dx
    grid = GridSpec(
        prob_lo=prob_lo,
        prob_hi=prob_hi,
        n_cell=n_cell,
        max_level=0,
        dx=dx,
    )

    containing_cells = np.asarray([
        [4, 5, 6],
        [8, 7, 5],
        [12, 10, 9],
    ])
    cell_fractions = np.asarray([
        [0.50, 0.50, 0.50],
        [0.22, 0.67, 0.41],
        [0.78, 0.31, 0.84],
    ])
    positions = prob_lo + (containing_cells + cell_fractions) * dx
    id_map = {
        marker_id: tuple(position)
        for marker_id, position in enumerate(positions)
    }

    cell_volume = float(np.prod(dx))
    lag_map = {}
    for marker_id, center_cell in enumerate(containing_cells):
        lagrangian_volume = cell_volume * (0.35 + 0.05 * marker_id)
        rows = []
        for k in range(center_cell[2] - 1, center_cell[2] + 2):
            for j in range(center_cell[1] - 1, center_cell[1] + 2):
                for i in range(center_cell[0] - 1, center_cell[0] + 2):
                    rows.append({
                        "i": int(i),
                        "j": int(j),
                        "k": int(k),
                        "w": 0.0,
                        "Vcell": 1.0,
                        "eps": lagrangian_volume / cell_volume,
                    })
        lag_map[marker_id] = rows

    return grid, positions, id_map, lag_map
