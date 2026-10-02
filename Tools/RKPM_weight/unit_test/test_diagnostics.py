# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for vectorized RKPM volume, force and torque diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from unit_test.common import build_small_fixture

from src import error
from src.weight_solver import RKPMSolver, stencil_centers


class VectorizedDiagnosticTests(unittest.TestCase):
    def test_rkpm_weights_conserve_volume_force_and_torque(self):
        grid, positions, _, lag_map = build_small_fixture()
        marker_ids, weights = RKPMSolver(grid, batch_size=2).solve_array(
            positions, lag_map
        )
        centers = stencil_centers(lag_map, grid, marker_ids)
        cell_volume = float(np.prod(grid.dx))
        supports = [
            np.column_stack(
                (marker_centers, np.full(len(marker_centers), cell_volume))
            )
            for marker_centers in centers
        ]
        eulerian_points = np.unique(centers.reshape(-1, 3), axis=0)
        lagrangian_volume = lag_map[0][0]["eps"] * cell_volume

        volume_errors = error.compute_error_volume(
            positions, supports, weights, lagrangian_volume
        )
        force_error, torque_error = error.compute_conservation_check(
            positions.mean(axis=0),
            eulerian_points,
            positions,
            supports,
            weights,
            lagrangian_volume,
        )
        self.assertLess(float(np.max(volume_errors)), 1.0e-10)
        self.assertLess(force_error, 1.0e-10)
        self.assertLess(torque_error, 1.0e-10)


if __name__ == "__main__":
    unittest.main()
