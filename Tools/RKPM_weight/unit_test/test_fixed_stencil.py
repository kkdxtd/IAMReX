# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for fixed 3x3x3 MPMD stencil validation."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from unit_test.common import build_small_fixture

from src.weight_solver import validate_fixed_stencils, validate_stencils


class FixedStencilTests(unittest.TestCase):
    def test_fixed_stencil_accepts_valid_markers_and_rejects_invalid_state(self):
        grid, positions, _, lag_map = build_small_fixture()
        marker_ids = validate_stencils(lag_map, expected_size=27)
        validate_fixed_stencils(positions, lag_map, grid, marker_ids)

        crossed = positions.copy()
        crossed[1, 0] += grid.dx[0]
        with self.assertRaisesRegex(ValueError, "left the center cell"):
            validate_fixed_stencils(crossed, lag_map, grid, marker_ids)

        incomplete = copy.deepcopy(lag_map)
        incomplete[2].pop()
        with self.assertRaisesRegex(ValueError, "not a complete 3x3x3 stencil"):
            validate_fixed_stencils(positions, incomplete, grid, marker_ids)

    def test_grid_line_roundoff_snaps_to_high_index_cell(self):
        grid, positions, _, lag_map = build_small_fixture()
        marker_ids = validate_stencils(lag_map, expected_size=27)

        expected_center = np.asarray([4, 5, 6])
        grid_line = grid.prob_lo + expected_center * grid.dx

        on_grid_line = positions.copy()
        on_grid_line[0] = grid_line
        validate_fixed_stencils(on_grid_line, lag_map, grid, marker_ids)

        roundoff_below = on_grid_line.copy()
        roundoff_below[0] = np.nextafter(grid_line, -np.inf)
        validate_fixed_stencils(roundoff_below, lag_map, grid, marker_ids)

        genuinely_left = on_grid_line.copy()
        genuinely_left[0, 0] -= 1.0e-8 * grid.dx[0]
        with self.assertRaisesRegex(ValueError, "left the center cell"):
            validate_fixed_stencils(genuinely_left, lag_map, grid, marker_ids)


if __name__ == "__main__":
    unittest.main()
