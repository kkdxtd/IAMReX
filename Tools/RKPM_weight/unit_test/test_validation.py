# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for zeroth- and first-moment validation."""

from __future__ import annotations

import unittest

import numpy as np

from unit_test.common import build_small_fixture

from src.validation import compute_weight_moment_errors
from src.weight_solver import stencil_centers


class MomentValidationTests(unittest.TestCase):
    def setUp(self):
        grid, positions, _, lag_map = build_small_fixture()
        self.grid = grid
        self.position = positions[0:1]
        self.centers = np.asarray(stencil_centers(lag_map, grid, [0]))
        distances = np.linalg.norm(
            self.centers[0] - self.position[0], axis=1
        )
        self.center_index = int(np.argmin(distances))
        self.weights = np.zeros((1, 27))
        self.weights[0, self.center_index] = 1.0

    def test_moment_check_accepts_valid_weights(self):
        errors = compute_weight_moment_errors(
            self.position, self.centers, self.weights, self.grid.dx
        )
        self.assertTrue(errors.passes(1.0e-12, 1.0e-12))

    def test_moment_check_distinguishes_sum_and_first_moment_errors(self):
        bad_sum = self.weights.copy()
        bad_sum[0, self.center_index] += 0.01
        sum_errors = compute_weight_moment_errors(
            self.position, self.centers, bad_sum, self.grid.dx
        )
        self.assertAlmostEqual(sum_errors.sum_error, 0.01)

        center_ijk = np.asarray([
            4,
            5,
            6,
        ])
        neighbor_ijk = center_ijk + np.asarray([1, 0, 0])
        neighbor_index = next(
            index
            for index, center in enumerate(self.centers[0])
            if np.allclose(
                center,
                self.grid.prob_lo + (neighbor_ijk + 0.5) * self.grid.dx,
            )
        )
        bad_first = self.weights.copy()
        bad_first[0, self.center_index] -= 0.01
        bad_first[0, neighbor_index] += 0.01
        first_errors = compute_weight_moment_errors(
            self.position, self.centers, bad_first, self.grid.dx
        )
        self.assertLess(first_errors.sum_error, 1.0e-14)
        self.assertAlmostEqual(first_errors.moment_error, 0.01)


if __name__ == "__main__":
    unittest.main()
