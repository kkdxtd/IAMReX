# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Acceptance test for the bundled full RKPM mapping fixture."""

from __future__ import annotations

import unittest

import numpy as np

from unit_test.verify_rkpm_alignment import (
    DEFAULT_ID_FILE,
    DEFAULT_INPUTS,
    DEFAULT_LAG_FILE,
    TOL_MOMENT_CELLS,
    TOL_SECOND_MOMENT_CELLS2,
    TOL_SUMW,
    compute_reproduction_residuals,
    grid_from_inputs,
    load_id,
    load_lag,
)


class RealMappingAlignmentTests(unittest.TestCase):
    def test_real_mapping_reproduces_quadratic_basis(self):
        prob_lo, dx = grid_from_inputs(DEFAULT_INPUTS)
        points = load_id(DEFAULT_ID_FILE)
        lag = load_lag(DEFAULT_LAG_FILE)
        sumw, first, second = compute_reproduction_residuals(
            points, lag, prob_lo, dx
        )

        self.assertLess(float(np.max(np.abs(sumw - 1.0))), TOL_SUMW)
        self.assertLess(float(np.max(np.abs(first))), TOL_MOMENT_CELLS)
        self.assertLess(
            float(np.max(np.abs(second))), TOL_SECOND_MOMENT_CELLS2
        )


if __name__ == "__main__":
    unittest.main()
