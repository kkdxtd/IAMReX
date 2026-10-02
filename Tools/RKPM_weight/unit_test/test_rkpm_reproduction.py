# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for traditional RKPM polynomial reproduction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from unit_test.common import build_small_fixture

from src import mapping
from src.weight_solver import RKPMSolver
from unit_test.verify_rkpm_alignment import (
    compute_reproduction_residuals,
    load_id,
    load_lag,
)


class RKPMReproductionTests(unittest.TestCase):
    def test_rkpm_reproduces_quadratic_basis(self):
        grid, _, id_map, lag_map = build_small_fixture()
        weights = RKPMSolver(grid).solve(id_map, lag_map)
        solved_lag = mapping.replace_mapping_weights(lag_map, weights)

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "tiny_mapping"
            mapping.save_mappings_txt(id_map, solved_lag, str(prefix))
            points = load_id(prefix.with_suffix(".id"))
            verifier_lag = load_lag(prefix.with_suffix(".lag"))

        sumw, first, second = compute_reproduction_residuals(
            points, verifier_lag, grid.prob_lo, grid.dx
        )
        np.testing.assert_allclose(sumw, 1.0, rtol=0.0, atol=1.0e-10)
        np.testing.assert_allclose(first, 0.0, rtol=0.0, atol=1.0e-10)
        np.testing.assert_allclose(second, 0.0, rtol=0.0, atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
