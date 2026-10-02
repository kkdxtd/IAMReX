# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Regression and quality tests for the Transolver weight backend."""

from __future__ import annotations

import unittest

import numpy as np

from unit_test.common import IAMREX_ROOT, build_small_fixture

from src.validation import compute_weight_moment_errors, weights_as_array
from src.weight_solver import MLWeightSolver, RKPMSolver, stencil_centers


class MLWeightSolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = IAMREX_ROOT / "Transolver_v2" / "out_inv"
        cls.model_code = IAMREX_ROOT / "Transolver_v2" / "transolver_slim.py"
        required = [
            cls.model_dir / "model_best.pt",
            cls.model_dir / "norm_stats.npz",
            cls.model_code,
        ]
        if not all(path.is_file() for path in required):
            raise unittest.SkipTest("Transolver model files are not available")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise unittest.SkipTest("PyTorch is not available") from exc

    def test_ml_output_is_finite_normalized_and_close_to_rkpm(self):
        grid, positions, _, lag_map = build_small_fixture()
        marker_ids = sorted(lag_map)
        ml_weights = MLWeightSolver(
            grid, self.model_dir, self.model_code
        ).solve(positions, lag_map)
        rkpm_weights = RKPMSolver(grid).solve(positions, lag_map)

        ml_array = weights_as_array(ml_weights, marker_ids)
        rkpm_array = weights_as_array(rkpm_weights, marker_ids)
        self.assertEqual(ml_array.shape, (3, 27))
        self.assertTrue(np.all(np.isfinite(ml_array)))
        np.testing.assert_allclose(
            ml_array.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-6
        )

        difference = ml_array - rkpm_array
        rmse = float(np.sqrt(np.mean(difference**2)))
        max_error = float(np.max(np.abs(difference)))
        self.assertLess(rmse, 2.0e-3)
        self.assertLess(max_error, 1.0e-2)

        centers = np.asarray(stencil_centers(lag_map, grid, marker_ids))
        errors = compute_weight_moment_errors(
            positions, centers, ml_array.astype(np.float32), grid.dx
        )
        self.assertLess(errors.moment_error, 1.0e-2)

    def test_ml_batching_preserves_marker_and_stencil_order(self):
        grid, positions, _, lag_map = build_small_fixture()
        solver = MLWeightSolver(
            grid, self.model_dir, self.model_code, batch_size=1
        )
        chunked_ids, chunked = solver.solve_array(positions, lag_map)
        solver.batch_size = len(positions)
        single_ids, single_batch = solver.solve_array(positions, lag_map)

        self.assertEqual(chunked_ids, single_ids)
        np.testing.assert_allclose(chunked, single_batch, rtol=1.0e-6, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
