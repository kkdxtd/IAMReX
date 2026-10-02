# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Real-case regression tests for the vectorized traditional RKPM solver."""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from unit_test.common import TEST_DIR

from src import mapping, window
from src.weight_solver import GridSpec, RKPMSolver, stencil_centers


REAL_CASE_DIR = TEST_DIR / "fixtures" / "real_case"
LEGACY_ROOT = TEST_DIR / "fixtures" / "RKPM_weight_commit_2cf74f4" / "src"
ARTIFACTS_DIR = TEST_DIR / "artifacts"
LEGACY_LAG_ARTIFACT = ARTIFACTS_DIR / "rkpm_mappings_commit_2cf74f4.lag"
VECTORIZED_LAG_ARTIFACT = ARTIFACTS_DIR / "rkpm_mappings_vectorized.lag"
FLOAT64_EPSILON = np.finfo(np.float64).eps
MOMENT_PRECISION_FACTOR = 64.0
WEIGHT_NORM_TOLERANCE = 1.0e-11


def _load_legacy_module(name: str, path: Path):
    """Load one module from the frozen pre-vectorization fixture."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy RKPM module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VectorizedRKPMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.legacy_window = _load_legacy_module(
            "rkpm_legacy_window_2cf74f4",
            LEGACY_ROOT / "window.py",
        )
        cls.legacy_mapping = _load_legacy_module(
            "rkpm_legacy_mapping_2cf74f4",
            LEGACY_ROOT / "mapping.py",
        )

        cls.inputs_path = REAL_CASE_DIR / "inputs.3d.flow_past_ellipsoid"
        cls.id_path = REAL_CASE_DIR / "rkpm_mappings.id"
        cls.lag_path = REAL_CASE_DIR / "rkpm_mappings.lag"
        cls.grid = GridSpec.from_inputs(cls.inputs_path)
        cls.id_map = mapping.load_id_map(cls.id_path)
        cls.lag_map = mapping.load_lag_map(cls.lag_path)
        cls.marker_ids = tuple(
            mapping.validate_mapping_ids(cls.id_map, cls.lag_map)
        )
        cls.positions = np.asarray(
            [cls.id_map[marker_id] for marker_id in cls.marker_ids],
            dtype=float,
        )

        centers = stencil_centers(cls.lag_map, cls.grid, cls.marker_ids)
        cell_volume = float(np.prod(cls.grid.dx))
        cls.supports = np.empty((*centers.shape[:2], 4), dtype=float)
        cls.supports[..., :3] = centers
        cls.supports[..., 3] = cell_volume
        cls.scales = cls.grid.dx * 1.001

        cls.lagrangian_volumes = np.empty(len(cls.marker_ids), dtype=float)
        for marker_index, marker_id in enumerate(cls.marker_ids):
            eps = np.asarray(
                [row["eps"] for row in cls.lag_map[marker_id]], dtype=float
            )
            np.testing.assert_array_equal(eps, np.full_like(eps, eps[0]))
            cls.lagrangian_volumes[marker_index] = eps[0] * cell_volume

        cls.legacy_moment_matrices = np.asarray(
            [
                cls.legacy_window.compute_m_ab_matrix(
                    cls.supports[index],
                    cls.positions[index],
                    cls.scales[0],
                    cls.scales[1],
                    cls.scales[2],
                    cls.lagrangian_volumes[index],
                )
                for index in range(len(cls.marker_ids))
            ]
        )

        # Capture the exact matrices assembled inside the production batched
        # path without duplicating its einsum expression in this test.
        correction_shape = (len(cls.marker_ids), window.POLYNOMIAL_SIZE)
        with mock.patch.object(
            window,
            "compute_b_I",
            return_value=np.zeros(correction_shape, dtype=float),
        ) as solve_mock:
            window.compute_modified_window_functions_batch(
                cls.supports,
                cls.positions,
                cls.scales[0],
                cls.scales[1],
                cls.scales[2],
                cls.lagrangian_volumes,
            )
        solve_mock.assert_called_once()
        cls.vectorized_moment_matrices = np.asarray(
            solve_mock.call_args.args[0], dtype=float
        ).copy()

    def _assert_moment_matrices_match(self):
        differences = np.max(
            np.abs(
                self.vectorized_moment_matrices
                - self.legacy_moment_matrices
            ),
            axis=(1, 2),
        )
        scales = np.maximum(
            1.0,
            np.max(np.abs(self.legacy_moment_matrices), axis=(1, 2)),
        )
        limits = MOMENT_PRECISION_FACTOR * FLOAT64_EPSILON * scales
        self.assertTrue(
            np.all(differences <= limits),
            msg=(
                "Vectorized moment matrices differ from commit 2cf74f4 "
                "beyond float64 machine-precision accumulation error: "
                f"max normalized error={np.max(differences / scales):.17e}, "
                f"limit={MOMENT_PRECISION_FACTOR * FLOAT64_EPSILON:.17e}"
            ),
        )
        return float(np.max(differences))

    def _assert_weights_match(self, vectorized_weights, legacy_weights):
        differences = np.max(
            np.abs(vectorized_weights - legacy_weights), axis=1
        )
        scales = np.max(np.abs(legacy_weights), axis=1)
        normalized = differences / scales
        self.assertTrue(
            np.all(normalized <= WEIGHT_NORM_TOLERANCE),
            msg=(
                "Vectorized weights differ from commit 2cf74f4 beyond the "
                "real-case regression tolerance: "
                f"max markerwise norm error={np.max(normalized):.17e}, "
                f"limit={WEIGHT_NORM_TOLERANCE:.17e}"
            ),
        )
        return float(np.max(differences)), float(np.max(normalized))

    def test_real_case_moment_matrices_match_legacy_commit_at_machine_precision(self):
        maximum_error = self._assert_moment_matrices_match()
        print(
            "[vectorized-rkpm] moment matrix maximum absolute error: "
            f"{maximum_error:.17e}",
            flush=True,
        )

    def test_real_case_weights_and_lag_output_match_legacy_commit(self):
        # Do not compare downstream weights unless their input moment matrices
        # have already passed the stricter machine-precision check.
        self._assert_moment_matrices_match()

        legacy_weights = np.asarray(
            [
                self.legacy_window.modified_window_function(
                    self.supports[index],
                    self.positions[index],
                    self.legacy_window.compute_b_I(
                        self.legacy_moment_matrices[index]
                    ),
                    self.scales[0],
                    self.scales[1],
                    self.scales[2],
                    self.lagrangian_volumes[index],
                )
                for index in range(len(self.marker_ids))
            ]
        )
        solved_ids, vectorized_weights = RKPMSolver(
            self.grid,
            batch_size=512,
        ).solve_array(self.positions, self.lag_map)

        self.assertEqual(tuple(solved_ids), self.marker_ids)
        self._assert_weights_match(vectorized_weights, legacy_weights)

        # The legacy fixture has one physical Lagrangian volume shared by all
        # markers. Assert that contract before calling its scalar mapping API.
        np.testing.assert_array_equal(
            self.lagrangian_volumes,
            np.full_like(
                self.lagrangian_volumes, self.lagrangian_volumes[0]
            ),
        )
        legacy_lag_map = self.legacy_mapping.build_lag_to_eul_map(
            self.positions,
            self.supports,
            legacy_weights,
            self.grid.prob_lo,
            self.grid.dx,
            self.lagrangian_volumes[0],
        )
        vectorized_lag_map = mapping.replace_mapping_weights(
            self.lag_map,
            {
                marker_id: vectorized_weights[index]
                for index, marker_id in enumerate(self.marker_ids)
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            legacy_prefix = Path(directory) / "legacy"
            vectorized_prefix = Path(directory) / "vectorized"
            self.legacy_mapping.save_mappings_txt(
                self.id_map, legacy_lag_map, str(legacy_prefix)
            )
            mapping.save_mappings_txt(
                self.id_map, vectorized_lag_map, str(vectorized_prefix)
            )
            legacy_lag_path = legacy_prefix.with_suffix(".lag")
            vectorized_lag_path = vectorized_prefix.with_suffix(".lag")

            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(legacy_lag_path, LEGACY_LAG_ARTIFACT)
            shutil.copyfile(vectorized_lag_path, VECTORIZED_LAG_ARTIFACT)

        serialized_legacy = mapping.load_lag_map(LEGACY_LAG_ARTIFACT)
        serialized_vectorized = mapping.load_lag_map(VECTORIZED_LAG_ARTIFACT)

        self.assertEqual(tuple(serialized_legacy), tuple(serialized_vectorized))
        serialized_legacy_weights = []
        serialized_vectorized_weights = []
        for marker_id in self.marker_ids:
            legacy_rows = serialized_legacy[marker_id]
            vectorized_rows = serialized_vectorized[marker_id]
            self.assertEqual(len(legacy_rows), len(vectorized_rows))
            for legacy_row, vectorized_row in zip(legacy_rows, vectorized_rows):
                self.assertEqual(
                    (legacy_row["i"], legacy_row["j"], legacy_row["k"]),
                    (
                        vectorized_row["i"],
                        vectorized_row["j"],
                        vectorized_row["k"],
                    ),
                )
                self.assertEqual(legacy_row["Vcell"], vectorized_row["Vcell"])
                self.assertEqual(legacy_row["eps"], vectorized_row["eps"])
                serialized_legacy_weights.append(legacy_row["w"])
                serialized_vectorized_weights.append(vectorized_row["w"])

        lag_maximum_error, lag_maximum_normalized_error = self._assert_weights_match(
            np.asarray(serialized_vectorized_weights).reshape(
                len(self.marker_ids), -1
            ),
            np.asarray(serialized_legacy_weights).reshape(
                len(self.marker_ids), -1
            ),
        )
        print(
            "[vectorized-rkpm] .lag maximum absolute weight error: "
            f"{lag_maximum_error:.17e}; maximum markerwise normalized error: "
            f"{lag_maximum_normalized_error:.17e}",
            flush=True,
        )
        print(
            f"[vectorized-rkpm] wrote {LEGACY_LAG_ARTIFACT}",
            flush=True,
        )
        print(
            f"[vectorized-rkpm] wrote {VECTORIZED_LAG_ARTIFACT}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
