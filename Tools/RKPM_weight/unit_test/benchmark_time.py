# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the scalar and vectorized RKPM core calculation times."""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
TOOL_ROOT = TEST_DIR.parent
REAL_CASE_DIR = TEST_DIR / "fixtures" / "real_case"
LEGACY_WINDOW_PATH = (
    TEST_DIR
    / "fixtures"
    / "RKPM_weight_commit_2cf74f4"
    / "src"
    / "window.py"
)

if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

from src import mapping, window  # noqa: E402
from src.weight_solver import GridSpec, stencil_centers  # noqa: E402


def load_legacy_window():
    """Load the scalar window implementation from commit 2cf74f4."""
    spec = importlib.util.spec_from_file_location(
        "rkpm_benchmark_legacy_window",
        LEGACY_WINDOW_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load legacy RKPM code: {LEGACY_WINDOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(function, repeat=3):
    """Run a calculation repeatedly and return its result and median time."""
    elapsed = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        elapsed.append(time.perf_counter() - start)

    return result, statistics.median(elapsed)


def main():
    legacy_window = load_legacy_window()

    grid = GridSpec.from_inputs(
        REAL_CASE_DIR / "inputs.3d.flow_past_ellipsoid"
    )
    id_map = mapping.load_id_map(REAL_CASE_DIR / "rkpm_mappings.id")
    lag_map = mapping.load_lag_map(REAL_CASE_DIR / "rkpm_mappings.lag")
    marker_ids = tuple(mapping.validate_mapping_ids(id_map, lag_map))
    positions = np.asarray([id_map[marker_id] for marker_id in marker_ids])

    centers = stencil_centers(lag_map, grid, marker_ids)
    cell_volume = float(np.prod(grid.dx))
    supports = np.empty((*centers.shape[:2], 4), dtype=float)
    supports[..., :3] = centers
    supports[..., 3] = cell_volume

    lagrangian_volumes = np.asarray(
        [lag_map[marker_id][0]["eps"] * cell_volume for marker_id in marker_ids]
    )
    np.testing.assert_array_equal(
        lagrangian_volumes,
        np.full_like(lagrangian_volumes, lagrangian_volumes[0]),
    )

    scales = grid.dx * 1.001
    marker_count = len(marker_ids)
    delta = np.full(marker_count, scales[0])
    eta = np.full(marker_count, scales[1])
    theta = np.full(marker_count, scales[2])

    def run_legacy():
        return legacy_window.compute_all_modified_window_functions(
            supports,
            positions,
            delta,
            eta,
            theta,
            lagrangian_volumes[0],
        )

    def run_vectorized():
        return window.compute_modified_window_functions_batch(
            supports,
            positions,
            scales[0],
            scales[1],
            scales[2],
            lagrangian_volumes,
        )

    repeat = 3
    legacy_result, legacy_time = measure(run_legacy, repeat=repeat)
    vectorized_weights, vectorized_time = measure(
        run_vectorized, repeat=repeat
    )
    legacy_weights = np.asarray(legacy_result)

    np.testing.assert_allclose(
        vectorized_weights,
        legacy_weights,
        rtol=1.0e-11,
        atol=1.0e-12,
    )

    maximum_error = float(np.max(np.abs(vectorized_weights - legacy_weights)))
    speedup = legacy_time / vectorized_time
    print(f"Markers:               {marker_count}")
    print(f"Repeated runs:         {repeat} (median reported)")
    print(f"Legacy RKPM time:      {legacy_time:.6f} s")
    print(f"Vectorized RKPM time:  {vectorized_time:.6f} s")
    print(f"Speedup:               {speedup:.2f}x")
    print(f"Maximum weight error:  {maximum_error:.17e}")


if __name__ == "__main__":
    main()
