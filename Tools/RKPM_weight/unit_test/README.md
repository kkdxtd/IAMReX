<!--
SPDX-FileCopyrightText: 2026 IAMReX contributors
SPDX-License-Identifier: BSD-3-Clause
-->

# RKPM Weight Tests

Run the complete suite from any working directory:

```bash
/path/to/IAMReX/Tools/RKPM_weight/unit_test/run_all_tests.sh
```

The script uses `python3` by default. To use a particular virtual environment
without relying on its activation state, provide the path to its interpreter:

```bash
RKPM_TEST_PYTHON=/path/to/python \
  /path/to/IAMReX/Tools/RKPM_weight/unit_test/run_all_tests.sh
```

NumPy is required for the core suite. The ML test is skipped if PyTorch or the
model files are unavailable. The MPMD test is skipped if either `mpiexec` or
`mpi4py` is unavailable. `mpi4py` and `mpiexec` must use compatible MPI
implementations for that test.

Most tests use a three-marker synthetic fixture with a nonzero domain origin,
anisotropic cell spacing and complete 3x3x3 stencils. The full alignment and
pre-/post-vectorization regressions additionally use the bundled 2830-marker
files under `fixtures/real_case/`.

## Test Details

Twelve `test_*.py` files currently define nineteen `unittest` test methods.

### `test_architecture_doc.py`

#### `ArchitectureDocumentTests.test_architecture_document_lists_every_production_module`

- **What it tests:** The maintained architecture document does not silently
  omit a production Python module.
- **How it tests it:** It discovers `main.py` and every `src/*.py` file, then
  requires each repository-relative module path to appear in
  `ARCHITECTURE.md`.

### `test_rkpm_reproduction.py`

#### `RKPMReproductionTests.test_rkpm_reproduces_quadratic_basis`

- **What it tests:** The traditional RKPM solver reproduces the constant,
  linear and quadratic polynomial bases on a nonzero-origin, anisotropic grid.
- **How it tests it:** It solves three synthetic 3x3x3 marker stencils, writes
  and reloads temporary `.id/.lag` files, and evaluates them with
  `compute_reproduction_residuals()`. Zeroth-, first- and second-order
  residuals must all be within `1e-10`.

### `test_real_mapping_alignment.py`

#### `RealMappingAlignmentTests.test_real_mapping_reproduces_quadratic_basis`

- **What it tests:** The complete bundled 2830-marker mapping remains aligned
  with its AMReX grid and satisfies constant, linear and quadratic reproduction.
- **How it tests it:** It reads the real inputs, `.id` and `.lag` files,
  reconstructs cell centers using the actual domain origin and separate
  directional grid spacings, and computes the reproduction residuals. Their
  maximum errors must be below `1e-9`, `1e-6` cell and `1e-6` cell squared,
  respectively.

### `test_validation.py`

#### `MomentValidationTests.test_moment_check_accepts_valid_weights`

- **What it tests:** A valid normalized weight set passes both the zeroth- and
  first-moment checks.
- **How it tests it:** It places unit weight at a cell center coincident with
  the marker, computes the two errors, and requires both to pass `1e-12`
  tolerances.

#### `MomentValidationTests.test_moment_check_distinguishes_sum_and_first_moment_errors`

- **What it tests:** The validator distinguishes a normalization error from a
  pure first-moment error.
- **How it tests it:** It first adds `0.01` to one weight and expects a `0.01`
  sum error. It then transfers `0.01` from the center to the positive-x neighbor;
  the sum must remain correct while the first-moment error becomes `0.01` cell.

### `test_mapping.py`

#### `MappingTests.test_mapping_round_trip_preserves_order_and_metadata`

- **What it tests:** Weight replacement and `.id/.lag` serialization preserve
  marker ordering, stencil metadata and values without mutating the input map.
- **How it tests it:** It inserts known marker-dependent weights, confirms the
  original weights remain zero, saves to a temporary directory, reloads both
  files, and compares marker IDs, row order, `i/j/k`, `Vcell`, `eps` and every
  weight value.

### `test_fixed_stencil.py`

#### `FixedStencilTests.test_fixed_stencil_accepts_valid_markers_and_rejects_invalid_state`

- **What it tests:** The fixed-stencil guard accepts a valid stencil and rejects
  states that the current weight-only MPMD protocol cannot represent safely.
- **How it tests it:** A valid 3x3x3 stencil must pass. Moving one marker by one
  x-direction cell width must raise the center-cell-crossing error, and removing
  one of the 27 rows must raise the incomplete-stencil error.

#### `FixedStencilTests.test_grid_line_roundoff_snaps_to_high_index_cell`

- **What it tests:** A marker on an internal grid line follows the shared
  half-open-cell convention and remains in the high-index cell when floating-
  point roundoff places it infinitesimally below that line.
- **How it tests it:** It checks the exact grid-line coordinate and the next
  representable double below it, then verifies that a marker genuinely farther
  into the low-index cell is still rejected.

### `test_mpmd_transport.py`

#### `MPMDTransportTests.test_mpmd_preserves_double_positions_and_float32_weight_order`

- **What it tests:** The Python MPMD loop receives double-precision marker
  positions, preserves the float32 weight values and marker/stencil ordering
  across repeated exchanges, and shuts down cleanly.
- **How it tests it:** It starts two real MPI application contexts.
  `mpmd_client.py` emulates the C++ root rank, while `mpmd_server.py` runs the
  production `serve_mpmd()` loop and its direct array-output fast path with
  deterministic weights that encode the exchange, marker, stencil position and
  received x-coordinate. One marker is placed exactly on a grid line to
  exercise the double-plus-snap path. The client performs two exact array
  comparisons, sends a zero marker count, and the parent process enforces a
  30-second deadlock timeout.

This test verifies the Python server and the agreed wire protocol; it does not
compile or execute the IAMReX C++ implementation itself.

### `test_ml_solver.py`

#### `MLWeightSolverTests.test_ml_output_is_finite_normalized_and_close_to_rkpm`

- **What it tests:** The Transolver backend returns structurally valid,
  normalized weights whose values and first moment remain reasonably close to
  the traditional RKPM reference.
- **How it tests it:** It runs both solvers on the same three-marker fixture,
  checks shape `(3, 27)`, finiteness and normalization within `1e-6`, and then
  requires RMSE below `2e-3`, maximum weight error below `1e-2`, and maximum
  first-moment error below `1e-2` cell. It skips when PyTorch or model files are
  unavailable.

#### `MLWeightSolverTests.test_ml_batching_preserves_marker_and_stencil_order`

- **What it tests:** Splitting Transolver inference into bounded batches does
  not change marker order, stencil order or predicted weights beyond normal
  floating-point roundoff.
- **How it tests it:** It evaluates the same markers first one at a time and
  then in one batch with the same loaded model, compares the returned marker IDs
  and checks all weights with tight `1e-6`/`1e-7` tolerances.

### `test_vectorized_rkpm.py`

#### `VectorizedRKPMTests.test_real_case_moment_matrices_match_legacy_commit_at_machine_precision`

- **What it tests:** The vectorized RKPM moment matrices preserve the numerical
  result of commit `2cf74f4` at float64 machine-precision scale on the complete
  2830-marker real case.
- **How it tests it:** It executes the frozen scalar `window.py` from
  `fixtures/RKPM_weight_commit_2cf74f4/`, captures the moment matrices assembled
  inside the current production batch path, and compares their markerwise
  maximum absolute errors with a tolerance proportional to machine epsilon. A
  successful run also prints the maximum absolute matrix error.

#### `VectorizedRKPMTests.test_real_case_weights_and_lag_output_match_legacy_commit`

- **What it tests:** The current solver preserves the legacy corrected weights
  and final `.lag` mapping, including marker order, stencil order, `i/j/k`,
  `Vcell` and `eps`, on the same real case.
- **How it tests it:** It solves all markers with both the frozen scalar code and
  the current batched solver, first requiring the moment-matrix comparison to
  pass. It then regenerates both `.lag` files in a temporary directory and
  compares their parsed marker order, stencil rows and metadata exactly, with a
  tight markerwise norm tolerance for weights. Text bytes are not compared
  because floating-point accumulation and linear-algebra libraries can alter
  only the final printed digits across implementations and platforms. A
  successful run prints the maximum absolute `.lag` weight error and maximum
  markerwise normalized error, and writes both mappings to
  `artifacts/rkpm_mappings_commit_2cf74f4.lag` and
  `artifacts/rkpm_mappings_vectorized.lag` for manual inspection.

### `test_support_generation.py`

#### `SupportGenerationTests.test_generated_supports_are_complete_and_use_expected_cells`

- **What it tests:** Vectorized point-cloud support generation creates the
  expected 3x3x3 cells on a nonzero-origin, anisotropic grid and reports exactly
  the union of affected Eulerian cells.
- **How it tests it:** It writes a temporary four-marker geometry, disables only
  plotting, reconstructs all support indices and verifies the 27 offsets,
  nearest cell centers, cell volumes and affected-cell union.

### `test_diagnostics.py`

#### `VectorizedDiagnosticTests.test_rkpm_weights_conserve_volume_force_and_torque`

- **What it tests:** The vectorized post-solve diagnostics preserve the volume,
  total-force and total-torque checks used by file-mode generation.
- **How it tests it:** It spreads an asymmetric vector force through RKPM
  weights on overlapping synthetic stencils, accumulates contributions with
  array scatter operations and requires all three conservation residuals to be
  below `1e-10`.

### `test_runtime_config.py`

#### `RuntimeConfigTests.test_config_loads_batch_size_and_device`

- **What it tests:** The persistent run configuration recognizes the new batch
  size and ML device controls.
- **How it tests it:** It parses the bundled `inputs.rkpm` and compares both
  values with the documented defaults in that file.

#### `RuntimeConfigTests.test_command_line_overrides_batch_size_and_device`

- **What it tests:** Explicit command-line tuning still takes precedence over
  values loaded from the configuration file.
- **How it tests it:** It parses the same configuration with different
  `--batch-size` and `--device` arguments and verifies the overridden values.

#### `RuntimeConfigTests.test_geometry_generation_requires_explicit_coordinate_frame`

- **What it tests:** New `.id/.lag` generation cannot silently guess whether a
  point cloud uses body or world coordinates.
- **How it tests it:** It requests file-mode generation with a geometry path but
  without either frame option and checks that argument parsing exits with the
  dedicated coordinate-frame error.

#### `RuntimeConfigTests.test_geometry_generation_accepts_body_or_world_frame`

- **What it tests:** Both supported coordinate systems remain available when
  they are selected explicitly.
- **How it tests it:** It parses otherwise identical geometry-generation
  commands with `--body-frame` and `--world-frame`, then verifies their distinct
  Boolean values.

## Supporting Files

The following Python files support the tests but are not discovered as
independent tests because their names do not start with `test_`:

- `common.py` constructs the shared three-marker fixture, including a nonzero
  domain origin, anisotropic grid spacing and complete 3x3x3 stencils.
- `mpmd_client.py` acts as the IAMReX-side protocol emulator used only by the
  MPMD transport test.
- `mpmd_server.py` starts the production transport loop with the deterministic
  solver used only by that test.
- `verify_rkpm_alignment.py` provides the reproduction-residual calculations
  reused by two tests and can also be run as a standalone full-mapping checker.
- `benchmark_time.py` compares the scalar implementation from commit `2cf74f4`
  with the current vectorized core calculation on the 2830-marker real case. It
  reports the median runtime over three runs, speedup and maximum weight error;
  it is a manual benchmark rather than an automatically discovered unit test.
- `fixtures/RKPM_weight_commit_2cf74f4/` is a frozen copy of the scalar RKPM
  implementation used as the independent pre-vectorization reference.

The full mapping verifier remains directly executable:

```bash
cd /path/to/IAMReX/Tools/RKPM_weight
python3 unit_test/verify_rkpm_alignment.py
```

Run the vectorization benchmark separately:

```bash
cd /path/to/IAMReX/Tools/RKPM_weight
python3 unit_test/benchmark_time.py
```
