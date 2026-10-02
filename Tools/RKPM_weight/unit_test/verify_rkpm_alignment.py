#!/usr/bin/env python3
"""
Acceptance test for RKPM mapping files (rkpm_mappings.id / rkpm_mappings.lag).

It checks the discrete reproduction conditions that the diffused-IB method
relies on, evaluated IN THE SOLVER'S grid frame (cell centers at
prob_lo + (n+0.5)*dx, per direction). These are the quantities that actually
govern force and torque conservation when the solver applies the weights:

    (0th moment)  Sum_c w_c            == 1     -> conserves total force
    (1st moment)  Sum_c w_c (x_c - x_l) == 0     -> conserves torque (no spurious
                                                    force dipole / torque)

It also checks the six quadratic basis terms used by the current RKPM moment
matrix: xy, yz, zx, x^2, y^2 and z^2. These terms are not required for the two
conservation statements above, but they verify the solver's second-order
polynomial reproduction.

The grid is ANISOTROPIC-AWARE: dx, dy, dz are independent, and prob_lo need not
be the origin. Passing a single scalar dx to a run whose grid has dx != dy != dz
is itself a (very loud) failure mode -- the cell centers get reconstructed at the
wrong physical positions and the 1st moment explodes to O(100) cells in the
mismatched directions.

The classic *real* failure mode is a constant nonzero 1st moment caused by the
RKPM generator building its Euler grid on an origin that is not an integer
multiple of dx (int(sx/dx) truncation in mapping.py). That shows up here as a
per-marker 1st moment that is (a) nonzero and (b) IDENTICAL across all markers.
After the grid-alignment fix it should drop to ~1e-12 cells.

Usage:
    # preferred: use the bundled real-case fixture with no arguments
    python unit_test/verify_rkpm_alignment.py

    # or provide custom mapping and grid files
    python unit_test/verify_rkpm_alignment.py custom.id custom.lag \
        --inputs custom_inputs.3d

    # legacy positional form still works for isotropic grids
    python verify_rkpm_alignment.py [dx] [id_file] [lag_file]

Exit code 0 if PASS, 1 if FAIL.
"""

import sys
import re
import ast
import argparse
from pathlib import Path

import numpy as np

TOL_SUMW = 1e-9          # |Sum w - 1|
TOL_MOMENT_CELLS = 1e-6  # |Sum w (x_c - x_l)| / dx  (cells, per direction)
TOL_SECOND_MOMENT_CELLS2 = 1e-6

MAX_LEVEL_KEY = "amr.max_level"
SCRIPT_DIR = Path(__file__).resolve().parent
REAL_CASE_DIR = SCRIPT_DIR / "fixtures" / "real_case"
DEFAULT_INPUTS = REAL_CASE_DIR / "inputs.3d.flow_past_ellipsoid"
DEFAULT_ID_FILE = REAL_CASE_DIR / "rkpm_mappings.id"
DEFAULT_LAG_FILE = REAL_CASE_DIR / "rkpm_mappings.lag"


# --- AMReX inputs parsing (same conventions as main.py) ---------------------
def parse_inputs(path):
    params = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or '=' not in line:
                continue
            key, _, val = line.partition('=')
            params[key.strip()] = val.strip()
    return params


def grid_from_inputs(path):
    """Return (prob_lo, dx_finest) as (3,) arrays, matching main.py."""
    p = parse_inputs(path)
    prob_lo = np.array([float(x) for x in p["geometry.prob_lo"].split()])
    prob_hi = np.array([float(x) for x in p["geometry.prob_hi"].split()])
    n_cell = np.array([float(x) for x in p["amr.n_cell"].split()])
    max_level = int(p[MAX_LEVEL_KEY].split()[0])
    dx_finest = (prob_hi - prob_lo) / (n_cell * (2 ** max_level))
    return prob_lo, dx_finest


def load_id(path):
    with open(path) as f:
        d = ast.literal_eval(f.read())
    n = len(d)
    pts = np.zeros((n, 3))
    for k, v in d.items():
        pts[int(k)] = v
    return pts


def load_lag(path):
    with open(path, encoding="utf-8") as stream:
        txt = stream.read()
    blocks = re.findall(r'(\d+)\s*:\s*\[(.*?)\]', txt, re.S)
    out = {}
    for bid, body in blocks:
        entries = re.findall(r'\{([^}]*)\}', body)
        rows = []
        for e in entries:
            kv = dict(re.findall(r'"(\w+)"\s*:\s*([-\d.eE]+)', e))
            rows.append((int(kv['i']), int(kv['j']), int(kv['k']),
                         float(kv['w']), float(kv['Vcell']), float(kv['eps'])))
        out[int(bid)] = np.array(rows, dtype=float)
    return out


def compute_reproduction_residuals(pts, lag, prob_lo, dx):
    """Return zeroth-, first- and second-order residuals for every marker.

    First-order residuals are normalized by one grid spacing in the matching
    direction. Second-order residuals are normalized by the corresponding
    products of grid spacings, so they are expressed in cell-coordinate units.
    The second-order columns are ordered as ``xy, yz, zx, x2, y2, z2``.
    """
    marker_ids = sorted(lag)
    sumw = np.zeros(len(marker_ids))
    first = np.zeros((len(marker_ids), 3))
    second = np.zeros((len(marker_ids), 6))

    for row, marker_id in enumerate(marker_ids):
        stencil = lag[marker_id]
        ijk = stencil[:, 0:3]
        weights = stencil[:, 3]
        centers = prob_lo + (ijk + 0.5) * dx
        relative = (centers - pts[marker_id]) / dx
        x, y, z = relative.T

        sumw[row] = weights.sum()
        first[row] = (weights[:, None] * relative).sum(axis=0)
        quadratic_basis = np.column_stack((
            x * y,
            y * z,
            z * x,
            x**2,
            y**2,
            z**2,
        ))
        second[row] = (weights[:, None] * quadratic_basis).sum(axis=0)

    return sumw, first, second


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Verify RKPM mapping files against the solver grid.")
    ap.add_argument("id_file", nargs="?", default=DEFAULT_ID_FILE)
    ap.add_argument("lag_file", nargs="?", default=DEFAULT_LAG_FILE)
    ap.add_argument("--inputs", default=None,
                    help=f"AMReX inputs file to read the grid from "
                         f"(default: {DEFAULT_INPUTS})")
    ap.add_argument("--dx", type=float, nargs="+", default=None,
                    metavar="D",
                    help="finest-level cell size: 1 value (isotropic) or 3 "
                         "values (dx dy dz). Overrides --inputs.")
    ap.add_argument("--prob-lo", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="domain lower corner (default: from --inputs, else 0 0 0)")
    args = ap.parse_args(argv)

    # Legacy positional form: first positional is a number -> it is dx.
    try:
        legacy_dx = float(args.id_file)
    except (TypeError, ValueError):
        legacy_dx = None
    if legacy_dx is not None:
        if args.dx is None:
            args.dx = [legacy_dx]
        args.id_file, args.lag_file = args.lag_file, DEFAULT_LAG_FILE
        if args.id_file == DEFAULT_LAG_FILE:
            args.id_file = DEFAULT_ID_FILE
    return args


def resolve_grid(args):
    prob_lo = np.zeros(3)
    dx = None

    inputs_path = args.inputs
    if inputs_path is None:
        cand = DEFAULT_INPUTS
        if cand.exists():
            inputs_path = str(cand)
    if inputs_path is not None:
        prob_lo, dx = grid_from_inputs(inputs_path)
        print(f"[grid] from inputs: {inputs_path}")

    if args.dx is not None:
        if len(args.dx) == 1:
            dx = np.full(3, args.dx[0])
        elif len(args.dx) == 3:
            dx = np.array(args.dx, dtype=float)
        else:
            raise SystemExit("--dx takes either 1 or 3 values")
    if dx is None:
        dx = np.full(3, 6.0 / (256 * 8))   # historical default
        print("[grid] WARNING: no --inputs / --dx given, assuming isotropic "
              "6/(256*8)")
    if args.prob_lo is not None:
        prob_lo = np.array(args.prob_lo, dtype=float)
    return prob_lo, dx


def main(argv=None):
    args = parse_args(argv)
    prob_lo, dx = resolve_grid(args)

    print(f"dx      = ({dx[0]:.10g}, {dx[1]:.10g}, {dx[2]:.10g})")
    print(f"prob_lo = ({prob_lo[0]:.10g}, {prob_lo[1]:.10g}, {prob_lo[2]:.10g})")
    if not np.allclose(dx, dx[0]):
        print(f"          anisotropic grid: dy/dx = {dx[1]/dx[0]:.6g}, "
              f"dz/dx = {dx[2]/dx[0]:.6g}")
    print(f"id  = {args.id_file}")
    print(f"lag = {args.lag_file}\n")

    pts = load_id(args.id_file)
    lag = load_lag(args.lag_file)
    N = len(lag)
    print(f"markers: {N} (id), {len(pts)} (lag)")

    sizes = np.array([len(lag[b]) for b in lag])
    print(f"stencil size: min={sizes.min()} max={sizes.max()} mean={sizes.mean():.2f}")

    sumw, moment, second_moment = compute_reproduction_residuals(
        pts, lag, prob_lo, dx
    )

    err_sumw = np.abs(sumw - 1.0).max()
    abs_moment = np.abs(moment)
    max_moment = abs_moment.max(axis=0)
    mean_moment = moment.mean(axis=0)
    std_moment = moment.std(axis=0)
    max_second_moment = np.abs(second_moment).max(axis=0)

    print()
    print(f"[0th moment]  max|Sum w - 1|          = {err_sumw:.3e}   "
          f"(tol {TOL_SUMW:.0e})")
    print(f"[1st moment]  max|Sum w (x-x_l)|/dx   = "
          f"({max_moment[0]:.3e}, {max_moment[1]:.3e}, {max_moment[2]:.3e}) cells   "
          f"(tol {TOL_MOMENT_CELLS:.0e})")
    print(f"              mean over markers       = "
          f"({mean_moment[0]:+.3e}, {mean_moment[1]:+.3e}, {mean_moment[2]:+.3e}) cells")
    print(f"              std  over markers       = "
          f"({std_moment[0]:.3e}, {std_moment[1]:.3e}, {std_moment[2]:.3e}) cells")
    print(f"[2nd moment]  max normalized residual = "
          f"(xy={max_second_moment[0]:.3e}, yz={max_second_moment[1]:.3e}, "
          f"zx={max_second_moment[2]:.3e}, x2={max_second_moment[3]:.3e}, "
          f"y2={max_second_moment[4]:.3e}, z2={max_second_moment[5]:.3e}) "
          f"cells^2   (tol {TOL_SECOND_MOMENT_CELLS2:.0e})")

    bad = max_moment > TOL_MOMENT_CELLS
    if bad.any():
        print()
        if max_moment.max() > 1.0:
            print("  >> DIAGNOSIS: 1st moment is O(1) cells or larger. The cell "
                  "centers reconstructed")
            print("     here do not match the ones the generator used. Check that "
                  "dx/dy/dz and prob_lo")
            print("     passed to this script are the SOLVER'S finest-level values "
                  "for each direction")
            print("     (a scalar dx on an anisotropic grid produces exactly this "
                  "signature, and only in")
            print("     the directions whose spacing was wrong: "
                  f"{[ 'xyz'[i] for i in np.nonzero(bad)[0] ]}).")
        elif std_moment.max() < 1e-6:
            print("  >> DIAGNOSIS: 1st moment is a NONZERO CONSTANT across all markers.")
            print("     This is the generator/solver grid-origin misalignment "
                  "(int(sx/dx) truncation).")
            print(f"     Effective constant shift = "
                  f"({mean_moment[0]:+.3f}, {mean_moment[1]:+.3f}, "
                  f"{mean_moment[2]:+.3f}) cells.")
            print("     Fix: snap sx,sy,sz to integer multiples of dx in main.py "
                  "(and dx = Lx/nxc must equal the solver finest dx).")

    ok = (
        err_sumw < TOL_SUMW
        and max_moment.max() < TOL_MOMENT_CELLS
        and max_second_moment.max() < TOL_SECOND_MOMENT_CELLS2
    )
    print()
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
