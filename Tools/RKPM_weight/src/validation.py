# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Numerical validation helpers shared by file and MPMD workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WeightMomentErrors:
    """Maximum zeroth- and first-moment residuals for a marker set."""

    sum_error: float
    moment_error: float
    moment_error_by_axis: np.ndarray

    def passes(self, sum_tolerance: float, moment_tolerance: float) -> bool:
        """Return whether both residuals satisfy their configured tolerances."""
        return (
            self.sum_error <= sum_tolerance
            and self.moment_error <= moment_tolerance
        )


def weights_as_array(
    weights: Mapping[int, Sequence[float]], marker_ids: Sequence[int]
) -> np.ndarray:
    """Convert a marker-indexed weight mapping to an ``[N, stencil]`` array."""
    missing = set(marker_ids) - set(weights)
    extra = set(weights) - set(marker_ids)
    if missing or extra:
        raise ValueError(
            f"Weight and stencil IDs differ; missing {sorted(missing)[:5]}, "
            f"extra {sorted(extra)[:5]}"
        )
    try:
        array = np.asarray([weights[marker_id] for marker_id in marker_ids])
    except KeyError as exc:
        raise ValueError(f"Missing weights for marker {exc.args[0]}") from exc

    if array.ndim != 2 or array.shape[0] != len(marker_ids):
        raise ValueError(
            "Marker weights must form a rectangular [marker, stencil] array; "
            f"got shape {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("Marker weights contain NaN or Inf")
    return array


def compute_weight_moment_errors(
    positions: np.ndarray,
    stencil_centers: np.ndarray,
    weights: np.ndarray,
    dx: np.ndarray,
) -> WeightMomentErrors:
    """Compute per-marker zeroth- and first-moment residual maxima.

    The first moment is normalized by the grid spacing in each direction, so
    its residual is expressed in units of Eulerian cells.
    """
    positions = np.asarray(positions, dtype=float)
    stencil_centers = np.asarray(stencil_centers, dtype=float)
    weights = np.asarray(weights)
    dx = np.asarray(dx, dtype=float)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Marker positions must have shape (N, 3), got {positions.shape}"
        )
    if positions.shape[0] == 0:
        raise ValueError("At least one marker is required for a moment check")
    if weights.ndim != 2 or weights.shape[0] != positions.shape[0]:
        raise ValueError(
            "Weights must have shape (N, stencil) matching marker positions; "
            f"got {weights.shape}"
        )
    expected_centers = (positions.shape[0], weights.shape[1], 3)
    if stencil_centers.shape != expected_centers:
        raise ValueError(
            f"Stencil centers must have shape {expected_centers}, "
            f"got {stencil_centers.shape}"
        )
    if dx.shape != (3,) or not np.all(np.isfinite(dx)) or np.any(dx <= 0.0):
        raise ValueError(f"Grid spacing must contain three positive values, got {dx}")
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(stencil_centers))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("Moment-check inputs contain NaN or Inf")

    sum_error = float(np.max(np.abs(weights.sum(axis=1, dtype=float) - 1.0)))
    relative = (stencil_centers - positions[:, None, :]) / dx
    moments = np.einsum(
        "ni,nij->nj", weights.astype(float, copy=False), relative
    )
    moment_error_by_axis = np.max(np.abs(moments), axis=0)
    return WeightMomentErrors(
        sum_error=sum_error,
        moment_error=float(np.max(moment_error_by_axis)),
        moment_error_by_axis=moment_error_by_axis,
    )
