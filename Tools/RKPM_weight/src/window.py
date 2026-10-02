# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized RKPM window and polynomial-correction operations."""

from __future__ import annotations

from typing import Sequence

import numpy as np


POLYNOMIAL_SIZE = 10


def window_function_d(r):
    """Evaluate the three-point one-dimensional kernel on scalars or arrays."""
    values = np.asarray(r, dtype=float)
    absolute = np.abs(values)
    result = np.zeros_like(values)

    inner = absolute < 0.5
    middle = (absolute >= 0.5) & (absolute <= 1.5)
    result[inner] = (1.0 / 3.0) * (
        1.0 + np.sqrt(np.maximum(0.0, 1.0 - 3.0 * values[inner] ** 2))
    )
    result[middle] = (1.0 / 6.0) * (
        5.0
        - 3.0 * absolute[middle]
        - np.sqrt(
            np.maximum(0.0, 1.0 - 3.0 * (1.0 - absolute[middle]) ** 2)
        )
    )
    return float(result) if result.ndim == 0 else result

# input : relative displacements of stencil points from the lagrangian point, shape (..., 3)
# output: polynomial basis evaluated at each displacement, shape (..., 10)
def polynomial_basis(displacements: np.ndarray) -> np.ndarray:
    """Return ``[1,x,y,z,xy,yz,zx,x^2,y^2,z^2]`` for each displacement."""
    displacements = np.asarray(displacements, dtype=float)
    if displacements.shape[-1:] != (3,):
        raise ValueError(
            "RKPM displacements must end in three coordinates; "
            f"got shape {displacements.shape}"
        )
    x = displacements[..., 0]
    y = displacements[..., 1]
    z = displacements[..., 2]
    return np.stack(
        (
            np.ones_like(x),
            x,
            y,
            z,
            x * y,
            y * z,
            z * x,
            x * x,
            y * y,
            z * z,
        ),
        axis=-1,
    )


def _marker_vector(values, marker_count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        array = np.full(marker_count, float(array))
    if array.shape != (marker_count,):
        raise ValueError(
            f"{name} must be scalar or shape ({marker_count},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return array


def _prepare_batch(
    support_domains,
    lagrangian_points,
    delta,
    eta,
    theta,
    lagrangian_volume,
):
    supports = np.asarray(support_domains, dtype=float)
    points = np.asarray(lagrangian_points, dtype=float)
    if supports.ndim != 3 or supports.shape[2] != 4:
        raise ValueError(
            "Support domains must have shape (markers, stencil, 4), "
            f"got {supports.shape}"
        )
    marker_count = supports.shape[0]
    if points.shape != (marker_count, 3):
        raise ValueError(
            "Lagrangian points must have shape "
            f"({marker_count}, 3), got {points.shape}"
        )
    if not np.all(np.isfinite(supports)) or not np.all(np.isfinite(points)):
        raise ValueError("RKPM inputs contain NaN or Inf")
    if np.any(supports[..., 3] <= 0.0):
        raise ValueError("Eulerian cell volumes must be positive")

    scales = np.column_stack(
        (
            _marker_vector(delta, marker_count, "delta"),
            _marker_vector(eta, marker_count, "eta"),
            _marker_vector(theta, marker_count, "theta"),
        )
    )
    volumes = _marker_vector(lagrangian_volume, marker_count, "V_lag")
    displacements = supports[..., :3] - points[:, None, :]
    base_weights = (
        np.prod(window_function_d(displacements / scales[:, None, :]), axis=-1)
        * supports[..., 3]
        / volumes[:, None]
    )
    basis = polynomial_basis(displacements)
    return base_weights, basis


def compute_m_ab_matrix(S_I, lagrangian_points, delta_I, eta_I, theta_I, V_lag):
    """Compute one marker's ``(10,10)`` RKPM moment matrix."""
    base_weights, basis = _prepare_batch(
        np.asarray(S_I, dtype=float)[None, ...],
        np.asarray(lagrangian_points, dtype=float)[None, ...],
        delta_I,
        eta_I,
        theta_I,
        V_lag,
    )
    return np.einsum(
        "nsi,ns,nsj->nij", basis, base_weights, basis, optimize=True
    )[0]


def compute_b_I(M_I):
    """Solve one moment system, or a batch of moment systems, for corrections."""
    matrices = np.asarray(M_I, dtype=float)
    if matrices.shape[-2:] != (POLYNOMIAL_SIZE, POLYNOMIAL_SIZE):
        raise ValueError(
            "Moment matrices must end in shape (10, 10), "
            f"got {matrices.shape}"
        )
    right_hand_side = np.zeros(matrices.shape[:-1], dtype=float)
    right_hand_side[..., 0] = 1.0
    # NumPy >= 2.0 treats b as a vector only when b.ndim == 1. Represent the
    # batched right-hand sides as explicit columns so this works consistently
    # across NumPy versions, then remove the singleton column from the result.
    return np.linalg.solve(matrices, right_hand_side[..., None])[..., 0]


def modified_window_function(S_I, lagrangian_point, d_I, delta, eta, theta, V_lag):
    """Compute corrected window values for one Lagrangian marker."""
    base_weights, basis = _prepare_batch(
        np.asarray(S_I, dtype=float)[None, ...],
        np.asarray(lagrangian_point, dtype=float)[None, ...],
        delta,
        eta,
        theta,
        V_lag,
    )
    corrections = np.asarray(d_I, dtype=float)
    if corrections.shape != (POLYNOMIAL_SIZE,):
        raise ValueError(f"d_I must have shape (10,), got {corrections.shape}")
    return (base_weights[0] * (basis[0] @ corrections)).tolist()


def compute_modified_window_functions_batch(
    support_domains,
    lagrangian_points,
    delta,
    eta,
    theta,
    V_lag,
) -> np.ndarray:
    """Compute corrected weights for a rectangular marker batch.

    All stencil points and all ten-by-ten moment systems in the batch are
    evaluated with NumPy array operations. The returned array has shape
    ``(markers, stencil)``.
    """
    base_weights, basis = _prepare_batch(
        support_domains,
        lagrangian_points,
        delta,
        eta,
        theta,
        V_lag,
    )
    moment_matrices = np.einsum(
        "nsi,ns,nsj->nij", basis, base_weights, basis, optimize=True
    )
    corrections = compute_b_I(moment_matrices)
    return base_weights * np.einsum(
        "nsi,ni->ns", basis, corrections, optimize=True
    )


def compute_all_modified_window_functions(
    all_S_I: Sequence[np.ndarray],
    lagrangian_points,
    delta_I,
    eta_I,
    theta_I,
    V_lag,
):
    """Compute corrected windows for all markers, grouping ragged stencils.

    The common 3x3x3 case is one fully vectorized call. Grouping by stencil
    length preserves compatibility with truncated boundary stencils without
    falling back to a per-marker numerical solve.
    """
    marker_count = len(all_S_I)
    points = np.asarray(lagrangian_points, dtype=float)
    if points.shape != (marker_count, 3):
        raise ValueError(
            "Lagrangian points must have shape "
            f"({marker_count}, 3), got {points.shape}"
        )
    delta = _marker_vector(delta_I, marker_count, "delta_I")
    eta = _marker_vector(eta_I, marker_count, "eta_I")
    theta = _marker_vector(theta_I, marker_count, "theta_I")
    volumes = _marker_vector(V_lag, marker_count, "V_lag")

    grouped_indices = {}
    for marker_index, support in enumerate(all_S_I):
        grouped_indices.setdefault(len(support), []).append(marker_index)

    result = [None] * marker_count
    for marker_indices in grouped_indices.values():
        selected = np.asarray(marker_indices, dtype=int)
        solved = compute_modified_window_functions_batch(
            np.stack([all_S_I[index] for index in marker_indices]),
            points[selected],
            delta[selected],
            eta[selected],
            theta[selected],
            volumes[selected],
        )
        for local_index, marker_index in enumerate(marker_indices):
            result[marker_index] = solved[local_index]
    return result
