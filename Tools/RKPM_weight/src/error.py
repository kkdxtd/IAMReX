# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Vectorized interpolation and conservation diagnostics for RKPM mappings."""

from __future__ import annotations

import numpy as np


def test_function(x, y):
    """Return ``sin(pi*x)*cos(pi*y)`` for scalar or array coordinates."""
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def compute_infinity_norm_error(original_values, interpolated_values):
    """Return the infinity-norm error between two value arrays."""
    return np.max(np.abs(original_values - interpolated_values))


def _flatten_supports(all_S_I, all_modified_w):
    if len(all_S_I) != len(all_modified_w):
        raise ValueError("Support and weight marker counts differ")
    counts = np.asarray([len(support) for support in all_S_I], dtype=int)
    supports = np.concatenate(
        [np.asarray(support, dtype=float) for support in all_S_I], axis=0
    )
    weights = np.concatenate(
        [
            np.asarray(marker_weights, dtype=float).reshape(-1)
            for marker_weights in all_modified_w
        ]
    )
    if supports.shape != (int(counts.sum()), 4):
        raise ValueError(
            "Flattened support domains must have four columns; "
            f"got {supports.shape}"
        )
    if len(weights) != len(supports):
        raise ValueError(
            f"Support entries ({len(supports)}) and weights ({len(weights)}) differ"
        )
    marker_indices = np.repeat(np.arange(len(counts)), counts)
    return supports, weights, marker_indices, counts


def _support_to_euler_indices(eulerian_points, support_points):
    """Map exact pipeline coordinates without per-point Python dictionary lookups."""
    eulerian_points = np.asarray(eulerian_points, dtype=float)
    combined = np.concatenate((eulerian_points, support_points), axis=0)
    _, inverse = np.unique(combined, axis=0, return_inverse=True)
    eulerian_codes = inverse[: len(eulerian_points)]
    support_codes = inverse[len(eulerian_points) :]
    code_to_euler = np.full(int(inverse.max()) + 1, -1, dtype=int)
    code_to_euler[eulerian_codes] = np.arange(len(eulerian_points))
    return code_to_euler[support_codes]


def compute_error_volume(lagrangian_points, all_S_I, all_modified_w, V_lag):
    """Compute each marker's relative volume-reproduction residual."""
    del lagrangian_points
    _, weights, marker_indices, _ = _flatten_supports(all_S_I, all_modified_w)
    weight_sums = np.zeros(len(all_S_I))
    np.add.at(weight_sums, marker_indices, weights)
    relative_error = np.abs(weight_sums - 1.0)
    if not np.isfinite(V_lag) or V_lag <= 0.0:
        raise ValueError("V_lag must be finite and positive")
    print(
        "Relative volume error: "
        f"Min: {np.min(relative_error):.6e}, Max: {np.max(relative_error):.6e}"
    )
    return relative_error


def compute_error(
    eulerian_points,
    lagrangian_points,
    all_S_I,
    all_modified_w,
    V_lag,
):
    """Evaluate constant-field interpolation error and force conservation."""
    original_values = np.ones(len(lagrangian_points))
    supports, weights, marker_indices, _ = _flatten_supports(
        all_S_I, all_modified_w
    )
    euler_indices = _support_to_euler_indices(
        eulerian_points, supports[:, :3]
    )
    valid = euler_indices >= 0

    dispersion_values = np.zeros(len(eulerian_points))
    contributions = (
        original_values[marker_indices[valid]]
        * weights[valid]
        * V_lag
        / supports[valid, 3]
    )
    np.add.at(dispersion_values, euler_indices[valid], contributions)

    interpolated_values = np.zeros(len(lagrangian_points))
    np.add.at(
        interpolated_values,
        marker_indices[valid],
        dispersion_values[euler_indices[valid]] * weights[valid],
    )

    force_lagrangian = np.sum(original_values * V_lag)
    grid_volumes = np.zeros(len(eulerian_points))
    grid_volumes[euler_indices[valid]] = supports[valid, 3]
    force_euler = np.sum(dispersion_values * grid_volumes)
    print(
        "force check",
        force_lagrangian,
        force_euler,
        abs(force_lagrangian - force_euler) / force_lagrangian,
    )
    return compute_infinity_norm_error(original_values, interpolated_values)


def compute_conservation_check(
    center,
    eulerian_points,
    lagrangian_points,
    all_S_I,
    all_modified_w,
    V_lag,
):
    """Check force and moment/torque conservation simultaneously."""
    center = np.asarray(center, dtype=float)
    lagrangian_points = np.asarray(lagrangian_points, dtype=float)
    supports, weights, marker_indices, _ = _flatten_supports(
        all_S_I, all_modified_w
    )
    euler_indices = _support_to_euler_indices(
        eulerian_points, supports[:, :3]
    )
    valid = euler_indices >= 0

    rel_pos = lagrangian_points - center
    omega = np.array([0.7, -0.3, 0.5])
    constant_force = np.array([1.0, 0.8, 1.2])
    lagrangian_forces = constant_force + np.cross(omega, rel_pos)

    total_force_lag = np.sum(lagrangian_forces * V_lag, axis=0)
    total_torque_lag = np.sum(
        np.cross(rel_pos, lagrangian_forces) * V_lag, axis=0
    )
    print(f"Lagrangian Total Force: {total_force_lag}")
    print(f"Lagrangian Total Torque: {total_torque_lag}")

    eulerian_forces = np.zeros((len(eulerian_points), 3))
    contributions = (
        lagrangian_forces[marker_indices[valid]]
        * weights[valid, None]
        * V_lag
        / supports[valid, 3, None]
    )
    np.add.at(eulerian_forces, euler_indices[valid], contributions)

    grid_volumes = np.zeros(len(eulerian_points))
    grid_volumes[euler_indices[valid]] = supports[valid, 3]
    affected = grid_volumes > 0.0
    affected_points = np.asarray(eulerian_points)[affected]
    affected_forces = eulerian_forces[affected]
    affected_volumes = grid_volumes[affected, None]

    total_force_euler = np.sum(affected_forces * affected_volumes, axis=0)
    total_torque_euler = np.sum(
        np.cross(affected_points - center, affected_forces) * affected_volumes,
        axis=0,
    )
    print(f"Eulerian Total Force: {total_force_euler}")
    print(f"Eulerian Total Torque: {total_torque_euler}")

    force_relative_error = np.linalg.norm(
        total_force_lag - total_force_euler
    ) / np.linalg.norm(total_force_lag)
    torque_relative_error = np.linalg.norm(
        total_torque_lag - total_torque_euler
    ) / np.linalg.norm(total_torque_lag)
    print("-" * 30)
    print(f"Force Conservation Relative Error: {force_relative_error:.6e}")
    print(f"Torque Conservation Relative Error: {torque_relative_error:.6e}")
    return force_relative_error, torque_relative_error
