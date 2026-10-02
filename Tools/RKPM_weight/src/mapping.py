"""Build 3D RKPM mappings.

This module provides marker-ID-to-coordinate and Lagrangian-to-Eulerian
mappings. Eulerian points are global cell centers on the solver's finest grid,
so their indices are computed directly as
``floor((x - prob_lo) / dx_finest)`` after snapping roundoff-close grid-line
coordinates, without a local-grid offset. This avoids the ambiguity caused by
the former ``int(sx/dx)`` truncation.
"""

import ast
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Union

import numpy as np

from .grid_index import containing_cell_indices


NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def build_lagrangian_id_to_coord_map(lagrangian_points: np.ndarray) -> Dict[int, tuple]:
    """Build ``{marker_id: (xp, yp, zp)}`` from an ``(Ne, 3)`` array."""
    return {
        lag_id: tuple(coord)
        for lag_id, coord in enumerate(np.asarray(lagrangian_points))
    }


def load_id_map(filename: Union[str, Path]) -> Dict[int, tuple]:
    """Read ``rkpm_mappings.id`` and validate marker IDs and coordinates."""
    with open(filename, "r", encoding="utf-8") as f:
        raw = ast.literal_eval(f.read())

    if not isinstance(raw, dict):
        raise ValueError(f"The top level of an ID file must be a dictionary: {filename}")

    result = {}
    for marker_id, coord in raw.items():
        marker_id = int(marker_id)
        if len(coord) != 3:
            raise ValueError(f"Marker {marker_id} does not have 3D coordinates: {coord}")
        result[marker_id] = tuple(float(value) for value in coord)
    return result


def load_lag_map(
    filename: Union[str, Path]
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Read stencil and weight metadata from ``rkpm_mappings.lag``."""
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    result = {}
    for marker_id, body in re.findall(r"(\d+)\s*:\s*\[(.*?)\]", text, re.S):
        rows = []
        for entry in re.findall(r"\{([^}]*)\}", body):
            values = dict(re.findall(
                rf'"(\w+)"\s*:\s*({NUMBER_PATTERN})', entry
            ))
            missing = {"i", "j", "k"} - values.keys()
            if missing:
                raise ValueError(
                    f"Marker {marker_id} stencil entry is missing fields: {sorted(missing)}"
                )
            rows.append({
                "i": int(values["i"]),
                "j": int(values["j"]),
                "k": int(values["k"]),
                "w": float(values.get("w", 0.0)),
                "Vcell": float(values.get("Vcell", 1.0)),
                "eps": float(values.get("eps", 0.0)),
            })
        result[int(marker_id)] = rows

    if not result:
        raise ValueError(f"No markers found in lag file: {filename}")
    return result


def validate_mapping_ids(
    id_to_coord_map: Mapping[int, Sequence[float]],
    lag_to_eul_map: Mapping[int, Sequence[Mapping[str, Union[int, float]]]],
    *,
    require_contiguous: bool = True,
) -> List[int]:
    """Validate matching marker sets and return their deterministic order."""
    coord_ids = set(id_to_coord_map)
    stencil_ids = set(lag_to_eul_map)
    if coord_ids != stencil_ids:
        only_id = sorted(coord_ids - stencil_ids)
        only_lag = sorted(stencil_ids - coord_ids)
        raise ValueError(
            f".id/.lag marker IDs differ; only in .id: {only_id[:5]}, "
            f"only in .lag: {only_lag[:5]}"
        )

    marker_ids = sorted(stencil_ids)
    if require_contiguous and marker_ids != list(range(len(marker_ids))):
        raise ValueError(
            "Marker IDs must be contiguous from zero to match the C++ flat array"
        )
    return marker_ids


def replace_mapping_weights(
    lag_to_eul_map: Mapping[int, Sequence[Mapping[str, Union[int, float]]]],
    weights: Mapping[int, Sequence[float]],
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Preserve stencil/volume metadata and replace only each entry's ``w``."""
    result = deepcopy(lag_to_eul_map)
    if set(result) != set(weights):
        raise ValueError("Weight-result IDs do not match the lag mapping IDs")

    for marker_id, rows in result.items():
        marker_weights = np.asarray(weights[marker_id], dtype=float).reshape(-1)
        if len(rows) != len(marker_weights):
            raise ValueError(
                f"Marker {marker_id} has {len(marker_weights)} weights but "
                f"{len(rows)} stencil entries"
            )
        for row, weight in zip(rows, marker_weights):
            row["w"] = float(weight)
    return result


def build_lag_to_eul_map(
    lagrangian_points: np.ndarray,
    all_S_I: List[np.ndarray],
    all_modified_w: List[List[float]],
    prob_lo: np.ndarray,
    dx_finest: np.ndarray,
    V_lag: float
) -> Dict[int, List[Dict[str, Union[int, float]]]]:
    """Build the Lagrangian-to-Eulerian force-spreading mapping.

    Eulerian points in ``all_S_I`` are global cell centers on the finest grid.
    Therefore, snapping roundoff-close grid-line coordinates and then applying
    ``floor((x - prob_lo) / dx_finest)`` recovers global cell indices directly
    without a local-to-global offset.
    """
    prob_lo = np.asarray(prob_lo, dtype=float)
    dx_finest = np.asarray(dx_finest, dtype=float)
    lag_to_eul_map = {}

    for lag_id in range(len(lagrangian_points)):
        S_I = np.asarray(
            all_S_I[lag_id], dtype=float
        )  # Eulerian points and volumes in the support domain
        modified_w = np.asarray(
            all_modified_w[lag_id], dtype=float
        ).reshape(-1)
        if S_I.shape != (len(modified_w), 4):
            raise ValueError(
                f"Marker {lag_id} support must have shape "
                f"({len(modified_w)}, 4), got {S_I.shape}"
            )

        cell_indices = containing_cell_indices(
            S_I[:, :3], prob_lo, dx_finest
        )
        # np.lexsort uses the final key as primary: i, then j, then k.
        order = np.lexsort(
            (cell_indices[:, 2], cell_indices[:, 1], cell_indices[:, 0])
        )
        cell_indices = cell_indices[order]
        modified_w = modified_w[order]
        eps = float(V_lag) / S_I[order, 3]
        eulerian_data = [
            {
                "i": int(cell[0]),
                "j": int(cell[1]),
                "k": int(cell[2]),
                "w": float(weight),
                "Vcell": 1.0,
                "eps": float(marker_eps),
            }
            for cell, weight, marker_eps in zip(
                cell_indices, modified_w, eps
            )
        ]

        # Store the marker mapping.
        lag_to_eul_map[lag_id] = eulerian_data

    return lag_to_eul_map

def save_mappings_txt(
    id_to_coord_map: Dict[int, tuple],
    lag_to_eul_map: Dict[int, List[Dict[str, Union[int, float]]]],
    filename: str
) -> None:
    """Write ID-to-coordinate and Lagrangian-to-Eulerian text mappings."""
    # Write the marker-ID-to-coordinate mapping.
    with open(filename + ".id", 'w') as f:
        f.write("{\n")
        for ids, (x, y, z) in id_to_coord_map.items():
            f.write(f"    {ids}: ({x}, {y}, {z}),\n")
        f.write("}")

    # Write lag_to_eul_map in the format consumed by IAMReX.
    with open(filename + ".lag", 'w') as f:
        f.write("{\n")
        for lag_id, eul_list in lag_to_eul_map.items():
            # Write the marker ID and its Eulerian stencil.
            f.write(f"    {lag_id}: [\n")
            for eul_info in eul_list:
                # Format one Eulerian cell entry.
                line = (
                    f"        {{"
                    f"\"i\": {eul_info['i']}, "
                    f"\"j\": {eul_info['j']}, "
                    f"\"k\": {eul_info['k']}, "
                    f"\"w\": {eul_info['w']}, "
                    f"\"Vcell\": {eul_info['Vcell']}, "
                    f"\"eps\": {eul_info['eps']}"
                    f"}},\n"
                )
                f.write(line)
            f.write("    ],\n")  # End the current marker list.
        f.write("}")  # End the mapping dictionary.
