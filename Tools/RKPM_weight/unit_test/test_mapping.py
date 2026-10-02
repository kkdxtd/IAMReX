# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for RKPM mapping serialization and weight replacement."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from unit_test.common import build_small_fixture

from src import mapping


class MappingTests(unittest.TestCase):
    def test_mapping_round_trip_preserves_order_and_metadata(self):
        _, _, id_map, lag_map = build_small_fixture()
        weights = {
            marker_id: np.linspace(
                marker_id, marker_id + 0.26, len(rows), dtype=float
            )
            for marker_id, rows in lag_map.items()
        }
        solved = mapping.replace_mapping_weights(lag_map, weights)

        for rows in lag_map.values():
            self.assertTrue(all(row["w"] == 0.0 for row in rows))

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "round_trip"
            mapping.save_mappings_txt(id_map, solved, str(prefix))
            loaded_id = mapping.load_id_map(prefix.with_suffix(".id"))
            loaded_lag = mapping.load_lag_map(prefix.with_suffix(".lag"))

        self.assertEqual(mapping.validate_mapping_ids(loaded_id, loaded_lag), [0, 1, 2])
        self.assertEqual(loaded_id, id_map)
        for marker_id in solved:
            self.assertEqual(len(loaded_lag[marker_id]), 27)
            for expected, actual in zip(solved[marker_id], loaded_lag[marker_id]):
                self.assertEqual(actual["i"], expected["i"])
                self.assertEqual(actual["j"], expected["j"])
                self.assertEqual(actual["k"], expected["k"])
                self.assertEqual(actual["Vcell"], expected["Vcell"])
                self.assertEqual(actual["eps"], expected["eps"])
                self.assertEqual(actual["w"], expected["w"])


if __name__ == "__main__":
    unittest.main()
