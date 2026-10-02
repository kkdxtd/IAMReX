# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for vectorization-related runtime configuration."""

from __future__ import annotations

import contextlib
import io
import unittest

from unit_test.common import TEST_DIR

import main


class RuntimeConfigTests(unittest.TestCase):
    def test_config_loads_batch_size_and_device(self):
        args = main.parse_args(["--config", str(TEST_DIR / "inputs.rkpm")])
        self.assertEqual(args.batch_size, 4096)
        self.assertEqual(args.device, "cpu")
        self.assertIs(args.body_frame, False)

    def test_command_line_overrides_batch_size_and_device(self):
        args = main.parse_args(
            [
                "--config",
                str(TEST_DIR / "inputs.rkpm"),
                "--batch-size",
                "17",
                "--device",
                "auto",
            ]
        )
        self.assertEqual(args.batch_size, 17)
        self.assertEqual(args.device, "auto")

    def test_geometry_generation_requires_explicit_coordinate_frame(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            main.parse_args(
                [
                    "--transport",
                    "file",
                    "--geometry",
                    "points.txt",
                ]
            )
        self.assertIn(
            "requires an explicit coordinate frame",
            stderr.getvalue(),
        )

    def test_geometry_generation_accepts_body_or_world_frame(self):
        body_args = main.parse_args(
            ["--transport", "file", "--geometry", "points.txt", "--body-frame"]
        )
        world_args = main.parse_args(
            ["--transport", "file", "--geometry", "points.txt", "--world-frame"]
        )
        self.assertIs(body_args.body_frame, True)
        self.assertIs(world_args.body_frame, False)


if __name__ == "__main__":
    unittest.main()
