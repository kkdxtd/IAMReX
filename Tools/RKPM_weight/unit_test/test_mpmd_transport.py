# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Cross-process test for the RKPM MPMD protocol."""

from __future__ import annotations

import shutil
import subprocess
import sys
import unittest

from unit_test.common import TEST_DIR, TOOL_ROOT


class MPMDTransportTests(unittest.TestCase):
    def test_mpmd_preserves_double_positions_and_float32_weight_order(self):
        mpiexec = shutil.which("mpiexec")
        if mpiexec is None:
            self.skipTest("mpiexec is not available")
        try:
            import mpi4py  # noqa: F401
        except ImportError:
            self.skipTest("mpi4py is not available")

        command = [
            mpiexec,
            "-n",
            "1",
            sys.executable,
            str(TEST_DIR / "mpmd_client.py"),
            ":",
            "-n",
            "1",
            sys.executable,
            str(TEST_DIR / "mpmd_server.py"),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=TOOL_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(f"MPMD protocol test timed out: {exc}")

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, msg=output)
        self.assertIn("MPMD_CLIENT_PASS", output)
        self.assertIn("Received shutdown signal", output)


if __name__ == "__main__":
    unittest.main()
