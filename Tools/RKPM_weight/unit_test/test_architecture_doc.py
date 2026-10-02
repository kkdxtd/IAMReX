# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

"""Structural checks for the maintained RKPM Weight architecture document."""

from __future__ import annotations

import unittest

from unit_test.common import TOOL_ROOT


class ArchitectureDocumentTests(unittest.TestCase):
    def test_architecture_document_lists_every_production_module(self):
        architecture = (TOOL_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
        production_modules = [TOOL_ROOT / "main.py", *sorted((TOOL_ROOT / "src").glob("*.py"))]
        missing = [
            str(module.relative_to(TOOL_ROOT))
            for module in production_modules
            if str(module.relative_to(TOOL_ROOT)) not in architecture
        ]
        self.assertEqual(
            missing,
            [],
            msg=f"Production modules missing from ARCHITECTURE.md: {missing}",
        )


if __name__ == "__main__":
    unittest.main()
