#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 IAMReX contributors
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

test_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tool_root="$(dirname -- "${test_dir}")"
test_python="${RKPM_TEST_PYTHON:-python3}"

if ! command -v "${test_python}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${test_python}" >&2
    echo "Set RKPM_TEST_PYTHON to an installed Python executable." >&2
    exit 127
fi

cd "${tool_root}"
exec "${test_python}" -m unittest discover \
    -s unit_test \
    -p 'test_*.py' \
    -v
