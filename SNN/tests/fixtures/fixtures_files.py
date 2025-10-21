# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
log fixtures
============

Fixtures used for the lof tests
"""

import os
import pytest

@pytest.fixture
def tmp_files_path(tmp_path):
    p = tmp_path / "test_files"
    if not os.path.exists(p):
        p.mkdir()
    return p

@pytest.fixture
def tmp_file(tmp_files_path):
    p_init = tmp_files_path / "test_file.txt"
    p_init.touch()
    return str(p_init)

@pytest.fixture
def second_tmp_file(tmp_files_path):
    p_init = tmp_files_path / "second_test_file.txt"
    p_init.touch()
    return str(p_init)
