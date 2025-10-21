# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
log fixtures
============

Fixtures used for the lof tests
"""

import pytest

@pytest.fixture
def tmp_ioh_obj(tmp_files_path):
    return "tests/files/conf"



