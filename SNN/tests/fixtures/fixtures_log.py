# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
log fixtures
============

Fixtures used for the lof tests
"""

import pytest

from SNN2.src.io.logger import LogHandler as LH

@pytest.fixture
def log_file(tmp_path):
    p = tmp_path / "log"
    p.mkdir()
    p_init = p / "log.log"
    p_init.touch()
    return p_init

@pytest.fixture
def dummy_log(log_file):
    return LH(log_file, 0)


