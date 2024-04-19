# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Logger fixtures
===============

Use this module to manage the fixtures for the configuration tests
"""

import pytest
import yaml

from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.io.IOHandler import IOHandler as IOH

@pytest.fixture
def test_logger(test_io: IOH) -> LH:
    return LH(test_io[s.log_file], 6)
