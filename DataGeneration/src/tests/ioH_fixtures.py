# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
io fixtures
===========

Use this module to manage the fixtures for the configuration tests
"""

import pytest

from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.configuration import ConfHandler as CH
from Dkr5G.src.io.IOHandler import IOHandler as IOH

@pytest.fixture
def test_configuration():
    conf_file = "tests/files/test_io_conf.cfg"
    conf_fh = FH(conf_file, create=False)
    return CH(conf_fh)

@pytest.fixture
def test_configuration_error():
    conf_file = "tests/files/test_io_conf_error.cfg"
    conf_fh = FH(conf_file, create=False)
    return CH(conf_fh)

@pytest.fixture
def test_cfg_pkg():
    conf_file = "tests/files/test_cfg_pkg.cfg"
    conf_fh = FH(conf_file, create=False)
    return CH(conf_fh)

@pytest.fixture
def test_io():
    conf_file = "tests/files/test_io_conf.cfg"
    return IOH.from_cfg(CH(FH(conf_file, create=False)))

@pytest.fixture
def command_conf():
    conf_file = "tests/files/test_io_command_conf.cfg"
    return CH(FH(conf_file, create=False))

