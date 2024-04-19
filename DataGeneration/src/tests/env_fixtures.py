# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
conf fixtures
=============

Use this module to manage the fixtures for the configuration tests
"""

import pytest

from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.configuration import ConfHandler as CH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.core.environment import EnvironmentHandler as ENV

@pytest.fixture
def environment_configuration():
    conf_file = "tests/files/test_environment.cfg"
    conf_fh = FH(conf_file, create=False)
    return CH(conf_fh)

@pytest.fixture
def env_io(test_configuration):
    io = IOH.from_cfg(test_configuration)
    return io

@pytest.fixture
def env_logger(env_io):
    return LH(env_io["Log"], LH.DEBUG)

@pytest.fixture
def test_env(environment_configuration, test_io, test_logger) -> ENV:
    return ENV.from_cfg(environment_configuration,
                        test_io,
                        test_logger)
