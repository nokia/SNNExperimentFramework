# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
conf fixtures
=============

Use this module to manage the fixtures for the configuration tests
"""

import pytest

@pytest.fixture
def conf_file():
    return "tests/files/test_configuration.cfg"

@pytest.fixture
def nodefault_conf_file():
    return "tests/files/test_configuration_not_default.cfg"

@pytest.fixture
def conf_dir():
    return "tests/files/envFolder"

@pytest.fixture
def other_conf_dir():
    return "tests/files/secondEnvFolder"
