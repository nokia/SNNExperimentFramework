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
def test_file():
    return "tests/files/test_file.txt"

@pytest.fixture
def second_test_file():
    return "tests/files/test_file2.txt"

@pytest.fixture
def detect_test_folder():
    return "tests/files/autoDetect"

@pytest.fixture
def detect_test_folder_error():
    return "tests/files/autoDetectError"
