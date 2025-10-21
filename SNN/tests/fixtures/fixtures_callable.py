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
def dummy_callable():
    def dummy(*args, **kwargs):
        return args, kwargs
    return dummy
