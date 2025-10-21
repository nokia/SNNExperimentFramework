# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
log fixtures
============

Fixtures used for the lof tests
"""

import pytest

from SNN2.src.util.strings import s


@pytest.fixture
def generic_param():
    return {
        "name": "test",
        "type": "generic",
        "value": 12
    }

@pytest.fixture
def callable_param():
    return {
        "name": "test",
        "type": "generic",
        "value": 12,
        s.param_action_args: (1, 2, 3),
        s.param_action_kwargs: {"1": 1, "2": 2, "3": 3}
    }

@pytest.fixture
def env_rng_param():
    return {
        "name": f"{s.env_rng}",
        "type": "environment",
        "value": "0"
    }

@pytest.fixture
def env_tf_rng_param():
    return {
        "name": f"{s.env_tf_rng}",
        "type": "environment",
        "value": "0"
    }
