# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
conf fixtures
=============

Use this module to manage the fixtures for the configuration tests
"""

import pytest
import yaml

from Dkr5G.src.core.graphHandler import GraphHandler as GH
from Dkr5G.src.core.environment import EnvironmentHandler as ENV
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.io.logger import LogHandler as LH
from typing import List, Dict, Any

@pytest.fixture
def exp_graph(test_graph) -> GH:
    return test_graph

@pytest.fixture
def exp_env(test_env) -> ENV:
    return test_env

@pytest.fixture
def exp_io(test_io) -> IOH:
    return test_io

@pytest.fixture
def exp_logger(test_logger) -> LH:
    return test_logger

@pytest.fixture
def exp_events() -> List[Dict[str, Any]]:
    events_file = "tests/files/test_events_format.yaml"
    return yaml.full_load(open(events_file, "r"))

@pytest.fixture
def exp_PostEvents() -> List[Dict[str, Any]]:
    events_file = "tests/files/test_post_events_format.yaml"
    return yaml.full_load(open(events_file, "r"))
