# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Graph fixtures
==============

Use this module to manage the fixtures for the configuration tests
"""

import pytest
import yaml

from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.core.graphHandler import GraphHandler as GH
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.io.files import FileHandler as FH


@pytest.fixture
def graph_file() -> str:
    return "tests/files/test_graph.graphml"

@pytest.fixture
def test_graph(graph_file: str, test_logger: LH) -> GH:
    return GH(FH(graph_file), test_logger)
