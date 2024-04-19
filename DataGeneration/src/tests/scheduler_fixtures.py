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

from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.core.scheduler import scheduler
from Dkr5G.src.core.scheduler import Event

@pytest.fixture
def test_scheduler():
    events = yaml.full_load(open("tests/files/test_events.yaml", "r"))
    sch = scheduler([], )
    sch.extend([Event(evt) for evt in events[s.events_key]])
    return sch

@pytest.fixture
def empty_scheduler():
    return scheduler([])

@pytest.fixture
def test_events():
    events = yaml.full_load(open("tests/files/test_events.yaml", "r"))
    return [Event(evt) for evt in events[s.events_key]]
