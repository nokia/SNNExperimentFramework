# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from Dkr5G.src.core.graphHandler import GraphHandler as GH
from Dkr5G.src.util.strings import strings as s

class TestGRaphHandler:

    @pytest.mark.parametrize(("obj", "expected"), [
            ("{test[ipv4]}", "192.168.11.10")
        ])
    def test_objEval(self, exp_graph, exp_env, exp_io, obj, expected):
        assert exp_graph.evaluateObj(obj, exp_env, exp_io) == expected
