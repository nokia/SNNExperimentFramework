# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import datetime

from Dkr5G.src.core.commandInterpolation import fn as CI

class TestCommandInterpolation:

    def test_dateTime(self):
        assert CI("!datetime") == datetime.datetime.now().strftime('%d.%m.%Y-%H.%M.%S')
