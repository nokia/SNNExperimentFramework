# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


import pytest

from SNN2.src.io.logger import LogHandler as LH

class TestLog():

    def test_init(self, log_file):
        l = LH(str(log_file), LH.DEBUG)
        assert isinstance(l, LH)
