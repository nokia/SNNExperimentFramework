# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import os
import pytest


from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.io.configuration import ConfHandler
from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.logger import LogHandler as LH

class TestLogger:

    def test_logger(self, test_configuration):
        assert isinstance(test_configuration, ConfHandler)
        io = IOH.from_cfg(test_configuration)
        assert io["Log"] == "tests/files/test_log.log"
        assert FH.exists(io["Log"])
        logger = LH(io["Log"], LH.DEBUG)
        assert isinstance(logger, LH)
        os.remove(io["Log"])
        assert not FH.exists(io["Log"])

    def test_logger_level(self):
        assert LH.findLL(1) == LH.CRITICAL
        assert LH.findLL(2) == LH.ERROR
        assert LH.findLL(3) == LH.WARNING
        assert LH.findLL(4) == LH.INFO
        assert LH.findLL(5) == LH.DEBUG
        assert LH.findLL(16) == LH.DEBUG
