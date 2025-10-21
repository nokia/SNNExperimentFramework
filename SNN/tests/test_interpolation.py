# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


from datetime import datetime
import tensorflow as tf
import pytest

from SNN2.src.io.commandInterpolation import fn as CI

class TestInterpolation():


    def test_dateTime(self) -> None:
        assert CI("!datetime") == datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
