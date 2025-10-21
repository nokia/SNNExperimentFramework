# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import os
import sys

from SNN2.src.io.utils import redirect

class TestIoUtils():

    def test_redirect(self):
        dst = os.devnull
        f = redirect(dst)
        assert sys.stdout.name == dst
        f.close()
