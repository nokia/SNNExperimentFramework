# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
redirect module
===============

Module used to set the redirect of the output in the program

"""

import sys

from typing import TextIO

def redirect(output: str) -> TextIO:
    f = open(output,"w")
    sys.stdout = f
    return f


