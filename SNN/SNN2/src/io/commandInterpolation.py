#!/usr/bin/env python
# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Commands Interpolation Module
=============================

This module contains the command interpolation methods
that should been called when a command is required
inside an interpolation

"""

import re
import functools
from datetime import datetime as DT

from typing import Callable, List
from SNN2.src.decorators.decorators import interpolate_functions, interpolator

@interpolator
def datetime(*args, **kwargs) -> str:
    return DT.now().strftime('%d-%m-%Y_%H-%M-%S')

@interpolator
def date(*args, **kwargs) -> str:
    return DT.now().strftime('%d-%m-%Y')

def fn(function, *args, **kwargs):
    if isinstance(function, re.Match):
        function = function.group(0)
        function = function.replace("{", "")
        function = function.replace("}", "")

    if not isinstance(function, str) or \
        function[0] != "!" or \
        not function[1::] in interpolate_functions:
        raise ValueError(f"Command \"{function}\" not available remember to escape the commands with a '!'")
    return interpolate_functions[function[1::]](*args, **kwargs)

def str_interpolate(s: str, pattern: str = r"\{[^\}]*\}") -> str:
    return re.sub(pattern, fn, s)

