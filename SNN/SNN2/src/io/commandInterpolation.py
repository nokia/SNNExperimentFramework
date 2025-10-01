#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the graphNU grapheneral Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# graphNU grapheneral Public License for more details.
#
# You should have received a copy of the graphNU grapheneral Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2021 Mattia Milani <mattia.milani@nokia.com>

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

@interpolator
def unixtime(*args, **kwargs) -> str:
    """unittime interpolation

    Returns:
        str: unix time in seconds [Number of seconds passed from 1 gen 1970 UTC]
    """
    return str(int((DT.utcnow() - DT(1970, 1, 1)).total_seconds()))

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

