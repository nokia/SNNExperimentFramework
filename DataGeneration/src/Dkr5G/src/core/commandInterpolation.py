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

import functools
import datetime

from typing import Callable, List

functions = []

def fn_wrapper(func: Callable):
    """fn_wrapper.
    Function wrapper for commands fn
    Every function decorated with @fn_wrapper will be included
    into the functions list
    The function name must be `interpolate_<name>_fn`.
    The function name must not include `_`
    The functions can receive arguments and kwargs but must return
    a single string

    Parameters
    ----------
    func : Callable
        func
    """
    functions.append(func.__qualname__.split('_')[1])
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@fn_wrapper
def interpolate_datetime_fn(*args, **kwargs) -> str:
    return datetime.datetime.now().strftime('%d.%m.%Y-%H.%M.%S')

def fn(function, *args, **kwargs):
    if not isinstance(function, str) or \
        function[0] != "!" or \
        not function[1::] in functions:
        raise ValueError(f"Command \"{function[1::]}\" not available remember to escape the commands with a '!'")
    return eval("interpolate_" + function[1::] + "_fn")(*args, **kwargs)
