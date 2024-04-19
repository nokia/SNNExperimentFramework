#!/usr/bin/env python
# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Loss function Wrapper
=====================

Use this module to load a custom loss function

"""


from SNN2.src.decorators.decorators import loss_functions

def fn(function, *args, **kwargs):
    if function in loss_functions:
        return loss_functions[function](*args, **kwargs)
    else:
        raise ValueError(f"Loss \"{function}\" not available")

