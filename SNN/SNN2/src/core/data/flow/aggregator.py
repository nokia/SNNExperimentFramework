# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class module
============

Use this module to manage a flow objects.
"""

from SNN2.src.decorators.decorators import flows

def Flow_Selector(function, *args, **kwargs):
    if function in flows.keys():
        return flows[function](*args, **kwargs)
    raise ValueError(f"Flow \"{function}\" not available, current available callbacks: {list(flows.keys())}")
