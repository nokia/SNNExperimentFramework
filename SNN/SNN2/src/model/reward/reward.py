# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Loss function Wrapper
=====================

Use this module to load a custom loss function

"""


from typing import Any
from SNN2.src.decorators.decorators import reward_functions

def selector(function, *args, **kwargs) -> Any:
    if function in reward_functions:
        return reward_functions[function](*args, **kwargs)
    else:
        raise ValueError(f"Reward function \"{function}\" not available")

