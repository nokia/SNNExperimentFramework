# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from SNN2.src.decorators.decorators import ModelManagers

def selector(handler, *args, **kwargs):
    if handler in ModelManagers:
        return ModelManagers[handler](*args, **kwargs)
    else:
        raise ValueError(f"Model Manager \"{handler}\" not available")
