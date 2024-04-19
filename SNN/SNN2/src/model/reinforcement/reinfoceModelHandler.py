# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from SNN2.src.decorators.decorators import RLModelHandlers

def ReinforceModelHandler(handler, *args, **kwargs):
    if handler in RLModelHandlers:
        return RLModelHandlers[handler](*args, **kwargs)
    else:
        print(RLModelHandlers)
        raise ValueError(f"RL Model handler \"{handler}\" not available")
