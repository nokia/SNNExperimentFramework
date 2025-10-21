# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import functools

from SNN2.src.model.custom.siamese import SiameseModel
from SNN2.src.decorators.decorators import models, cmodel

@cmodel
def none_customModel(*args, **kwargs) -> None:
    pass

def CustomModel_selector(obj, *args, **kwargs):
    if obj in models.keys():
        return models[obj](*args, **kwargs)
    else:
        raise ValueError(f"Model \"{obj}\" not available")


