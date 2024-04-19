# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import functools

from SNN2.src.decorators.decorators import embeddings

def Embedding_selector(obj, *args, **kwargs):
    if obj in embeddings.keys():
        return embeddings[obj](*args, **kwargs)
    else:
        raise ValueError(f"Embedding \"{obj}\" not available")

