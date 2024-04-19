# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
progress bar module
===================

Module used to set the progress bar properly across the whole framework

"""

import os
from tqdm import tqdm

class pb:

    silent = False

    @classmethod
    def bar(cls, *args, **kwargs):
        if cls.silent:
            return tqdm(*args, file=open(os.devnull, "w"), **kwargs)
        return tqdm(*args, **kwargs)
