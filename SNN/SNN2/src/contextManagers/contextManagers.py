# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import functools
import inspect
from time import time
from contextlib import contextmanager
from typing import Callable, Dict, Any, Optional
from SNN2.src.decorators.decorators import f_logger

from SNN2.src.io.logger import LogHandler

def get_realName(obj: Any):
    if obj.__name__ == "augmented_cls":
        return obj.c_passed_name
    return obj.__qualname__


@contextmanager
@f_logger
def timeit_cnt(name, *args, **kwargs):
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]

    start_time = time()
    yield
    delta_t = time() - start_time
    write_msg(f"Context {name} finished in {int(delta_t*1000)} ms", LogHandler.DEBUG)
    print(f"Context {name} finished in {int(delta_t*1000)} ms")

