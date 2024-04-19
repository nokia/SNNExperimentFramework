#!/usr/bin/env python
# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Loss function Wrapper
=====================

Use this module to load a custom loss function

"""

import ast
import functools

from SNN2.src.decorators.decorators import callbacks, ccb
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TensorBoard

from typing import List, Callable, Tuple, Dict, Union


@ccb
def csvLogger(*args, append: Union[str, bool] = False, **kwargs) -> CSVLogger:
    if isinstance(append, str):
        append= ast.literal_eval(append)
    kwargs["append"] = append
    return CSVLogger(*args, **kwargs)

@ccb
def earlyStopping(*args,
                  patience: Union[str, int] = 10,
                  min_delta: Union[str, float] = 0.0001,
                  restore_best_weights: Union[str, bool] = True,
                  **kwargs) -> EarlyStopping:
    if isinstance(patience, str):
        patience = int(patience)
    if isinstance(min_delta, str):
        min_delta= float(min_delta)
    restore_best_weights = restore_best_weights == "True"

    return EarlyStopping(*args, patience=patience,
                         min_delta=min_delta,
                         restore_best_weights=restore_best_weights, **kwargs)

@ccb
def modelCheckpoint(*args,
                    save_weights_only: Union[str, bool] = True,
                    save_best_only: Union[str, bool] = True,
                    **kwargs) -> EarlyStopping:
    save_weights_only = save_weights_only == "True" if isinstance(save_weights_only, str) else save_weights_only
    save_best_only = save_best_only == "True" if isinstance(save_best_only, str) else save_best_only
    return ModelCheckpoint(*args, save_weights_only=save_weights_only, save_best_only=save_best_only, **kwargs)

@ccb
def tensorBoard(*args, profile_batch: Union[str, int, Tuple[int, int]] = 0, **kwargs) -> CSVLogger:
    if isinstance(profile_batch, str):
        profile_batch = ast.literal_eval(profile_batch)
    kwargs["profile_batch"] = profile_batch
    return TensorBoard(*args, **kwargs)

def Callback_Selector(function, *args, **kwargs):
    if function in callbacks.keys():
        return callbacks[function](*args, **kwargs)
    else:
        raise ValueError(f"Callback \"{function}\" not available, current available callbacks: {list(callbacks.keys())}")

