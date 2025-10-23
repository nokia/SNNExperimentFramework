#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the graphNU grapheneral Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# graphNU grapheneral Public License for more details.
#
# You should have received a copy of the graphNU grapheneral Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2021 Mattia Milani <mattia.milani@nokia.com>

"""
Fit function Wrapper
====================


"""

import os
import numpy as np
import functools
import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import History

from SNN2.src.util.helper import chain_histories
from SNN2.src.io.pickleHandler import PickleHandler as PH
from SNN2.src.decorators.decorators import f_logger, fitMethods, fitMethod

from typing import Any, List, Callable, Tuple, Union, Generator, Optional

@fitMethod
def stepFit(model,
            callbacks: List[Callback],
            data: tf.data.Dataset,
            start: int,
            *args,
            verbose: Optional[Union[str, int]] = 1,
            **kwargs):
    if isinstance(verbose, str) and verbose != "auto":
        verbose = int(verbose)
    return model.model.fit(data, *args,
                     callbacks=callbacks,
                     initial_epoch=start,
                     verbose=verbose,
                     **kwargs)

@fitMethod
@f_logger
def default(model,
            callbacks: List[Callback],
            data: tf.data.Dataset,
            *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]
    history = stepFit(model, callbacks, data, 0, *args, **kwargs)
    return history

@fitMethod
def grays(model,
          callbacks: List[Callback],
          data: tf.data.Dataset,
          grayAction: Generator,
          epochs: int = 0,
          skip: Union[int, str] = 0,
          step: Union[int, str] = 1,
          *args, **kwargs) -> History:
    if isinstance(skip, str):
        skip = int(skip)
    if isinstance(step, str):
        step = int(step)

    if skip >= epochs:
        return default(model, callbacks, data, *args, epochs=epochs, **kwargs)

    histories = []
    start = 0
    if skip > 0:
        histories.append(
                    stepFit(model, callbacks, data, start, *args, epochs=skip, **kwargs)
                )
        start += skip
        data = next(grayAction)[0]

    for start_point in range(start, epochs, step):
        start = start_point + step
        if start > epochs:
            start = epochs

        histories.append(
                    stepFit(model, callbacks, data, start_point, *args, epochs=start, **kwargs)
                )
        data = next(grayAction)[0]

    return chain_histories(histories)

@fitMethod
def graysAfterFit(model,
                  callbacks: List[Callback],
                  data: tf.data.Dataset,
                  grayAction: Generator,
                  reps: int = 10,
                  *args, **kwargs) -> History:
    if isinstance(reps, str):
        reps = int(reps)

    histories = []
    for _ in range(reps):
        histories.append(default(model, callbacks, data, *args, **kwargs))
        data = next(grayAction)[0]
        # result = model.model.evaluate(data, callbacks=callbacks, return_dict=True)
        # tf.print(f"Results evaluate: {result}")
        model.compile()

    return chain_histories(histories)

@fitMethod
@f_logger
def graysAfterFit_netReset(model,
                           callbacks: List[Callback],
                           data: tf.data.Dataset,
                           grayAction: Generator,
                           reps: int = 10,
                           active_reset: bool = False,
                           *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]
    write_msg(f"GRAYSAFTER FIT for the actuall SNN training, {reps} number of reps required")

    if isinstance(reps, str):
        reps = int(reps)

    model_weights = model.get_weights()
    histories = []
    for i in range(reps):
        write_msg(f"Repetition {i}")
        for cb in callbacks:
            if type(cb).__name__ == "augmented_cls" and cb.c_passed_name == "RL_partial_env_manager":
                write_msg(f"Cycle {i} is the last one, must communicate this to the callback")
                cb.last_cycle = True
        histories.append(default(model, callbacks, data, *args, **kwargs))

        if i == reps-1:
            write_msg(f"Exit condition True")
            model.compile(active_reset=active_reset)
            break
        data = next(grayAction)
        model.set_weights(model_weights)
        model.compile(active_reset=active_reset)

    return chain_histories(histories)

@fitMethod
@f_logger
def graysAfterFitCURE_netReset(model,
                               callbacks: List[Callback],
                               data: tf.data.Dataset,
                               grayAction: Generator,
                               reps: int = 10,
                               active_reset: bool = False,
                               checkpoint_file: Optional[str] = None,
                               *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]
    write_msg(f"GRAYSAFTER FIT for the actuall SNN training, {reps} number of reps required")

    if isinstance(reps, str):
        reps = int(reps)

    # model.model_definition.load_weights(checkpoint_file)
    # print("TEMPORARY LOAD OF THE MODEL!!! this should be removed otherwise the starting point is always the same")
    model_weights = model.get_weights()
    histories = []
    for i in range(reps):
        write_msg(f"Repetition {i}")
        histories.append(default(model, callbacks, data, *args, **kwargs))

        if i == reps-1:
            write_msg("Exit condition True")
            model.compile(active_reset=active_reset)
            break
        data = next(grayAction)
        model.set_weights(model_weights)
        model.compile(active_reset=active_reset)

    return chain_histories(histories)

@fitMethod
@f_logger
def RLTrnFrz_graysAfterFit(model,
                           callbacks: List[Callback],
                           data: tf.data.Dataset,
                           grayAction: Generator,
                           rl_model,
                           reps: int = 10,
                           *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]

    if isinstance(reps, str):
        reps = int(reps)

    rl_model.trained = False

    model_weights = model.get_weights()

    train_history = graysAfterFit_netReset(model, callbacks, data, grayAction, reps=reps+round(reps/2), active_reset=True, *args, **kwargs)
    if rl_model.training and rl_model.use_stop_thresholds:
        write_msg(f"RL model training error")
        raise Exception(f"RL model training error")

    rl_model.save(obj=rl_model.emb)
    rl_model.trained=True
    model.set_weights(model_weights)
    inference_history = graysAfterFit_netReset(model, callbacks, data, grayAction, reps=reps, *args, **kwargs)

    return chain_histories([train_history, inference_history])

@fitMethod
@f_logger
def graysAfterFit_netReset_RLLast(model,
                                  callbacks: List[Callback],
                                  data: tf.data.Dataset,
                                  grayAction: Generator,
                                  reps: int = 10,
                                  active_reset: bool = False,
                                  rl_controller_cb: Optional[Any] = None,
                                  *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]
    if isinstance(reps, str):
        reps = int(reps)

    pre_model_weights = model.get_weights()
    if rl_controller_cb is None:
        for cb in callbacks:
            if type(cb).__name__ == "augmented_cls" and cb.c_passed_name == "RL_partial_env_manager":
                rl_controller_cb = cb

    for i in range(reps):
        write_msg(f"Executing cycle: {i}")

        if i == reps-1 and rl_controller_cb is not None:
            write_msg(f"Cycle {i} is the last one, must communicate this to the callback")
            rl_controller_cb.last_cycle = True

        default(model, callbacks, data, *args, **kwargs)

        if i == reps-1 or (not rl_controller_cb is None and rl_controller_cb.training_interrupted):
            write_msg(f"Interrupting the training episode, i: {i}")
            if not rl_controller_cb is None:
                write_msg(f"The controller flag: {rl_controller_cb.training_interrupted}")
            return None

        data = next(grayAction)
        write_msg("Executing weights reset and compile")
        model.set_weights(pre_model_weights)
        model.compile(active_reset=active_reset, force_keeping=True)

    return None

@fitMethod
@f_logger
def RLTrnFrz_External_graysAfterFit(model,
                                    callbacks: List[Callback],
                                    data: tf.data.Dataset,
                                    grayAction: Generator,
                                    rl_model,
                                    slReps: int = 2,
                                    slInRLReps: int = 10,
                                    rlReps: int = 10,
                                    correct_exp_threshold: int = 1,
                                    last_exp_to_consider: int = 10,
                                    *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]

    slReps = int(slReps) if isinstance(slReps, str) else slReps
    rlReps = int(rlReps) if isinstance(rlReps, str) else rlReps

    rl_model.trained = False

    model_weights = model.get_weights()
    rl_controller_cb = None
    for cb in callbacks:
        if type(cb).__name__ == "augmented_cls" and cb.c_passed_name == "RL_partial_env_manager":
            rl_controller_cb = cb

    correct_exp_queue = []

    for i in range(rlReps):
        write_msg(f"Executing RL training cycle {i}")
        graysAfterFit_netReset_RLLast(model, callbacks,
                                      data, grayAction,
                                      reps=slInRLReps+1,
                                      active_reset=True,
                                      logger=logger,
                                      rl_controller_cb=rl_controller_cb,
                                      *args, **kwargs)

        model.set_weights(model_weights)
        model.compile(active_reset=True)
        if not rl_controller_cb is None:
            rl_controller_cb.training_interrupted = False
        if not rl_model.training:
            write_msg(f"The RL model has reached a good training cycle")
            correct_exp_queue.append(1)
        else:
            write_msg(f"The RL model failed this training cycle")
            correct_exp_queue.append(0)
        if len(correct_exp_queue) > last_exp_to_consider:
            correct_exp_queue = correct_exp_queue[-last_exp_to_consider:]

        if sum(correct_exp_queue) >= correct_exp_threshold:
            write_msg(f"SUCCESS! {sum(correct_exp_queue)} training cycles ended with an AVG reward higher than the threshold")
            print(f"SUCCESS! {sum(correct_exp_queue)} training cycles ended with an AVG reward higher than the threshold")
            print(f"Solved after {i} episodes")
            break
        rl_model.training = True


    write_msg(f"The training of the RL network is concluded")
    if rl_model.training and rl_model.use_stop_thresholds:
        write_msg(f"RL model training not at the required threshold")
        print(f"RL model training not at the required threshold")
        # raise Exception(f"RL model training error")

    rl_model.save(obj=rl_model.emb)
    rl_model.trained=True
    if not rl_controller_cb is None:
        rl_controller_cb.last_cycle = True
        rl_controller_cb.exit_function = None

    graysAfterFit_netReset(model, callbacks, data, grayAction,
                           reps=slReps, logger=logger,
                           *args, **kwargs)
    return None

def get_rl_controller(callbacks: List[Callback],
                      cls_name: str = "augmented_cls",
                      controller_name: str = "RL_partial_env_manager") -> Tuple[int, Callback]:
    for i, cb in enumerate(callbacks):
        if type(cb).__name__ == cls_name and cb.c_passed_name == controller_name:
            return i, cb
    return -1, None

@fitMethod
@f_logger
def graysAfterFit_netReset_RLnoFirst(model,
                                     callbacks: List[Callback],
                                     data: tf.data.Dataset,
                                     grayAction: Generator,
                                     reps: int = 10,
                                     active_reset: bool = False,
                                     *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]
    if isinstance(reps, str):
        reps = int(reps)

    pre_model_weights = model.get_weights()
    rl_controller_cb_i, rl_controller_cb = get_rl_controller(callbacks)
    assert rl_controller_cb_i > -1 and not rl_controller_cb is None

    del callbacks[rl_controller_cb_i]

    for i in range(reps):
        write_msg(f"Executing cycle: {i}")

        if i == reps-1 and rl_controller_cb is not None:
            write_msg(f"Cycle {i} is the last one, must communicate this to the callback")
            rl_controller_cb.last_cycle = True

        default(model, callbacks, data, *args, **kwargs)

        if i == 0:
            callbacks.append(rl_controller_cb)

        if i == reps-1 or (not rl_controller_cb is None and rl_controller_cb.training_interrupted):
            write_msg(f"Interrupting the training episode, i: {i}")
            if not rl_controller_cb is None:
                write_msg(f"The controller flag: {rl_controller_cb.training_interrupted}")
            return None

        data = next(grayAction)
        write_msg("Executing weights reset and compile")
        model.set_weights(pre_model_weights)
        model.compile(active_reset=active_reset, force_keeping=True)

    return None

@fitMethod
@f_logger
def RLTrnFrz_External_AfterTrivial(model,
                                   callbacks: List[Callback],
                                   data: tf.data.Dataset,
                                   grayAction: Generator,
                                   rl_model,
                                   slReps: int = 2,
                                   slInRLReps: int = 10,
                                   rlReps: int = 10,
                                   *args, **kwargs) -> History:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    del kwargs["logger"]
    del kwargs["write_msg"]

    slReps = int(slReps) if isinstance(slReps, str) else slReps
    rlReps = int(rlReps) if isinstance(rlReps, str) else rlReps

    rl_model.trained = False

    model_weights = model.get_weights()
    rl_controller_cb = None
    for cb in callbacks:
        if type(cb).__name__ == "augmented_cls" and cb.c_passed_name == "RL_partial_env_manager":
            rl_controller_cb = cb

    for i in range(rlReps):
        write_msg(f"Executing RL training cycle {i}")
        graysAfterFit_netReset_RLnoFirst(model, callbacks,
                                         data, grayAction,
                                         reps=slInRLReps+1,
                                         active_reset=True,
                                         logger=logger,
                                         *args, **kwargs)

        model.set_weights(model_weights)
        model.compile(active_reset=True)
        if not rl_controller_cb is None:
            rl_controller_cb.training_interrupted = False
        if not rl_model.training:
            write_msg(f"The RL model has reached a good training cycle, it's possible to interrupt the training")
            write_msg(f"Solved in {i} episdes")
            print(f"Solved in {i} episdes")
            break

    write_msg(f"The training of the RL network is concluded")
    if rl_model.training and rl_model.use_stop_thresholds:
        write_msg(f"RL model training error")
        print(f"RL model training error")
        # raise Exception(f"RL model training error")

    rl_model.save(obj=rl_model.emb)
    rl_model.trained=True
    if not rl_controller_cb is None:
        rl_controller_cb.last_cycle = True
        rl_controller_cb.exit_function = None

    graysAfterFit_netReset(model, callbacks, data, grayAction,
                           reps=slReps, logger=logger,
                           *args, **kwargs)

    return None

@fitMethod
def graysAfterFit_modelCheckpoint(model,
                  callbacks: List[Callback],
                  data: tf.data.Dataset,
                  grayAction: Generator,
                  reps: int = 10,
                  checkpoint_file: Optional[str] = None,
                  *args, **kwargs) -> History:
    if isinstance(reps, str):
        reps = int(reps)

    histories = []
    for _ in range(reps):
        histories.append(default(model, callbacks, data, *args, **kwargs))
        model.model.load_weights(checkpoint_file)
        data = next(grayAction)[0]
        model.compile()
        callbacks[-1].best = 0.

    return chain_histories(histories)

@fitMethod
def graysAfterFitReinforce(model,
                           reinforceModel,
                           callbacks: List[Callback],
                           data: tf.data.Dataset,
                           grayAction: Generator,
                           reps: int = 10,
                           *args, **kwargs) -> History:
    if isinstance(reps, str):
        reps = int(reps)

    histories = []
    for _ in range(reps):
        histories.append(default(model, callbacks, data, *args, **kwargs))
        data = next(grayAction)[0]
        model.compile()

    return chain_histories(histories)


def Fit_Selector(function, *args, **kwargs):
    if function in fitMethods.keys():
        return fitMethods[function](*args, **kwargs)
    else:
        raise ValueError(f"fit \"{function}\" not available")

