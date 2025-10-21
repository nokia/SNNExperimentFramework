# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import tensorflow as tf

from SNN2.src.decorators.decorators import c_logger, reward_cls

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from SNN2.src.model.reward.accuracyMap import weightAccuracy
from SNN2.src.io.logger import LogHandler as LH

def convert(val: Any, input_type: Type, Transformer: Callable) -> Any:
    if isinstance(val, input_type):
        val = Transformer(val)
    return val

def normalize(cm: Dict[str, int]) -> Dict[str, float]:
    total = sum(cm.values())
    return {k: v/total for k, v in cm.items()}

@reward_cls
@c_logger
class comulativeAccuracy:

    @tf.function
    def aggregate(self, agg, x):
        return self.gamma*agg + x

    def discounted_sum(self, x: tf.Tensor) -> tf.Tensor:
        self.write_msg(f"Requiring discounted sum for {x}", level=LH.DEBUG)
        return tf.scan(self.aggregate, x)

    def __init__(self, *args, weight_acc_kwargs: Dict[str, int] = {
            "TP_factor": 1,
            "FP_factor": 2,
            "TN_factor": 1,
            "FN_factor": 1,
            "Undecided_factor": 1,
        },
        gamma: float = 0.9,
        **kwargs) -> None:
        self.__state = None
        self.weight_acc_kwargs = weight_acc_kwargs
        self.normal_acc_kwargs = {key: 1 for key in weight_acc_kwargs.keys()}
        self.gamma = gamma

    def update(self, cf_matrix, *args, **kwargs):
        weight_acc = round(weightAccuracy(cf_matrix, **self.weight_acc_kwargs), 4)
        normal_acc = round(weightAccuracy(cf_matrix, **self.normal_acc_kwargs), 4)
        self.write_msg(f"Weighted accuracy: {weight_acc}")
        self.write_msg(f"uniform accuracy: {normal_acc}")
        if self.__state is None:
            self.__state = {key: [value] for key, value in cf_matrix.items()}
            self.__state["WeightedAccuracy"] = [weight_acc]
            self.__state["normalAccuracy"] = [normal_acc]
        else:
            cf_matrix["WeightedAccuracy"] = weight_acc
            cf_matrix["normalAccuracy"] = normal_acc
            for key, value in cf_matrix.items():
                self.__state[key].append(value)
        self.write_msg(f"New state: {self.__state}")

    def state(self) -> Union[Dict[str, Any], None]:
        return self.__state

    def compute(self, *args, previous_comAccuracy: Optional[float] = None,**kwargs) -> Tuple[float, float]:
        assert self.__state is not None
        assert previous_comAccuracy is not None
        m_t = self.discounted_sum(tf.convert_to_tensor(self.__state["WeightedAccuracy"])).numpy()[-1]
        self.write_msg(f"M_t+1 = {m_t}")
        if m_t > previous_comAccuracy:
            return 1, round(m_t, 4)
        return -1, round(m_t, 4)

    def reset(self) -> None:
        self.accuracy = []
        self.__state = None

@reward_cls
@c_logger
class comulativeAccNG:

    @tf.function
    def aggregate(self, agg, x):
        return self.gamma*agg + x

    def discounted_sum(self, x: tf.Tensor) -> tf.Tensor:
        self.write_msg(f"Requiring discounted sum for {x}", level=LH.DEBUG)
        return tf.scan(self.aggregate, x)

    def __init__(self, *args, weight_acc_kwargs: Dict[str, int] = {
            "TP_factor": 1,
            "FP_factor": 2,
            "TN_factor": 1,
            "FN_factor": 1,
            "Undecided_factor": 1,
        },
        gamma: float = 0.5,
        mt_round: float = 4,
        **kwargs) -> None:
        self.__state = None
        self.weight_acc_kwargs = weight_acc_kwargs
        self.normal_acc_kwargs = {key: 1 for key in weight_acc_kwargs.keys()}
        self.gamma = gamma
        self.mt_round = mt_round

    def update(self, cf_matrix, *args, **kwargs):
        weight_acc = round(weightAccuracy(cf_matrix, **self.weight_acc_kwargs), 4)
        normal_acc = round(weightAccuracy(cf_matrix, **self.normal_acc_kwargs), 4)
        self.write_msg(f"Weighted accuracy: {weight_acc}")
        self.write_msg(f"uniform accuracy: {normal_acc}")
        if self.__state is None:
            self.__state = {key: [value] for key, value in cf_matrix.items()}
            self.__state["WeightedAccuracy"] = [weight_acc]
            self.__state["normalAccuracy"] = [normal_acc]
        else:
            cf_matrix["WeightedAccuracy"] = weight_acc
            cf_matrix["normalAccuracy"] = normal_acc
            for key, value in cf_matrix.items():
                self.__state[key].append(value)
        self.write_msg(f"New state: {self.__state}")

    def state(self) -> Union[Dict[str, Any], None]:
        return self.__state

    def compute(self, *args, previous_comAccuracy: Optional[float] = None,**kwargs) -> Tuple[float, float]:
        assert self.__state is not None
        assert previous_comAccuracy is not None
        m_t = self.discounted_sum(tf.convert_to_tensor(self.__state["WeightedAccuracy"])).numpy()[-1]
        m_t = round(m_t, self.mt_round)
        self.write_msg(f"M_t+1 = {m_t}")
        return m_t, m_t

    def reset(self) -> None:
        self.accuracy = []
        self.__state = None

@reward_cls
@c_logger
class comulativeAccSign:

    @tf.function
    def aggregate(self, agg, x):
        return self.gamma*agg + x

    def discounted_sum(self, x: tf.Tensor) -> tf.Tensor:
        self.write_msg(f"Requiring discounted sum for {x}", level=LH.DEBUG)
        return tf.scan(self.aggregate, x)

    def __init__(self, *args, weight_acc_kwargs: Dict[str, int] = {
            "TP_factor": 1,
            "FP_factor": 2,
            "TN_factor": 1,
            "FN_factor": 1,
            "Undecided_factor": 1,
        },
        gamma: float = 0.5,
        mt_round: float = 4,
        mt_delta: float = 0.000,
        **kwargs) -> None:
        self.__state = None
        self.weight_acc_kwargs = weight_acc_kwargs
        self.normal_acc_kwargs = {key: 1 for key in weight_acc_kwargs.keys()}
        self.gamma = gamma
        self.mt_round = mt_round
        self.mt_delta = mt_delta

    def update(self, cf_matrix, *args, **kwargs):
        weight_acc = round(weightAccuracy(cf_matrix, **self.weight_acc_kwargs), 4)
        normal_acc = round(weightAccuracy(cf_matrix, **self.normal_acc_kwargs), 4)
        self.write_msg(f"Weighted accuracy: {weight_acc}")
        self.write_msg(f"uniform accuracy: {normal_acc}")
        if self.__state is None:
            self.__state = {key: [value] for key, value in cf_matrix.items()}
            self.__state["WeightedAccuracy"] = [weight_acc]
            self.__state["normalAccuracy"] = [normal_acc]
        else:
            cf_matrix["WeightedAccuracy"] = weight_acc
            cf_matrix["normalAccuracy"] = normal_acc
            for key, value in cf_matrix.items():
                self.__state[key].append(value)
        self.write_msg(f"New state: {self.__state}")

    def state(self) -> Union[Dict[str, Any], None]:
        return self.__state

    def compute(self, *args, previous_comAccuracy: Optional[float] = None,**kwargs) -> Tuple[float, float]:
        assert self.__state is not None
        assert previous_comAccuracy is not None
        m_t = self.discounted_sum(tf.convert_to_tensor(self.__state["WeightedAccuracy"])).numpy()[-1]
        m_t = round(m_t, self.mt_round)
        self.write_msg(f"M_t+1 = {m_t}")
        if m_t <= previous_comAccuracy+self.mt_delta and m_t >= previous_comAccuracy-self.mt_delta:
            return 0, m_t
        if m_t > previous_comAccuracy + self.mt_delta:
            return 1, m_t
        if m_t < previous_comAccuracy - self.mt_delta:
            return -1, m_t
        raise Exception("this should not happen!")

    def reset(self) -> None:
        self.accuracy = []
        self.__state = None
