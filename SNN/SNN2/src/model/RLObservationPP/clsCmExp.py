# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import tensorflow as tf

from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.decorators.decorators import RLObservationPP_cls, c_logger, reward, f_logger
from SNN2.src.model.reward.accuracyMap import find_zone, weightAccuracy

def convert(val: Any, input_type: Type, Transformer: Callable) -> Any:
    if isinstance(val, input_type):
        val = Transformer(val)
    return val

@f_logger
def calculate_confusion_matrix(labels_separated: Tuple[tf.Tensor, tf.Tensor],
                               **kwargs) -> Dict[str, int]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    correct_l, wrong_l = labels_separated
    tp=  tf.size(tf.where(correct_l == 0)).numpy()
    fp = tf.size(tf.where(wrong_l == 0)).numpy()
    tn = tf.size(tf.where(correct_l == 1)).numpy()
    fn = tf.size(tf.where(wrong_l == 1)).numpy()
    u = tf.size(tf.where(wrong_l == 2)).numpy()
    matrix = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "U": u}
    write_msg(f"Confusion matrix computed: {matrix}")
    return matrix

def my_tf_round(x: tf.Tensor, decimals: int = 0) -> tf.Tensor:
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

@RLObservationPP_cls
@c_logger
class cmExp_movingAvg:

    def __init__(self, *args,
                 max_steps: int = 1000,
                 decimals: Union[str, int] = 1,
                 max_param: int = 100,
                 moving_avg_window: Union[str, int] = 10,
                 **kwargs) -> None:
        self.max_steps = max_steps
        self.decimals = convert(decimals, str, int)
        self.max_param = convert(max_param, str, int)
        self.cm_matrix_wdw = None
        self.window_limit = convert(moving_avg_window, str, int)

    def __call__(self,
                 observation: tf.Tensor,
                 labels_separated: Tuple[tf.Tensor, tf.Tensor],
                 *args,
                 current_step: Optional[int] = None,
                 **kwargs) -> Tuple[tf.Tensor, Dict[str, int]]:
        observation = observation/self.max_param
        cnct = [observation]

        self.write_msg(f"Expanding observation {observation}")
        conf_matrix = calculate_confusion_matrix(labels_separated, logger=self.logger)

        self.write_msg(f"Confusion matrix: {conf_matrix}", level=LH.DEBUG)
        tf_cm = tf.convert_to_tensor([*conf_matrix.values()])
        self.write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
        tf_cm = tf.cast(tf_cm, float)
        self.write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
        tf_cm = tf_cm/tf.math.reduce_sum(tf_cm)
        self.write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
        cnct.append(tf_cm)

        if self.cm_matrix_wdw is None:
            self.cm_matrix_wdw = {key: [value] for key, value in zip(conf_matrix.keys(), tf_cm.numpy())}
        else:
            for key, value in zip(conf_matrix.keys(), tf_cm.numpy()):
                self.cm_matrix_wdw[key].append(value)
                if len(self.cm_matrix_wdw[key]) > self.window_limit:
                    self.cm_matrix_wdw[key] = self.cm_matrix_wdw[key][-self.window_limit:]

        if current_step is not None:
            current_step = round(current_step / self.max_steps, self.decimals)
            current_step_tf = tf.convert_to_tensor([current_step], dtype=tf.float32)
            self.write_msg(f"Current step a tensor: {current_step_tf}", level=LH.DEBUG)
            cnct.append(current_step_tf)

        if self.cm_matrix_wdw is not None:
            all_values = np.array(list(self.cm_matrix_wdw.values()))
            all_avg = np.mean(all_values, axis=1)
            variation = tf_cm.numpy() - all_avg
            self.write_msg(f"Variation as array: {variation}")
            variation_tf = tf.convert_to_tensor(variation, dtype=tf.float32)
            cnct.append(variation_tf)


        observation_expanded: tf.Tensor = tf.concat(cnct, 0)
        observation_expanded = tf.cast(observation_expanded, tf.float64)
        observation_expanded = my_tf_round(observation_expanded, decimals=self.decimals)

        self.write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
        self.write_msg(f"Expanded observation: {observation_expanded}")
        return observation_expanded, conf_matrix

    def reset(self) -> None:
        self.cm_matrix_wdw = None

