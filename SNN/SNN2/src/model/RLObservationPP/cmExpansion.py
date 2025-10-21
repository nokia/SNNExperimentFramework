# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import tensorflow as tf

from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.decorators.decorators import RLObservationPP, f_logger, reward
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

@RLObservationPP
@f_logger
def cmExpansion(observation: tf.Tensor,
                labels_separated: Tuple[tf.Tensor, tf.Tensor],
                *args,
                decimals: Union[str, int] = 1,
                **kwargs) -> tf.Tensor:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    decimal_round: float = convert(decimals, str, int)

    write_msg(f"Expanding observation {observation}")
    conf_matrix = calculate_confusion_matrix(labels_separated, logger=logger)
    write_msg(f"Confusion matrix: {conf_matrix}", level=LH.DEBUG)
    tf_cm = tf.convert_to_tensor([*conf_matrix.values()])
    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    tf_cm = tf.cast(tf_cm, float)
    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    tf_cm = tf_cm/tf.math.reduce_sum(tf_cm)
    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)

    observation_expanded: tf.Tensor = tf.concat([observation, tf_cm], 0)
    observation_expanded = tf.cast(observation_expanded, tf.float64)
    observation_expanded = my_tf_round(observation_expanded, decimals=decimal_round)


    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    write_msg(f"Expanded observation: {observation_expanded}")
    return observation_expanded

@RLObservationPP
@f_logger
def cmExpansionNorm(observation: tf.Tensor,
                    labels_separated: Tuple[tf.Tensor, tf.Tensor],
                    *args,
                    current_step: Optional[int] = None,
                    max_steps: int = 1000,
                    decimals: Union[str, int] = 1,
                    max_param: int = 100,
                    **kwargs) -> Tuple[tf.Tensor, Dict[str, int]]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    decimal_round: float = convert(decimals, str, int)

    observation = observation/max_param
    cnct = [observation]

    write_msg(f"Expanding observation {observation}")
    conf_matrix = calculate_confusion_matrix(labels_separated, logger=logger)
    write_msg(f"Confusion matrix: {conf_matrix}", level=LH.DEBUG)
    tf_cm = tf.convert_to_tensor([*conf_matrix.values()])
    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    tf_cm = tf.cast(tf_cm, float)
    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    tf_cm = tf_cm/tf.math.reduce_sum(tf_cm)
    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    cnct.append(tf_cm)

    if current_step is not None:
        current_step = round(current_step / max_steps, decimal_round)
        current_step_tf = tf.convert_to_tensor([current_step], dtype=tf.float32)
        write_msg(f"Current step a tensor: {current_step_tf}", level=LH.DEBUG)
        cnct.append(current_step_tf)

    observation_expanded: tf.Tensor = tf.concat(cnct, 0)
    observation_expanded = tf.cast(observation_expanded, tf.float64)
    observation_expanded = my_tf_round(observation_expanded, decimals=decimal_round)

    write_msg(f"Confusion matrix as a tensor: {tf_cm}", level=LH.DEBUG)
    write_msg(f"Expanded observation: {observation_expanded}")
    return observation_expanded, conf_matrix

@RLObservationPP
@f_logger
def zoneExpansion(observation: tf.Tensor,
                  labels_separated: Tuple[tf.Tensor, tf.Tensor],
                  *args,
                  current_step: Optional[int] = None,
                  **kwargs) -> tf.Tensor:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    write_msg(f"Expanding observation {observation}")

    cnct = [observation]

    conf_matrix = calculate_confusion_matrix(labels_separated, logger=logger)
    weight_acc = weightAccuracy(conf_matrix)
    current_zone = find_zone(weight_acc)
    current_zone_f = tf.convert_to_tensor([current_zone])
    current_zone_f = tf.cast(current_zone_f, tf.float32)
    cnct.append(current_zone_f)

    if current_step is not None:
        current_step_tf = tf.convert_to_tensor([current_step], dtype=tf.float32)
        write_msg(f"Current step a tensor: {current_step_tf}", level=LH.DEBUG)
        cnct.append(current_step_tf)

    observation_expanded: tf.Tensor = tf.concat(cnct, 0)
    observation_expanded = tf.cast(observation_expanded, tf.float64)
    observation_expanded = my_tf_round(observation_expanded, decimals=1)

    write_msg(f"Expanded observation: {observation_expanded}")
    return observation_expanded

@RLObservationPP
@f_logger
def zoneExpansionNorm(observation: tf.Tensor,
                      labels_separated: Tuple[tf.Tensor, tf.Tensor],
                      *args,
                      current_step: Optional[int] = None,
                      max_steps: int = 1000,
                      max_zone: int = 7,
                      max_param: int = 100,
                      **kwargs) -> tf.Tensor:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    write_msg(f"Expanding observation {observation}")

    observation = observation/max_param

    cnct = [observation]

    conf_matrix = calculate_confusion_matrix(labels_separated, logger=logger)
    weight_acc = weightAccuracy(conf_matrix)
    current_zone = find_zone(weight_acc)
    current_zone = round(current_zone / max_zone, 4)
    current_zone_f = tf.convert_to_tensor([current_zone])
    current_zone_f = tf.cast(current_zone_f, tf.float32)
    cnct.append(current_zone_f)

    if current_step is not None:
        current_step = round(current_step / max_steps, 4)
        current_step_tf = tf.convert_to_tensor([current_step], dtype=tf.float32)
        write_msg(f"Current step a tensor: {current_step_tf}", level=LH.DEBUG)
        cnct.append(current_step_tf)

    observation_expanded: tf.Tensor = tf.concat(cnct, 0)
    observation_expanded = tf.cast(observation_expanded, tf.float64)
    observation_expanded = my_tf_round(observation_expanded, decimals=4)

    write_msg(f"Expanded observation: {observation_expanded}")
    return observation_expanded

