# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
import numpy as np
import tensorflow as tf

from SNN2.src.decorators.decorators import RLEnvExitFunction, c_logger

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

@RLEnvExitFunction
@c_logger
class default:

    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.write_msg("Calling the default Environment exit function")

@RLEnvExitFunction
@c_logger
class accEnvExit:

    def __init__(self,
                 *args,
                 acc_threshold: Union[float, str] = 0.5,
                 stp_threshold: Union[int, str] = 40,
                 expected_labels: Optional[tf.Tensor] = None,
                 **kwargs) -> None:
        self.acc_thr = ast.literal_eval(acc_threshold) if isinstance(acc_threshold, str) else acc_threshold
        self.stp_thr = ast.literal_eval(stp_threshold) if isinstance(stp_threshold, str) else stp_threshold
        self.counter = 0

        if expected_labels is None:
            raise Exception("the expected labels must be provided")
        self.expected_labels = expected_labels

    def get_correct_wrong(self, obtained: tf.Tensor, expected: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        correct = tf.reshape(tf.gather(obtained, tf.where(obtained == expected)), [-1])
        wrong = tf.reshape(tf.gather(obtained, tf.where(obtained != expected)), [-1])
        return correct, wrong

    def calculate_confusion_matrix(self, labels: tf.Tensor) -> Dict[str, int]:
        correct_l, wrong_l = self.get_correct_wrong(labels, self.expected_labels)
        tp=  tf.size(tf.where(correct_l == 0)).numpy()
        fp = tf.size(tf.where(wrong_l == 0)).numpy()
        tn = tf.size(tf.where(correct_l == 1)).numpy()
        fn = tf.size(tf.where(wrong_l == 1)).numpy()
        u = tf.size(tf.where(wrong_l == 2)).numpy()
        matrix = {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "U": u}
        self.write_msg(f"Confusion matrix computed: {matrix}")
        return matrix


    def __call__(self,
                 predictions: tf.Tensor,
                 current_state: np.ndarray,
                 current_step: int,
                 *args: Any, **kwds: Any) -> bool:
        self.write_msg(f"Calling the {self.c_passed_name} exit functhon")
        cm = self.calculate_confusion_matrix(predictions)
        acc = (cm["TP"] + cm["TN"])/sum(cm.values())
        self.write_msg(f"Accuracy calculated: {acc}")

        if acc < self.acc_thr:
            self.write_msg(f"The accuracy is lower than the threshold {self.acc_thr}")
            self.counter += 1
            self.write_msg(f"Current counter: {self.counter}")
            if self.counter >= self.stp_thr:
                self.write_msg(f"Counter over the threshold: {self.stp_thr} stopping the training")
                return True
        else:
            self.counter = 0

        return False

    def reset(self) -> None:
        self.counter = 0

