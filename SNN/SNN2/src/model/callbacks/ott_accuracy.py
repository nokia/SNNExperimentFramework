# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

import ast
from typing import Any, Optional, Tuple, Union

import tensorflow as tf

from SNN2.src.core.gray.grayWrapper import validation_flag
from SNN2.src.decorators.decorators import ccb
from SNN2.src.io.logger import LogHandler
from SNN2.src.util.dataManger import DataManager
from SNN2.src.util.helper import dst2tensor
from tensorflow.keras.callbacks import Callback


@ccb
class ott_accuracy(Callback):

    def write_msg(self, msg: str, level: int = LogHandler.INFO) -> None:
        """__write_msg.
        write a message into the log file with a defined log level

        parameters
        ----------
        msg : str
            msg to print
        level : int
            level default info

        returns
        -------
        none

        """
        if self.logger is None:
            return
        self.logger(f"{self.__class__.__name__}", f"{msg}", level=level)

    def __init__(self,
                 d_tst: tf.data.Dataset,
                 d_tst_cls: tf.Tensor,
                 expected_qoe: tf.Tensor,
                 origin: tf.Tensor,
                 env,
                 fixed_margin_flag: Union[bool, str] = True,
                 fixed_margin: Union[float, str] = 0.0,
                 output: Optional[str] = None,
                 qoe_threshold: Union[int, str] = 10,
                 logger: Optional[LogHandler] = None):
        if isinstance(qoe_threshold, str):
            qoe_threshold = ast.literal_eval(qoe_threshold)

        self.df: DataManager = DataManager(["Statistic", "Value"])
        self.logger: Optional[LogHandler] = logger
        self.output: str = output
        self.d_tst: tf.data.Dataset = d_tst
        self.d_tst_cls: tf.Tensor = d_tst_cls
        self.expected_qoe = expected_qoe
        self.qoe_threshold = qoe_threshold
        self.origin = origin
        self.env = env
        self.fixed_margin_flag = ast.literal_eval(fixed_margin_flag) if isinstance(fixed_margin_flag, str) else fixed_margin_flag
        self.fixed_margin = ast.literal_eval(fixed_margin) if isinstance(fixed_margin, str) else fixed_margin
        self.unique_origin, _, self.count_origin = tf.unique_with_counts(self.origin)
        self.write_msg(f"Origins: {self.unique_origin}, counters: {self.count_origin}")
        self.write_msg(f"OTT qoe threshold: {self.qoe_threshold}")
        self.write_msg(f"Expected qoe: {self.expected_qoe}")
        self.exp_flags = tf.cast(tf.reshape(tf.where(self.expected_qoe > self.qoe_threshold, 0, 1), [-1]), tf.int8)
        self.write_msg(f"Initialization done")
        self.write_msg(f"exp_flags: {self.exp_flags}")
        self.write_msg(f"OTT Expected 0: {tf.size(tf.where(self.exp_flags == 0))}")
        self.write_msg(f"OTT Expected 1: {tf.size(tf.where(self.exp_flags == 1))}")
        self.write_msg(f"MNO Expected 0: {tf.size(tf.where(self.d_tst_cls == 0))}")
        self.write_msg(f"MNO Expected 1: {tf.size(tf.where(self.d_tst_cls == 1))}")

    def get_correct_wrong(self, obtained: tf.Tensor, expected: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        correct = tf.reshape(tf.gather(obtained, tf.where(obtained == expected)), [-1])
        wrong = tf.reshape(tf.gather(obtained, tf.where(obtained != expected)), [-1])
        self.write_msg(f"Computed correct: {correct}, wrong: {wrong}")
        return correct, wrong

    def register(self, stat: str, value: Any) -> None:
        update = {"Statistic": stat,
                  "Value": value}
        self.write_msg(f"Update the df with {update}")
        self.df.append(update)

    def calculate_ott_accuracy(self, model) -> float:
        ap, an = self.model.predict(self.d_tst, verbose=0)
        self.write_msg(f"Obtained ap and an: {(ap, an)}")
        current_margin = round(self.env.state[0], 1) if not self.fixed_margin_flag else self.fixed_margin
        flags: tf.Tensor = validation_flag((ap, an),
                                            self.d_tst_cls,
                                            margin = current_margin,
                                            undecided_reverse=True)
        flags: tf.Tensor = tf.cast(flags, tf.int8)
        self.write_msg(f"Computed predicted flags: {flags}")
        self.write_msg(f"Predicted 0: {tf.size(tf.where(flags == 0))}")
        self.write_msg(f"Predicted 1: {tf.size(tf.where(flags == 1))}")
        correct, wrong = self.get_correct_wrong(flags, self.exp_flags)
        total = tf.size(correct).numpy() + tf.size(wrong).numpy()
        return tf.size(correct).numpy()/total


    def calculate_ott_cm(self, model) -> float:
        ap, an = self.model.predict(self.d_tst, verbose=0)
        self.write_msg(f"Obtained ap and an: {(ap, an)}")
        current_margin = round(self.env.state[0], 1) if not self.fixed_margin_flag else self.fixed_margin
        flags: tf.Tensor = validation_flag((ap, an),
                                            self.d_tst_cls,
                                            margin = current_margin)
        flags: tf.Tensor = tf.cast(flags, tf.int8)
        self.write_msg(f"Computed predicted flags: {flags}")
        self.write_msg(f"Predicted 0: {tf.size(tf.where(flags == 0))}")
        self.write_msg(f"Predicted 1: {tf.size(tf.where(flags == 1))}")
        for org in self.unique_origin:
            self.write_msg(f"Origin: {org}")
            org_flags = tf.cast(tf.reshape(tf.gather(flags, tf.where(self.origin == org)), [-1]), tf.int8)
            exp_flags = tf.cast(tf.reshape(tf.gather(self.exp_flags, tf.where(self.origin == org)), [-1]), tf.int8)
            p_expected = len(tf.where(exp_flags == 0))
            n_expected = len(tf.where(exp_flags == 1))
            self.write_msg(f"flag {org} expected P: {p_expected}, N: {n_expected}")
            self.write_msg(f"org_flags: {org_flags}, exp_flags: {exp_flags}")
            self.write_msg(f"Dim org flags: {len(org_flags)}, dim exp_flags: {len(exp_flags)}")

            correct_l, wrong_l = self.get_correct_wrong(org_flags, exp_flags)
            tp=  tf.size(tf.where(correct_l == 0)).numpy()
            fp = tf.size(tf.where(wrong_l == 0)).numpy()
            tn = tf.size(tf.where(correct_l == 1)).numpy()
            fn = tf.size(tf.where(wrong_l == 1)).numpy()
            u = tf.size(tf.where(wrong_l == 2)).numpy()
            matrix = {f"OTT_{org.numpy()}_TP": tp,
                      f"OTT_{org.numpy()}_FP": fp,
                      f"OTT_{org.numpy()}_TN": tn,
                      f"OTT_{org.numpy()}_FN": fn,
                      f"OTT_{org.numpy()}_U": u,
                      f"OTT_{org.numpy()}_expected_P": p_expected,
                      f"OTT_{org.numpy()}_expected_N": n_expected}
            self.write_msg(f"Confusion matrix computed: {matrix}")
            for key, item in matrix.items():
                self.register(key, item)

    def on_test_end(self, logs=None):
        accuracy = self.calculate_ott_accuracy(self.model)
        self.calculate_ott_cm(self.model)
        self.write_msg(f"Accuracy obtained: {accuracy}")
        self.register("OTT_accuracy", accuracy)
        self.df.save(self.output)
        self.write_msg(f"Df stats saved on: {self.output}")
