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
from typing import Any, List, Optional, Tuple, Union

import tensorflow as tf

from SNN2.src.core.gray.grayWrapper import validation_flag
from SNN2.src.decorators.decorators import ccb
from SNN2.src.io.logger import LogHandler
from SNN2.src.util.dataManger import DataManager
from SNN2.src.util.helper import dst2tensor
from tensorflow.keras.callbacks import Callback


@ccb
class activate_metric(Callback):

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
                 metrics_names: Union[List[str], str] = [],
                 attribute: str = "active",
                 logger: Optional[LogHandler] = None):
        if isinstance(metrics_names, str):
            metrics_names = ast.literal_eval(metrics_names)

        self.logger: Optional[LogHandler] = logger
        self.metrics = metrics_names
        self.attribute = attribute

    def on_test_begin(self, logs=None):
        self.write_msg(f"Detected test start, metrics: {self.model.metrics}")
        for metric in self.model.metrics:
            self.write_msg(f"{metric.name}")
            if metric.name in self.metrics:
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")
                metric.getattr(self.attribute).assign(True)
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")

    def on_test_end(self, logs=None):
        for metric in self.model.metrics:
            self.write_msg(f"{metric.name}")
            if metric.name in self.metrics:
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")
                metric.getattr(self.attribute).assign(False)
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")
@ccb
class set_metric_attr(Callback):

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
                 metrics_names: Union[List[str], str] = [],
                 attribute: str = "active",
                 set_to: Optional[Any]= None,
                 logger: Optional[LogHandler] = None):
        if isinstance(metrics_names, str):
            metrics_names = ast.literal_eval(metrics_names)

        self.logger: Optional[LogHandler] = logger
        self.metrics = metrics_names
        self.attribute = attribute
        self.value = set_to

    def on_train_begin(self, logs=None):
        self.write_msg("Detected train start")
        for metric in self.model.metrics:
            self.write_msg(f"{metric.name}")
            if metric.name in self.metrics:
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")
                metric.getattr(self.attribute).assign(self.value)
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")

    def on_test_begin(self, logs=None):
        self.write_msg(f"Detected test start, metrics: {self.model.metrics}")
        for metric in self.model.metrics:
            self.write_msg(f"{metric.name}")
            if metric.name in self.metrics:
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")
                metric.getattr(self.attribute).assign(self.value)
                self.write_msg(f"{metric.name}.{self.attribute} state: {metric.getattr(self.attribute)}")

@ccb
class mno_accuracy(Callback):

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
                 origin: tf.Tensor,
                 env,
                 fixed_margin_flag: Union[bool, str] = True,
                 fixed_margin: Union[float, str] = 0.0,
                 output: Optional[str] = None,
                 logger: Optional[LogHandler] = None):

        self.logger: Optional[LogHandler] = logger
        self.d_tst: tf.data.Dataset = d_tst
        self.d_tst_cls: tf.Tensor = tf.cast(d_tst_cls, tf.int8)
        self.output = output
        self.env = env
        self.df: DataManager = DataManager(["Evaluation", "Statistic", "Value"])
        self.origin = origin
        self.fixed_margin_flag = ast.literal_eval(fixed_margin_flag) if isinstance(fixed_margin_flag, str) else fixed_margin_flag
        self.fixed_margin = ast.literal_eval(fixed_margin) if isinstance(fixed_margin, str) else fixed_margin
        self.unique_origin, _, self.count_origin = tf.unique_with_counts(self.origin)
        self.write_msg(f"Origins: {self.unique_origin}, counters: {self.count_origin}")
        self.current_evaluation = 0

    def get_correct_wrong(self, obtained: tf.Tensor, expected: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        self.write_msg(f"dim obtained {len(obtained)} dim expected {len(expected)}")
        eq_idx = tf.where(obtained == expected)
        neq_idx = tf.where(obtained != expected)
        self.write_msg(f"indexes eq_idx: {len(eq_idx)}")
        self.write_msg(f"indexes neq_idx: {len(neq_idx)}")
        eq_ts = tf.gather(obtained, eq_idx)
        neq_ts = tf.gather(obtained, neq_idx)
        self.write_msg(f"eq_ts: {len(eq_ts)} - {eq_ts}")
        self.write_msg(f"neq_ts: {len(neq_ts)} - {neq_ts}")
        correct = tf.reshape(eq_ts, [-1])
        wrong = tf.reshape(neq_ts, [-1])
        self.write_msg(f"Computed correct: {correct}, wrong: {wrong}")
        self.write_msg(f"Computed len correct: {len(correct)}, len wrong: {len(wrong)}")
        return correct, wrong

    def register(self, stat: str, value: Any) -> None:
        update = {"Evaluation": self.current_evaluation,
                  "Statistic": stat,
                  "Value": value}
        self.write_msg(f"Update the df with {update}")
        self.df.append(update)

    def calculate_mno_cm(self, model) -> None:
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

        tot = {"TP": 0,"FP": 0,"TN": 0,"FN": 0,"U": 0}
        for org in self.unique_origin:
            self.write_msg(f"Origin: {org}")
            org_flags = tf.reshape(tf.gather(flags, tf.where(self.origin == org)), [-1])
            exp_flags = tf.reshape(tf.gather(self.d_tst_cls, tf.where(self.origin == org)), [-1])
            p_expected = len(tf.where(exp_flags == 0))
            n_expected = len(tf.where(exp_flags == 1))
            self.write_msg(f"flag {org} expected P: {p_expected}, N: {n_expected}")
            self.write_msg(f"org_flags: {org_flags}, exp_flags: {exp_flags}")
            self.write_msg(f"Dim org flags: {len(org_flags)}, dim exp_flags: {len(exp_flags)}")

            correct_l, wrong_l = self.get_correct_wrong(org_flags, exp_flags)
            self.write_msg(f"Dim correct: {len(correct_l)}, dim wrong: {len(wrong_l)}")
            tp = tf.size(tf.where(correct_l == 0)).numpy()
            fp = tf.size(tf.where(wrong_l == 0)).numpy()
            tn = tf.size(tf.where(correct_l == 1)).numpy()
            fn = tf.size(tf.where(wrong_l == 1)).numpy()
            u = tf.size(tf.where(wrong_l == 2)).numpy()
            tot["TP"] += tp
            tot["FP"] += fp
            tot["TN"] += tn
            tot["FN"] += fn
            tot["U"] += u
            matrix = {f"MNO_{org.numpy()}_TP": tp,
                      f"MNO_{org.numpy()}_FP": fp,
                      f"MNO_{org.numpy()}_TN": tn,
                      f"MNO_{org.numpy()}_FN": fn,
                      f"MNO_{org.numpy()}_U": u,
                      f"MNO_{org.numpy()}_expected_P": p_expected,
                      f"MNO_{org.numpy()}_expected_N": n_expected}
            self.write_msg(f"confusion matrix: {matrix}")
            for key, item in matrix.items():
                self.register(key, item)
        self.register("Accuracy", (tot["TP"]+tot["TN"])/(sum(tot.values())))

    def on_train_begin(self, logs=None) -> None:
        self.write_msg("the train has begin")

    def on_train_end(self, logs=None):
        self.write_msg(f"MNO metrics save")
        confusion_matrix = self.calculate_mno_cm(self.model)
        self.df.save(self.output)
        self.current_evaluation += 1

    # def on_test_end(self, logs=None):
    #     confusion_matrix = self.calculate_mno_cm(self.model)
    #     self.df.save(self.output)

