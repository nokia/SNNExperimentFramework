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

"""
PreProcessing module
====================

Use this module to preprocess the input data if necessary
"""

import ast
import numpy as np
import tensorflow as tf

from SNN2.src.util.strings import s
from SNN2.src.io.progressBar import pb
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.util.helper import dst2tensor

from typing import Tuple, List, Any, Dict, Callable

class PreProcessing:
    """PreProcessing.

    General class to pre process the input data if necessary
	"""

    def __write_msg(self, msg: str, level: int = LH.INFO) -> None:
        """__write_msg.
        Write a message into the log file with a defined log level

        Parameters
        ----------
        msg : str
            msg to print
        level : int
            level default INFO

        Returns
        -------
        None

        """
        self.logger(f"{self.__class__.__name__}: {msg}", level)

    def check_pkl_list(self, file_list: List[str]) -> bool:
        return all([self.ph.check(file) for file in file_list])

    def load_pkl_list(self, file_list: List[str], wrapper: Callable = None) -> Tuple[Any, ...]:
        if wrapper is None:
            def repeat(obj: Any):
                return obj
            wrapper = repeat
        return tuple([wrapper(self.ph.load(file)) for file in file_list])

    def save_pkl_dct(self, obj_dct: Dict[str, Any], wrapper: Callable = None) -> None:
        if wrapper is None:
            def repeat(obj: Any):
                return obj
            wrapper = repeat
        for key, obj in obj_dct.items():
            self.ph.save(wrapper(obj), key)

    def retrive(self, data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        wdw, trg, cls = data
        return (tf.data.Dataset.from_tensor_slices(wdw.numpy()),
                tf.data.Dataset.from_tensor_slices(trg.numpy()),
                tf.data.Dataset.from_tensor_slices(cls.numpy()))

    def prepare(self, dst: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (dst2tensor(dst[0]), dst2tensor(dst[1]), dst2tensor(dst[2]))

    def __init__(self,
                 pp_parm: PH,
                 action_parm: PH,
                 ph: PkH,
                 logger: LH):
        """__init__.

	    Parameters
		----------
		"""
        self.params = pp_parm
        self.ph: PkH = ph
        self.logger: LH = logger
        self.actions = ast.literal_eval(self.params[s.pp_actions])
        self.pbar = pb.bar(total=len(self.actions)+12)

        self.data = action_parm.get_handler(self.actions[0])()
        self.pbar.update(1)
        for action in self.actions[1:]:
            self.__write_msg(f"Execution of action {action} on the loaded data")
            self.data = action_parm.get_handler(action)(df=self.data)
            self.pbar.update(1)

        # Now it's mandatory to operate directly on the data
        # Maybe one day I will be able to transfer such operations to
        # the conf files.
        self.data_stats = action_parm.get_handler("dropColumns")(df=self.data, columns=["video", "problem", "value"])
        self.pbar.update(1)

        self.data_subDfs = action_parm.get_handler("durationSeparation")(df=self.data)
        self.pbar.update(1)

        pkl_l = [s.goods_wdw, s.goods_trg, s.goods_cls,
                 s.bads_wdw, s.bads_trg, s.bads_cls,
                 s.grays_wdw, s.grays_trg, s.grays_cls]

        if self.check_pkl_list(pkl_l):
            objs = self.load_pkl_list(pkl_l)
            self.goods_windows, self.goods_targets, self.goods_classes, \
            self.bads_windows, self.bads_targets, self.bads_classes, \
            self.grays_windows, self.grays_targets, self.grays_classes = objs
            self.pbar.update(2)
        else:
            self.windows, self.targets = action_parm.get_handler("listWindowing")(self.data_subDfs, window=int(self.params["window"]))
            wdw, trg, cls = action_parm.get_handler("windowsSeparation")(self.windows, self.targets, logger=self.logger)
            self.goods_windows, self.goods_targets, self.goods_classes = wdw[0], trg[0], cls[0]
            self.grays_windows, self.grays_targets, self.grays_classes = wdw[1], trg[1], cls[1]
            self.bads_windows, self.bads_targets, self.bads_classes = wdw[2], trg[2], cls[2]
            self.pbar.update(1)

            self.goods_windows, self.goods_targets, self.goods_classes = action_parm.get_handler("windowDropOutliers")(
                            (self.goods_windows, self.goods_targets, self.goods_classes),
                            logger=self.logger
                        )
            self.pbar.update(1)
            self.__write_msg(f"Number of samples: {len(self.goods_windows)} Goods - {len(self.grays_windows)} Grays - {len(self.bads_windows)} Bads")

            # drop goods_windows outliers
            objs = [self.goods_windows, self.goods_targets, self.goods_classes,
                    self.bads_windows, self.bads_targets, self.bads_classes,
                    self.grays_windows, self.grays_targets, self.grays_classes]
            save_dct = {key: obj for key, obj in zip(pkl_l, objs)}

            self.save_pkl_dct(save_dct)

        self.gray_post_train, self.gray_train, _ = action_parm.get_handler("TrnValTstSeparation")(
                    (self.grays_windows, self.grays_targets, self.grays_classes),
                    training_portion=ast.literal_eval(self.params[s.gray_post_train_portion]),
                    validation_portion=ast.literal_eval(self.params[s.gray_in_train_portion]),
                    test_portion=0.0
                )
        self.grays_windows, self.grays_targets, self.grays_classes = (dst2tensor(self.gray_post_train[0]), dst2tensor(self.gray_post_train[1]), dst2tensor(self.gray_post_train[2]))
        self.gray_train = (dst2tensor(self.gray_train[0]), dst2tensor(self.gray_train[1]), dst2tensor(self.gray_train[2]))
        if len(self.gray_train[0]) > 0:
            self.gray_train_classes = action_parm.get_handler("flag_windows")(
                        self.gray_train[0],
                        log=self.logger
                    )
            self.gray_train = (self.gray_train[0], self.gray_train[1], self.gray_train_classes)

        self.grays_dst = self.gray_post_train
        self.goods_dst = (tf.data.Dataset.from_tensor_slices(self.goods_windows),
                          tf.data.Dataset.from_tensor_slices(self.goods_targets),
                          tf.data.Dataset.from_tensor_slices(self.goods_classes))
        self.bads_dst = (tf.data.Dataset.from_tensor_slices(self.bads_windows),
                          tf.data.Dataset.from_tensor_slices(self.bads_targets),
                          tf.data.Dataset.from_tensor_slices(self.bads_classes))
        self.pbar.update(1)

        pkl_l = [s.pkl_training, s.pkl_validation, s.pkl_test,
                 s.goods_wdw_norm, s.grays_wdw_norm, s.bads_wdw_norm]

        if self.check_pkl_list(pkl_l):
            obj = self.load_pkl_list(pkl_l, wrapper=self.retrive)
            self.training, self.validation, self.test, self.goods_dst, self.grays_dst, self.bads_dst = obj
            self.pbar.update(3)
        else:
            # Genrate the first training-validation-test datasets
            self.training, self.validation, self.test = action_parm.get_handler("BalanceSeparationNG")(
                        [(self.goods_windows, self.goods_targets, self.goods_classes),
                         (self.bads_windows, self.bads_targets, self.bads_classes),
                         self.gray_train],
                        logger=self.logger
                    )
            self.pbar.update(1)

            self.train_mean = action_parm.get_handler("featureMean")(self.training[0])
            self.train_std = action_parm.get_handler("featureStd")(self.training[0])
            self.pbar.update(1)

            self.training = action_parm.get_handler("normalize")(self.training, self.train_mean, self.train_std)
            self.validation = action_parm.get_handler("normalize")(self.validation, self.train_mean, self.train_std)
            self.test = action_parm.get_handler("normalize")(self.test, self.train_mean, self.train_std)
            self.grays_dst = action_parm.get_handler("normalize")(self.grays_dst, self.train_mean, self.train_std)
            self.goods_dst = action_parm.get_handler("normalize")(self.goods_dst, self.train_mean, self.train_std)
            self.bads_dst = action_parm.get_handler("normalize")(self.bads_dst, self.train_mean, self.train_std)
            self.pbar.update(1)

            objs = [self.training, self.validation, self.test,
                    self.goods_dst, self.grays_dst, self.bads_dst]
            save_dct = {key: obj for key, obj in zip(pkl_l, objs)}

            self.save_pkl_dct(save_dct, wrapper=self.prepare)

        self.training_triplets = action_parm.get_handler("generateTriplets")(
                    self.training,
                )
        self.pbar.update(1)
        self.validation_triplets = action_parm.get_handler("generateTriplets")(
                    self.validation,
                )
        self.pbar.update(1)
        self.test_triplets = action_parm.get_handler("generateTriplets")(
                    self.test,
                )
        self.pbar.update(1)
        self.gray_triplets = action_parm.get_handler("generatePredictionTriplets")(
                    self.grays_dst, self.goods_dst, self.bads_dst
                )
        self.pbar.update(1)
