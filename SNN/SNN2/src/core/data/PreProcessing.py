# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
PreProcessing module
====================

Use this module to preprocess the input data if necessary
"""

import ast
from copy import deepcopy
import numpy as np
import pandas as pd
import tensorflow as tf
from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.util.dataManger import DataManager as MDF
from SNN2.src.decorators.decorators import c_logger

from SNN2.src.util.strings import s
from SNN2.src.io.progressBar import pb
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.util.helper import dst2tensor
from SNN2.src.actions.dataFrame import MNIST_unpack

from typing import Optional, Tuple, List, Any, Dict, Callable

@c_logger
class PreProcessing:
    """PreProcessing.

    General class to pre process the input data if necessary
	"""
    def __analyze(self, ids: List[int], stop: bool = True) -> None:
        for lf in ids:
            self.write_msg(f"{self.gray_triplets.dft('Windows')[lf]}")
            self.write_msg(f"{self.gray_triplets.dft('Targets')[lf]}")
            self.write_msg(f"{self.gray_triplets.dft('OriginIndexes')[lf]}")
            self.write_msg(f"{self.gray_triplets.dft('ExpID')[lf]}")
            self.write_msg(f"{self.gray_triplets.dft('ExpectedLabel')[lf]}")
            self.write_msg(f"{self.grays_prop.dft('ExpectedLabel')[lf]}")
            self.write_msg("-------------------------------------------")
        if stop:
            raise Exception

    def __init__(self,
                 pp_parm: PH,
                 action_parm: PH,
                 flow_param: PH,
                 ph: PkH):
        """__init__.

	    Parameters
		----------
		"""
        self.params = pp_parm
        self.action_parm = action_parm
        self.ph: PkH = ph

        self.flow_name = self.params[s.pp_flow_name]
        self.write_msg(f"Flow name: {self.flow_name}")
        self.write_msg(f"Flow PklH: {str(self.ph)}")
        self.flow = flow_param.get_handler(self.flow_name)(self.action_parm,
                                                           PklH=self.ph,
                                                           logger=self.logger)
        self.write_msg(f"Flow state: {self.flow.state}")
        self.flow.execute()
        self.write_msg(f"Flow state: {self.flow.state}")

    @property
    def training(self) -> tf.Tensor:
        return self.flow.training

    @property
    def validation(self) -> tf.Tensor:
        return self.flow.validation

    @property
    def test(self) -> tf.Tensor:
        return self.flow.test

    @property
    def training_triplets(self) -> tf.Tensor:
        return self.flow.training_triplets

    @property
    def validation_triplets(self) -> tf.Tensor:
        return self.flow.validation_triplets

    @property
    def test_triplets(self) -> tf.Tensor:
        return self.flow.test_triplets

    @property
    def gray_triplets(self) -> tf.Tensor:
        return self.flow.gray_triplets

    @property
    def train_norm(self) -> tf.Tensor:
        return self.flow.train_norm

    def __getattr__(self, name):
        return getattr(self.flow, name, None)

