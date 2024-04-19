# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
CallbackHelper module
=====================

Use this module to manage callbacks helper functions.
"""

import tensorflow as tf

from typing import Any, List, Optional, Tuple
from SNN2.src.decorators.decorators import c_logger

from SNN2.src.params.paramHandler import ParamHandler
from SNN2.src.core.data.PreProcessing import PreProcessing as PP
from SNN2.src.model.modelHandler import DefaultModelHandler as MH
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.model.reinforcement.reinfoceModelHandler import ReinforceModelHandler as RLMH
from SNN2.src.util.helper import dst2tensor
from SNN2.src.io.logger import LogHandler as LH
from tensorflow.keras.callbacks import Callback

@c_logger
class CallbackHelper:

    def __init__(self,
                 required_callbacks: List[str],
                 callbacksH: ParamHandler,
                 data: PP,
                 test_data: Optional[tf.data.Dataset] = None,
                 validation_data: Optional[tf.data.Dataset] = None,
                 snn_model: Optional[MH] = None,
                 ph: Optional[PH] = None,
                 rl_model: Optional[RLMH] = None,
                 EnvExitFunctionH: Optional[PH] = None) -> None:
        self.req_cbs = required_callbacks
        self.write_msg(f"Required callbacks: {self.req_cbs}")
        self.cbsH = callbacksH
        self.data = data
        self.test_data = test_data
        self.validation_data = validation_data
        self.unsigned_clbs = ["csvLogger", "earlyStopping", "modelCheckpoint", "tensorBoard"]
        self.model = snn_model
        self.ph = ph
        self.rl_model = rl_model
        self.EnvExitFunctionH = EnvExitFunctionH

    def __check(self, obj: Any, label = "Object") -> None:
        if obj is None:
            raise Exception(f"{label} cannot be None")

    def __ott_accuracy(self) -> Callback:
        self.__check(self.test_data, label="Test data")
        return self.cbsH.get_handler("ott_accuracy")(
                        self.test_data,
                        self.data.test_triplets.dft("ExpectedLabel"),
                        self.data.test_triplets.dft("Targets"),
                        self.data.test.dft("Classes"),
                        self.model,
                        logger = self.logger)

    def __mno_cm(self) -> Callback:
        self.__check(self.test_data, label="Test data")
        return self.cbsH.get_handler("mno_cm")(
                        self.test_data,
                        self.data.test_triplets.dft("ExpectedLabel"),
                        self.data.test.dft("Classes"),
                        self.model,
                        logger = self.logger)

    def get_saveEmbeddings_obj(self) -> Tuple[List[tf.Tensor], List[str]]:
        sample_sets = [self.data.goods_prop.dft("Windows"),
                       self.data.bads_prop.dft("Windows"),
                       self.data.grays_prop.dft("Windows"),
                       self.data.training.dft("Windows"),
                       self.data.validation.dft("Windows"),
                       self.data.test.dft("Windows"),
                       self.data.gray_triplets.dft("TripletDst")[:, 0],
                       self.data.gray_triplets.dft("TripletDst")[:, 1],
                       self.data.gray_triplets.dft("TripletDst")[:, 2]]
        labels = ["goods", "bads", "difficult", "training", "validation", "test",
                  "difficult_p", "difficult_a", "difficult_n"]
        return (sample_sets, labels)

    def get_pklObject_sets(self) -> Tuple[List[tf.Tensor], List[str]]:
        objects = [self.data.goods_prop.dft("Targets"),
                   self.data.goods_prop.dft("Classes"),
                   self.data.goods_prop.dft("ExpectedLabel"),
                   self.data.bads_prop.dft("Targets"),
                   self.data.bads_prop.dft("Classes"),
                   self.data.bads_prop.dft("ExpectedLabel"),
                   self.data.grays_prop.dft("Targets"),
                   self.data.grays_prop.dft("Classes"),
                   self.data.grays_prop.dft("ExpectedLabel"),
                   self.data.training.dft("Targets"),
                   self.data.training.dft("Classes"),
                   self.data.training.dft("ExpectedLabel"),
                   self.data.validation.dft("Targets"),
                   self.data.validation.dft("Classes"),
                   self.data.validation.dft("ExpectedLabel"),
                   self.data.test.dft("Targets"),
                   self.data.test.dft("Classes"),
                   self.data.test.dft("ExpectedLabel"),
                   self.data.gray_triplets.dft("Targets"),
                   self.data.gray_triplets.dft("Classes"),
                   self.data.gray_triplets.dft("ExpectedLabel"),
                  ]
        labels = ["goods_vmaf", "goods_origin", "goods_exp_l",
                  "bads_vmaf", "bads_origin", "bads_exp_l",
                  "difficult_vmaf", "difficult_origin", "difficult_exp_l",
                  "training_vmaf", "training_origin", "training_exp_l",
                  "validation_vmaf", "validation_origin", "validation_exp_l",
                  "test_vmaf", "test_origin", "test_exp_l",
                  "difficult_triplets_vmaf", "difficult_triplets_origin", "difficult_triplets_exp_l"]
        return (objects, labels)

    def __saveEmbeddings(self) -> Callback:
        self.__check(self.model, label="Model")
        self.__check(self.ph, label="PickleHandler")
        return self.cbsH.get_handler("saveEmbeddings")(
                        *self.get_saveEmbeddings_obj(),
                        self.model.emb,
                        self.ph)

    def __testParam(self) -> Callback:
        self.__check(self.model, label="Model")
        return self.cbsH.get_handler("testParam")(
                        self.model.loss_param_ref)

    def __controlledMargin(self) -> Callback:
        self.__check(self.model, label="Model")
        return self.cbsH.get_handler("controlledMargin")(
                        self.model.loss_param_ref, logger=self.logger)

    def __controlledCrossEntropy(self) -> Callback:
        self.__check(self.model, label="Model")
        return self.cbsH.get_handler("controlledCrossEntropy")(
                        self.model.loss_param_ref,
                        self.validation_data,
                        self.data.validation["SamplesDst"]["CategoricalTargets"],
                        self.ph,
                        logger=self.logger)

    def __AdvancedControlledCrossEntropy(self) -> Callback:
        self.__check(self.model, label="Model")
        return self.cbsH.get_handler("ACCCE")(
                        self.model.loss_param_ref,
                        self.validation_data,
                        self.data.validation["SamplesDst"]["CategoricalTargets"],
                        self.ph,
                        logger=self.logger)


    def __RL_manager(self, manager_name="RL_manager") -> Callback:
        self.__check(self.rl_model, label="RL Model")
        self.__check(self.validation_data, label="Validation data")
        return self.cbsH.get_handler(manager_name)(
                        self.rl_model,
                        self.model,
                        self.validation_data,
                        self.data.validation_triplets.dft("ExpectedLabel"),
                        EnvExitFunctionH=self.EnvExitFunctionH,
                        logger = self.logger)

    def __saveObject(self) -> Callback:
        return self.cbsH.get_handler("saveObject")(
                        *self.get_pklObject_sets(),
                        self.ph)

    def get_generic_callbacks(self) -> List[Callback]:
        result = []
        if "saveEmbeddings" in self.req_cbs:
            result.append(self.__saveEmbeddings())
        if "testParam" in self.req_cbs:
            result.append(self.__testParam())
        if "controlledMargin" in self.req_cbs:
            result.append(self.__controlledMargin())
        if "controlledCrossEntropy" in self.req_cbs:
            result.append(self.__controlledCrossEntropy())
        if "ACCCE" in self.req_cbs:
            result.append(self.__AdvancedControlledCrossEntropy())
        if "RL_manager" in self.req_cbs:
            result.append(self.__RL_manager())
        if "RL_partial_manager" in self.req_cbs:
            result.append(self.__RL_manager(manager_name="RL_partial_manager"))
        if "saveObject" in self.req_cbs:
            result.append(self.__saveObject())
        if "mno_cm" in self.req_cbs:
            result.append(self.__mno_cm())
        return result


    def get_evaluation_callbacks(self) -> List[Callback]:
        result = []
        if "ott_accuracy" in self.req_cbs:
            result.append(self.__ott_accuracy())
        # if "mno_cm" in self.req_cbs:
        #     result.append(self.__mno_cm())
        return result

    def get_unsigned_callbacks(self) -> List[Callback]:
        res = []
        for clb in self.unsigned_clbs:
            self.write_msg(f"Looking for {clb}", level=LH.DEBUG)
            if clb in self.req_cbs:
                res.append(self.cbsH.get_handler(clb)())
        return res

    def get_callbacks(self, caller="Fit") -> List[Callback]:
        cbs = self.get_generic_callbacks()
        if caller == "Evaluate":
            cbs.extend(self.get_evaluation_callbacks())
        cbs.extend(self.get_unsigned_callbacks())
        return cbs
