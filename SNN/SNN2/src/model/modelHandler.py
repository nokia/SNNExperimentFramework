# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class module
============

Use this module to manage the model objects and create a siamese network model
"""
import ast
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.model.custom.customWrapper import CustomModel_selector as CMS
from SNN2.src.model.embeddings.embeddingsWrapper import Embedding_selector as ES
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.util.strings import s
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

class DefaultModelHandler:

    def write_msg(self, msg: str, level: int = LH.INFO) -> None:
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
        self.logger(f"{self.__class__.__name__}", f"{msg}", level=level)


    def __init__(self,
                 params: PH,
                 embeddings: PH,
                 layers: PH,
                 metrics: PH,
                 losses: PH,
                 lossParam: PH,
                 numpyRNG: PH,
                 ph: PkH,
                 logger: LH,
                 hash: Optional[str] = None,
                 debug: bool = False,
                 **kwargs):
        self.params: PH = params
        self.embeddings: PH = embeddings
        self.layers: PH = layers
        self.metricsH: PH = metrics
        self.lossesH: PH = losses
        self.lossParamH: PH = lossParam
        self.ph: PkH = ph
        self.logger: LH = logger
        self.hash: Optional[str] = hash
        self.debug: bool = debug
        self.output_weights: Optional[str] = None
        self.model_definition: Optional[Model] = None
        self.model: Optional[Model] = None
        self.trained: bool = False
        self.loss_param_ref = None
        self.required_metrics: Optional[List[str]] = None
        self.loss_parameters: Optional[List[str]] = None
        self.required_losses: Optional[List[str]] = None
        self.np_rng = numpyRNG["numpy_rng"]

    def define_model(self, inputs, outputs) -> Model:
        if inputs is None:
            return outputs
        return Model(inputs=[*inputs], outputs=outputs)

    def compile(self, custom_model: str, *args,  **kwargs) -> Model:
        if custom_model is None:
            raise Exception("A custom model must be provided")
        if self.required_metrics is None:
            raise Exception("A list of metrics must be configured")
        if self.loss_parameters is None:
            raise Exception("A list of loss parameters must be configured")
        if self.required_losses is None:
            raise Exception("A list of loss functions must be configured")

        # print(f"Compiling the model with the following metrics: {self.required_metrics}")
        self.model = CMS(custom_model, self.model_definition, *args)

        metrics = [self.metricsH.get_handler(metric)() for metric in self.required_metrics]
        # print(f"{[m.name for m in metrics]}")
        self.loss_param_ref = {key: self.lossParamH.get_handler(key)() for key in self.loss_parameters}
        losses = [self.lossesH.get_handler(loss)(**self.loss_param_ref) for loss in self.required_losses]

        self.model.compile(loss=losses, metrics=metrics, **kwargs)
        self.write_msg("Model compiled")
        return self.model

    def save(self, obj: Optional[Any] = None) -> None:
        if self.debug:
            self.write_msg("Debug option active, not saving the model")
            return
        if self.output_weights is None:
            raise Exception("An output weight file must be setted by the implementation object")
        if self.model is None:
            raise Exception("A model must be provided")

        obj = self.model if obj is None else obj

        path = FH.hash_path(self.output_weights, hash=self.hash)
        obj.save_weights(path)
        self.write_msg("Model weights saved")

    def load(self) -> None:
        if self.output_weights is None:
            raise Exception("An output weight file must be setted by the implementation object")
        if self.model_definition is None:
            raise Exception("A model must be provided")

        path = FH.hash_path(self.output_weights, hash=self.hash)
        if not FH.exists(path):
            self.write_msg("The weights file does not exists, the model needs to be trained")
            return

        self.model_definition.load_weights(path)
        self.trained = True
        self.write_msg("Model weights loaded")

    def summary(self) -> str:
        if self.model is None:
            raise Exception("No model available")
        return self.model.summary()

    def get_weights(self) -> Any:
        if self.model is None:
            raise Exception("No model available")
        return self.model.get_weights()

    def set_weights(self, weights) -> Any:
        if self.model is None:
            raise Exception("No model available")
        return self.model.set_weights(weights)
