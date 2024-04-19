# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.model.custom.customWrapper import CustomModel_selector as CMS
from SNN2.src.model.embeddings.embeddingsWrapper import Embedding_selector as ES
from SNN2.src.model.modelHandler import DefaultModelHandler
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.util.strings import s
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from SNN2.src.decorators.decorators import ModelManager
from SNN2.src.core.data.PreProcessing import PreProcessing as PP

@ModelManager
class ClassificationEnv(DefaultModelHandler):
    """modelHandler.

    Generate and manage a neural network model
	"""

    def __init__(self,
                 *args,
                 pp: Optional[PP] = None,
                 **kwargs):
        if pp is None:
            raise Exception("ClassificationEnv requires the pre processing object!")

        self.data: PP = pp
        super().__init__(*args, **kwargs)

        self.required_layers = ast.literal_eval(self.params["layers"])
        self.required_metrics = ast.literal_eval(self.params["metrics"])
        self.required_losses = ast.literal_eval(self.params["losses"])
        self.loss_parameters = ast.literal_eval(self.params["lossParameters"])
        self.action_delta = ast.literal_eval(self.params["actionDelta"])
        self.output_weights = self.params["weights_file"]

        # define shape
        self.shape = ast.literal_eval(self.params[s.shape_key])
        self.nnodes = int(self.params[s.n_nodes_key])
        # Define if the model has already been trained
        self.trained = False

        # Define the embedding using the embedding class
        train_mean = self.data.train_norm[0]
        train_std = self.data.train_norm[1]
        self.emb = self.embeddings.get_handler(self.params["embedding_name"])(
                    [self.layers.get_handler(layer)(mean=train_mean,
                                                    variance=train_std)
                        if layer == "Normalization" else
                     self.layers.get_handler(layer)()
                        for layer in self.required_layers]
                ).layers

        # Create the model
        inpt = self.define_model_inputs()
        self.write_msg(f"Model inputs: {inpt}")
        self.model_definition = self.define_model(
                    inpt, self.define_model_output(self.emb, inpt, name="output")
                )

        self.write_msg("Model initialized")

        # If the weights files exists then load those
        self.load()

        # Compile the network
        # self.original_loss_params = self.lossParamH.objects["margin"].args
        self.model = self.compile()
        self.write_msg(f"Model compiled")

    def compile(self, *args, **kwargs) -> Model:
        if self.required_metrics is None:
            raise Exception("A list of metrics must be configured")
        if self.loss_parameters is None:
            raise Exception("A list of loss parameters must be configured")
        if self.required_losses is None:
            raise Exception("A list of loss functions must be configured")

        metrics = [self.metricsH.get_handler(metric)() for metric in self.required_metrics]
        self.loss_param_ref = {key: self.lossParamH.get_handler(key)() for key in self.loss_parameters}
        losses = [self.lossesH.get_handler(loss)(**self.loss_param_ref) for loss in self.required_losses]

        assert self.model_definition is not None

        self.model_definition.compile(loss=losses, metrics=["categorical_accuracy", tf.keras.metrics.CategoricalCrossentropy(), tf.keras.metrics.KLDivergence()], **kwargs)
        self.write_msg("Model compiled")
        return self.model_definition


    def define_model_inputs(self) -> List[Layer]:
        """define_model_inputs.

        Parameters
        ----------

        Returns
        -------
        Tuple[layers.Layer, ...]

        """
        anchor_input = self.layers.get_handler("input")(name="Input", shape=self.shape)
        self.write_msg("Three Input layers created")
        return [anchor_input]

    def define_model_output(self, emb, input, **kwargs) -> Layer:
        output = emb(input)
        return output

    def get_state(self) -> np.ndarray:
        return np.array([param.numpy() for param in self.loss_param_ref.values()])

    @property
    def state(self) -> np.ndarray:
        return np.array([param.numpy() for param in self.loss_param_ref.values()])

    def execute_action(self, action: int) -> None:
        if action == 0:
            # Increase the margin loss_parameter
            self.loss_param_ref["margin"].assign_add(self.action_delta)
        if action == 1:
            # Decrease the margin loss_parameter
            self.loss_param_ref["margin"].assign_sub(self.action_delta)
