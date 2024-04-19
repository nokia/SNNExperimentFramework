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

@ModelManager
class SiameseEnv(DefaultModelHandler):
    """modelHandler.

    Generate and manage a neural network model
	"""

    def __init__(self,
                 *args,
                 **kwargs):
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
        self.margin_reset = self.params["margin_reset", True]
        self.random_margin = self.params["random_margin", True]
        self.random_margin_lower = -1.0
        self.random_margin_higher = 1.0
        if self.random_margin:
            assert not self.margin_reset

        # Define the embedding using the embedding class
        self.emb = self.embeddings.get_handler(self.params["embedding_name"])(
                    [self.layers.get_handler(layer)() for layer in self.required_layers]
                ).layers

        # Create the model
        inpt = self.define_model_inputs()
        self.model_definition = self.define_model(
                    inpt, self.define_model_output(self.emb, inpt, name="output")
                )
        self.write_msg("Model initialized")

        # If the weights files exists then load those
        self.load()

        # Compile the network
        self.original_loss_params = self.lossParamH.objects["margin"].args
        self.model = self.compile()

    def compile(self, *args, active_reset: bool = False, force_keeping: bool = False, **kwargs) -> Model:
        # print(f"{active_reset}, {self.margin_reset}, {self.random_margin}, {force_keeping}, {self.loss_param_ref is None}")
        if ((active_reset and not self.margin_reset and not self.random_margin) or force_keeping) and not self.loss_param_ref is None:
            self.lossParamH.objects["margin"].args = (self.loss_param_ref["margin"].numpy(),)
        if active_reset and not self.margin_reset and self.random_margin and not force_keeping:
            random_margin = round(self.np_rng.uniform(self.random_margin_lower, self.random_margin_higher), 1)
            self.lossParamH.objects["margin"].args = (random_margin, )
        if active_reset and self.margin_reset and not self.random_margin and not force_keeping:
            self.lossParamH.objects["margin"].args = self.original_loss_params

        if not active_reset and not force_keeping:
            self.lossParamH.objects["margin"].args = self.original_loss_params

        return super().compile(self.params["CustomModel"], *args,
                               weighted_metrics=[],
                               **kwargs)

    def define_model_inputs(self) -> Tuple[Layer, ...]:
        """define_model_inputs.

        Parameters
        ----------

        Returns
        -------
        Tuple[layers.Layer, ...]

        """
        positive_input = self.layers.get_handler("input")(name="positive", shape=self.shape)
        anchor_input = self.layers.get_handler("input")(name="anchor", shape=self.shape)
        negative_input = self.layers.get_handler("input")(name="negative", shape=self.shape)
        self.write_msg("Three Input layers created")
        return positive_input, anchor_input, negative_input

    def define_model_output(self, emb, inputs, **kwargs) -> Layer:
        distances = self.layers.get_handler("distance")(**kwargs)(
            emb(inputs[0]),
            emb(inputs[1]),
            emb(inputs[2]),
        )
        self.write_msg("Distance layer initialized")
        return distances

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
