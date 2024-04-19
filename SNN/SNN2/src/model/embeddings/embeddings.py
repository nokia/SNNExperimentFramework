# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
class module
============

Use this module to manage all the parameters of a class.
"""

from __future__ import annotations


from SNN2.src.io.logger import LogHandler as LH
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential

from SNN2.src.decorators.decorators import cembedding


@cembedding
class defaultEmb:

    def none_log(self, *args, **kwargs):
        pass

    def __init__(self, layers,
                 logger: LH = None,
                 **kwargs):
        self.layers: Model = Sequential(layers, **kwargs)
        self.logger = logger

        if self.logger is None:
            self.logger = self.none_log
        self.logger(self.__class__.__name__, f"Embedding loaded")

    def __str__(self):
        return f"Layer embedding, call layers to have all the layers"

@cembedding
class ActorCriticEmb():

    def none_log(self, *args, **kwargs):
        pass

    def __init__(self, action_layer,
                 critic_layer,
                 input_layer,
                 common_layer,
                 *args,
                 logger: LH = None,
                 **kwargs):
        self.logger = logger
        self.inputs = input_layer
        self.common = common_layer(self.inputs)
        self.action = action_layer(self.common)
        self.critic = critic_layer(self.common)

        self.layers = Model(inputs=self.inputs,
                            outputs=[self.action, self.critic],
                            **kwargs)

        if self.logger is None:
            self.logger = self.none_log
        self.logger(self.__class__.__name__, f"Reinforcement actor critic embedding loaded")

    def __str__(self):
        return f"Layer embedding, call layers to have all the layers"

