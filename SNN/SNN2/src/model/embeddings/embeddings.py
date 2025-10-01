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

