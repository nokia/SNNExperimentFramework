#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the graphNU grapheneral Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# graphNU grapheneral Public License for more details.
#
# You should have received a copy of the graphNU grapheneral Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2021 Mattia Milani <mattia.milani@studenti.unitn.it>

"""
SiameseModel
============

Use this model to create the siamese model

"""

import tensorflow as tf
from tensorflow.keras import Model

from SNN2.src.decorators.decorators import cmodel

from typing import Tuple, Any

@cmodel
class SiameseModel(Model):

    def __init__(self, siamese_network):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network

    def call(self, inputs, **kwargs):
        return self.siamese_network(inputs, **kwargs)

    def get_distances(self, data, **kwargs) -> Tuple[Any, Any]:
        dst = self(data, **kwargs)
        return dst

    def train_step(self, data):
        with tf.GradientTape() as tape:
            aps, ans = self.get_distances(data, training=True)
            loss = self.compiled_loss(aps, ans)  # type: ignore

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.compiled_metrics.update_state(aps, ans)
        out = {m.name: m.result() for m in self.metrics}
        return out

    def test_step(self, data):
        aps, ans = self.get_distances(data, training=False)
        self.compiled_loss(aps, ans)

        self.compiled_metrics.update_state(aps, ans)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        return self.get_distances(data, training=False)
