#!/usr/bin/env python
# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

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
class ClassificationModel(Model):

    def __init__(self, classification_network):
        super(ClassificationModel, self).__init__()
        self.net =classification_network

    def call(self, inputs, **kwargs):
        return self.net(inputs, **kwargs)

    def get_distances(self, data, **kwargs) -> Tuple[Any, Any]:
        dst = self(data, **kwargs)
        return dst

    def train_step(self, data):
        with tf.GradientTape() as tape:
            result = self(data, training=True)
            loss = self.compiled_loss(result)  # type: ignore

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
