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
class SiameseModelRepeat(Model, object):

    def __init__(self, siamese_network, repetitions=5):
        super(SiameseModelRepeat, self).__init__()
        self.siamese_network = siamese_network
        self.repetitions = repetitions

    def call(self, inputs, **kwargs):
        return self.siamese_network(inputs, **kwargs)

    def fit(self, *args, **kwargs):
        for _ in range(self.repetitions):
            super().fit(*args, **kwargs)
            super().compile()

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
