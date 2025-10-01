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

from typing import Optional
import numpy as np
import tensorflow as tf
from tensorflow.core.framework.types_pb2 import DataType
from tensorflow.keras import Model

from SNN2.src.decorators.decorators import cmodel

@cmodel
class ActorCriticModel(Model):

    def __init__(self, reinforceNetwork,
                 gamma, beta):
        super(ActorCriticModel, self).__init__()
        self.reinforceNetwork = reinforceNetwork
        self.gamma = tf.Variable([gamma], dtype=tf.float32)
        self.eps = np.finfo(np.float32).eps.item()
        self.delta = 0.0001
        self.entropy_beta = beta

    def call(self, inputs, **kwargs):
        return self.reinforceNetwork(inputs, **kwargs)

    @tf.function(reduce_retracing=True)
    def train_step(self, data):
        with tf.GradientTape() as tape:
            states = data[0][0]
            actions = data[0][1]
            rewards = data[0][2]

            action_probs, critic_value = self(states, training=True)
            action_probs_taken = tf.gather(action_probs,
                                           tf.cast(tf.reshape(actions, [-1]), tf.int32),
                                           axis=1,
                                           batch_dims=actions.shape[1])
            action_probs_taken = tf.where(action_probs_taken <= self.delta, self.delta, action_probs_taken)
            critic_value = tf.reshape(critic_value, [-1])
            action_probs_taken = tf.math.log(action_probs_taken)
            model_output = tf.stack([action_probs_taken, critic_value], axis=1)
            # tf.print(model_output)

            cce = tf.keras.losses.CategoricalCrossentropy()
            entropy_loss = -self.entropy_beta*cce(action_probs, action_probs)
            loss = self.compiled_loss(model_output, rewards) + entropy_loss

        gradients = tape.gradient(loss, self.reinforceNetwork.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.reinforceNetwork.trainable_weights)
        )

        self.compiled_metrics.update_state(model_output, rewards)
        out = {m.name: m.result() for m in self.metrics}
        return out

    def test_step(self, data):
        action_probs, critic_value = self(data, training=False)
        raise Exception
        self.compiled_loss(aps, ans)

        self.compiled_metrics.update_state(aps, ans)
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        return self(data, training=False)
