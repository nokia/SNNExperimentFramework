# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import tensorflow as tf

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import ccb
from SNN2.src.io.pickleHandler import PickleHandler as PH

from typing import Any, List

@ccb
class saveEmbeddings(Callback):

    def __init__(self, samples: List[tf.Tensor],
                 emb_labels: List[str],
                 emb_handler,
                 ph: PH):
        self.samples = samples
        self.labels = emb_labels
        assert len(self.samples) == len(self.labels)
        self.ph = ph
        self.embh = emb_handler
        self.iteration = 0

    def on_train_end(self, logs=None):
        for sample, label in zip(self.samples, self.labels):
            sample = sample.batch(1)
            sample = sample.prefetch(tf.data.AUTOTUNE)
            embs = [self.embh(s) for s in sample]
            self.ph.save(embs, f"embedding_{label}_iteration_{self.iteration}",
                         unix_time=True)
        self.iteration += 1

    def on_test_end(self, logs=None):
        for sample, label in zip(self.samples, self.labels):
            sample = sample.batch(1)
            sample = sample.prefetch(tf.data.AUTOTUNE)
            embs = [self.embh(s) for s in sample]
            self.ph.save(embs, f"embedding_{label}_test",
                         unix_time=True)


@ccb
class saveObject(Callback):

    def __init__(self, objects: List[Any],
                 labels: List[str],
                 ph: PH) -> None:
        self.objects = objects
        self.labels = labels
        self.ph = ph
        assert len(self.objects) == len(self.labels)
        self.iteration = 0

    def on_train_end(self, logs=None):
        for sample, label in zip(self.objects, self.labels):
            self.ph.save(sample, f"object_{label}_iteration_{self.iteration}",
                         unix_time=True)
        self.iteration += 1

    def on_test_end(self, logs=None):
        for sample, label in zip(self.objects, self.labels):
            self.ph.save(sample, f"object_{label}_test",
                         unix_time=True)
        self.iteration += 1

