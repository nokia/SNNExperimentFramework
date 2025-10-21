# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from tensorflow.keras.callbacks import Callback
from SNN2.src.decorators.decorators import ccb


@ccb
class verbose(Callback):

    def __init__(self):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_train_end(self, logs=None):
        print("Training END")
