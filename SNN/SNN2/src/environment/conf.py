# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Environment conf module
=======================

Module used to configure the main environment points like
RNG and others.

"""

import random
import numpy as np
import tensorflow as tf

from SNN2.src.util.strings import s
from SNN2.src.io.progressBar import pb
from SNN2.src.params.paramHandler import ParamHandler as Par

def set_rngs(params: Par):
    # Define the rng seed
    # Retro compatibility with numpy seed
    np.random.seed(int(params[s.env_rng]))
    random.seed(int(params[s.env_rng]))
    tf.random.set_seed(int(params[s.env_tf_rng]))

def execute(tqdm_bar, function, *args, **kwargs):
    function(*args, **kwargs)
    tqdm_bar.update(1)

def conf(params: Par):
    pbar = pb.bar(total=1)
    execute(pbar, set_rngs, params)
    pbar.close()
