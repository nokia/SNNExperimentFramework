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
