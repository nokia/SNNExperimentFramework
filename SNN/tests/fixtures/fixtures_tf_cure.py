# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import tensorflow as tf

from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES

@pytest.fixture
def tf_sample_LSUN():
    return tf.convert_to_tensor(read_sample(FCPS_SAMPLES.SAMPLE_LSUN))


