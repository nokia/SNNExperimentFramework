# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import functools

from SNN2.src.model.metrics.metrics import distanceAccuracy

from SNN2.src.decorators.decorators import metrics

def Metrics_selector(obj, *args, **kwargs):
    if obj in metrics:
        return metrics[obj](*args, **kwargs)
    else:
        raise ValueError(f"Metric \"{function}\" not available")

