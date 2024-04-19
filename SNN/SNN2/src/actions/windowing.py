# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.util.deprecation import inspect
from SNN2.src.core.data.DataManager import DataManager

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from SNN2.src.decorators.decorators import actions, action, f_logger
from SNN2.src.io.logger import LogHandler as LH

from typing import Any, Callable, Dict, Optional, Union, List, Tuple

@action
@f_logger
def windowing(df: pd.DataFrame,
              requests: Dict[str, Dict[str, Any]],
              window: int,
              *args,
              **kwargs) -> List[tf.Tensor]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    write_msg(f"Requests: {requests}")
    duration = int(df["second"].max())
    write_msg(f"Minimum duration {duration}")

    if duration < window:
        raise Exception(f"The window value must be lower or equal to the minimum exp duration: {duration} < {window}")

    batch_size = (int(len(df)/duration)*duration)-(window-1)
    write_msg(f"Batch size: {batch_size}")
    # Mask to remove the inter experiment windows
    mask = np.array([[True]*(duration-(window-1)) + [False]*(window-1)])
    mask = np.repeat(mask, int(len(df)/duration), axis=0).flatten()
    mask = mask[:-(window-1)]

    for key, req in requests.items():
        write_msg(f"{key} - Requests columns: {req['columns']}")
        req["Values"] = df[req["columns"]].values
        write_msg(f"Request values: {req['Values']}")
        if "dtype" in req.keys():
            write_msg(f"Trying to convert from {req['Values'].dtype} into {req['dtype']} data type")
            req["Values"] = req["Values"].astype(req["dtype"])

        write_msg(f"Including {req['columns']} into {key}['Values'] request object")
        write_msg(f"Values: {req['Values']}")

        batch_stats = timeseries_dataset_from_array(req["Values"], None,
                                                    sequence_length=window,
                                                    batch_size=batch_size,
                                                    shuffle=False)
        stats_tf = tf.boolean_mask(list(batch_stats)[0], mask)
        write_msg(f"Applied boolean mask, {stats_tf}", level=LH.DEBUG)

        if req["post_operation"] is not None:
            for operation, o_arg, o_kwarg in zip(req["post_operation"],
                                                 req["post_operation_args"],
                                                 req["post_operation_kwargs"]):
                write_msg(f"Appling {operation.__qualname__}")
                stats_tf = operation(stats_tf, *o_arg, **o_kwarg)

        req.append_default(stats_tf)

    return [requests.dft(req) for req in requests]


@action
@f_logger
def listWindowing(dfs: List[pd.DataFrame],
                  *args,
                  window: int = 6,
                  results_filed: str = "results",
                  requests: Optional[DataManager] = None,
                  **kwargs) -> Dict[str, Dict[str, Any]]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    assert requests is not None

    write_msg(f"Number of dfs: {len(dfs)}")
    write_msg(f"Requests: {requests}", level=LH.DEBUG)
    for df in dfs:
        sub_objects = windowing(df, requests, window, logger=logger)
        write_msg(f"Length of sub_objects: {len(sub_objects)}")
        # for i, key in enumerate(requests.keys()):
        #     print(i, key, list(requests[key].keys()))
        #     print(results_filed not in list(requests[key].keys()))
        #     if results_filed not in list(requests[key].keys()):
        #         requests[key][results_filed] = [sub_objects[i]]
        #     else:
        #         requests[key][results_filed].append(sub_objects[i])
        #     write_msg(f"Intermidiate state requests[{key}][{i}]: {requests[key][results_filed]}")

    for req in requests:
        requests[req].set_default(tf.concat(requests[req].dft(), 0))

    return requests

