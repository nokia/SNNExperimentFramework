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
def applyRequests(df: pd.DataFrame,
                  requests: Dict[str, Dict[str, Any]],
                  *args,
                  window_size: int = 120,
                  stride: int = 1,
                  **kwargs) -> Dict[str, Dict[str, Any]]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    # Compute the mask to remove the windows between one op_id and the next one
    # Assume that every op_id has the same duration, compute it one time
    # using the first op_id
    duration = df.groupby("op_id").count().values[0][0]

    batch_size = (int(len(df)/duration)*duration)-(window_size-1)
    write_msg(f"Batch size: {batch_size}")
    write_msg(f"Stride: {stride}")

    n_true = int((duration-(window_size-1))/stride)
    n_false = int((window_size-1)/stride)
    mask = np.array([[True]*n_true + [False]*n_false])
    mask = np.repeat(mask, len(df)/duration, axis=0).flatten()
    mask = mask[:-n_false]
    mask_idx = np.where(mask)[0]
    #
    # mask = np.array([[True]*(duration-(window_size-1)) + [False]*(window_size-1)])
    # mask = np.repeat(mask, len(df)/duration, axis=0).flatten()
    # mask = mask[:-(window_size-1)]
    # mask_idx = np.where(mask)[0]
    write_msg(f"Mask to apply shape: {mask_idx.shape}")

    for key, req in requests.items():
        write_msg(f"{key} - Requests columns: {req['columns']}", level=LH.INFO)
        if "timestamp" in req["columns"]:
            df["timestamp"] = pd.to_datetime(df["timestamp"].values).astype(int) / 10**9

        tmp_values = df[req["columns"]].values
        write_msg(f"Request values type: {type(tmp_values[0])}", level=LH.DEBUG)
        write_msg(f"Request values: {tmp_values[0]}", level=LH.DEBUG)
        write_msg(f"Request values shape: {tmp_values.shape}", level=LH.DEBUG)
        if "dtype" in req.keys():
            write_msg(f"Trying to convert from {tmp_values.dtype} into {req['dtype']} data type")
            tmp_values = tmp_values.astype(req["dtype"])

        write_msg(f"Including {req['columns']} into {key}['Values'] request object", level=LH.INFO)
        write_msg(f"Chaining values shape for patching {tmp_values.shape}", level=LH.DEBUG)

        tmp_values = np.reshape(tmp_values, (1, tmp_values.shape[0], len(req['columns']), 1))
        write_msg(f"Values reshaped: {tmp_values.shape}")

        stats_tf = tf.image.extract_patches(images=tmp_values,
                                           sizes=[1, window_size, len(req['columns']), 1],
                                           strides=[1, stride, 1, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='VALID')

        stats_tf = tf.reshape(stats_tf, (-1, window_size, len(req['columns'])))
        stats_tf = tf.gather(stats_tf, mask_idx, axis=0)
        write_msg(f"Patches shape: {stats_tf.shape}")

        if req["post_operation"] is not None:
            for operation, o_arg, o_kwarg in zip(req["post_operation"],
                                                 req["post_operation_args"],
                                                 req["post_operation_kwargs"]):
                write_msg(f"Appling post-operation: {operation.__qualname__}")
                stats_tf = operation(stats_tf, *o_arg, **o_kwarg)

        write_msg(f"Appending default values to the request object, type: {type(stats_tf)}", level=LH.DEBUG)
        write_msg(f"Appending default values to the request object, value: {stats_tf}", level=LH.DEBUG)
        req.append_default(stats_tf)

    write_msg("Concatenating the requests")
    for req in requests:
        requests[req].set_default(tf.concat(requests[req].dft(), 0))
    write_msg("Concatenation compleated")

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

@action
def flag_anomaly_windows(df: pd.DataFrame,
                         anomaly_column: str = "anomaly",
                         window_column: str = "window",
                         new_clm_name: str = "anomaly_window",
                         drop_old_clm: bool = True) -> pd.DataFrame:
    """flag_anomaly_windows.
    flag the windows that contain at least one anomaly.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame
    anomaly_column : str
        The column containing the anomaly value default: "anomaly"
    window_column : str
        The column containing the window value, default: "window"
    new_clm_name : str
        The name of the new column that will contain the flag
        default: "anomaly_window"
    drop_old_clm : bool
        Drop the old anomaly column, default: True

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new column "anomaly_window" that flags the
        windows that contain at least one anomaly.
    """
    tmp_df = df.copy()
    tmp_df[anomaly_column] = tmp_df[anomaly_column].astype(int)
    tmp_df[window_column] = tmp_df[window_column].astype(int)
    # If at least one anomaly is present for a single window then the hole
    # window should be considered as anomalous.
    tmp_df[new_clm_name] = tmp_df.groupby(window_column)[anomaly_column].transform("max")
    if drop_old_clm:
        tmp_df.drop(columns=[anomaly_column], inplace=True)
    return tmp_df
