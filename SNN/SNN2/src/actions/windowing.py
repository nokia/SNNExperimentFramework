# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Windowing Actions Module
========================

This module provides actions for creating time series windows and sequence processing
within the SNN2 neural network framework. It contains utilities for:

- Time series windowing with configurable window sizes and strides
- Batch processing of multiple DataFrames with windowing operations
- Anomaly window flagging for time series data
- Post-processing operations on windowed data
- Masking operations to handle experiment boundaries

The module is designed to work with pandas DataFrames and TensorFlow tensors,
providing essential functionality for preparing sequential data for neural network
training and analysis.

Functions
---------
windowing : function
    Create time series windows from DataFrame data with masking.
applyRequests : function
    Apply windowing requests using TensorFlow's extract_patches for efficient processing.
listWindowing : function
    Apply windowing operations to a list of DataFrames and concatenate results.
flag_anomaly_windows : function
    Flag windows containing at least one anomaly for time series anomaly detection.

Notes
-----
All functions in this module use the @action and @f_logger decorators for consistent
logging and action tracking within the SNN2 framework. The module handles time series
data with proper masking to avoid creating windows that span across different
experiments or operational periods.

The windowing operations support various post-processing operations that can be
applied to the generated windows, allowing for flexible data preprocessing pipelines.

Examples
--------
Basic time series windowing workflow:

>>> # Define windowing requests
>>> requests = DataManager()
>>> requests['features'] = {
>>>     'columns': ['feature1', 'feature2'],
>>>     'dtype': np.float32,
>>>     'post_operation': None
>>> }
>>>
>>> # Apply windowing to single DataFrame
>>> windowed_data = windowing(df, requests, window=10)
>>>
>>> # Apply windowing to multiple DataFrames
>>> multi_windowed = listWindowing(df_list, window=6, requests=requests)
>>>
>>> # Flag anomaly windows
>>> flagged_df = flag_anomaly_windows(df, anomaly_column='anomaly')

See Also
--------
SNN2.src.core.data.DataManager : DataManager class for data handling
SNN2.src.decorators.decorators : Action and logging decorators
tensorflow.keras.preprocessing.timeseries_dataset_from_array : TensorFlow windowing utility
"""

from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from SNN2.src.core.data.DataManager import DataManager
from SNN2.src.decorators.decorators import action, f_logger
from SNN2.src.io.logger import LogHandler as LH


@action
@f_logger
def windowing(df: pd.DataFrame,
              requests: Dict[str, Dict[str, Any]],
              window: int,
              **kwargs) -> List[tf.Tensor]:
    """
    Create time series windows from DataFrame data with experiment boundary masking.

    This function creates sliding windows from time series data in a DataFrame,
    with intelligent masking to prevent windows from spanning across different
    experiments or operational periods. It supports post-processing operations
    and handles multiple data requests simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time series data with a 'second' column
        indicating time progression within experiments.
    requests : Dict[str, Dict[str, Any]]
        Dictionary of data extraction requests, where each request contains:
        - 'columns': List of column names to extract
        - 'dtype': Optional data type for conversion
        - 'post_operation': Optional list of operations to apply
        - 'post_operation_args': Arguments for post-operations
        - 'post_operation_kwargs': Keyword arguments for post-operations
    window : int
        Size of the sliding window (number of time steps).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    List[tf.Tensor]
        List of TensorFlow tensors containing windowed data for each request,
        with experiment boundary masking applied.

    Raises
    ------
    Exception
        If window size is larger than the minimum experiment duration.

    Notes
    -----
    The function creates a boolean mask to exclude windows that would span
    across experiment boundaries. This ensures that each window contains
    data from a single experimental run.

    The masking logic assumes that all experiments have the same duration,
    determined by the maximum value in the 'second' column.

    Post-processing operations are applied sequentially if specified in the
    request configuration.

    Examples
    --------
    >>> requests = {
    ...     'features': {
    ...         'columns': ['sensor1', 'sensor2'],
    ...         'dtype': np.float32,
    ...         'post_operation': None
    ...     }
    ... }
    >>> windowed_tensors = windowing(time_series_df, requests, window=10)
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    write_msg(f"Requests: {requests}")
    duration = int(df["second"].max())
    write_msg(f"Minimum duration {duration}")

    if duration < window:
        raise Exception(f"The window value must be lower or equal"\
                        f"to the minimum exp duration: {duration} < {window}")

    batch_size = int(len(df)/duration)
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
                  window_size: int = 120,
                  stride: int = 1,
                  **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Apply windowing requests using TensorFlow's extract_patches for efficient processing.

    This function creates sliding windows from DataFrame data using TensorFlow's
    image patch extraction technique adapted for time series data. It supports
    configurable stride lengths and handles multiple operational IDs with
    proper boundary masking.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time series data with 'op_id' column
        for grouping different operational periods.
    requests : Dict[str, Dict[str, Any]]
        Dictionary of data extraction requests, where each request contains:
        - 'columns': List of column names to extract
        - 'dtype': Optional data type for conversion
        - 'post_operation': Optional list of operations to apply
        - 'post_operation_args': Arguments for post-operations
        - 'post_operation_kwargs': Keyword arguments for post-operations
    window_size : int, default=120
        Size of the sliding window (number of time steps).
    stride : int, default=1
        Step size for sliding the window (1 = no overlap reduction).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Updated requests dictionary with windowed TensorFlow tensors added
        as default values for each request.

    Notes
    -----
    The function uses tf.image.extract_patches to efficiently create sliding
    windows from reshaped time series data. This approach is more efficient
    than traditional looping methods for large datasets.

    Masking is applied to prevent windows from spanning across different
    operational IDs (op_id), ensuring data integrity in multi-experiment
    scenarios.

    Special handling is provided for 'timestamp' columns, which are converted
    to Unix timestamps if present in the request.

    Examples
    --------
    >>> requests = {
    ...     'sensors': {
    ...         'columns': ['temp', 'pressure', 'timestamp'],
    ...         'dtype': np.float32,
    ...         'post_operation': [tf.nn.l2_normalize],
    ...         'post_operation_args': [()],
    ...         'post_operation_kwargs': [{'axis': -1}]
    ...     }
    ... }
    >>> processed_requests = applyRequests(
    ...     df, requests, window_size=60, stride=10
    ... )
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]
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

        write_msg(f"Appending default values to the request object, type:"\
                  f"{type(stats_tf)}", level=LH.DEBUG)
        write_msg(f"Appending default values to the request object, value:"\
                  f"{stats_tf}", level=LH.DEBUG)
        req.append_default(stats_tf)

    write_msg("Concatenating the requests")
    for req in requests:
        requests[req].set_default(tf.concat(requests[req].dft(), 0))
    write_msg("Concatenation compleated")

@action
@f_logger
def listWindowing(dfs: List[pd.DataFrame],
                  window: int = 6,
                  requests: Optional[DataManager] = None,
                  **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Apply windowing operations to a list of DataFrames and concatenate results.

    This function processes multiple DataFrames with the same windowing
    configuration and concatenates the results into unified tensors. It's
    useful for batch processing multiple experimental runs or data files.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of pandas DataFrames to be processed with windowing operations.
        Each DataFrame should have consistent structure and column names.
    window : int, default=6
        Size of the sliding window (number of time steps) to apply to each DataFrame.
    requests : Optional[DataManager], optional
        DataManager instance containing windowing requests configuration.
        Must not be None and should contain request specifications.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        DataManager instance with concatenated windowed tensors from all
        input DataFrames. Each request contains the merged results.

    Raises
    ------
    AssertionError
        If requests parameter is None.

    Notes
    -----
    The function applies the windowing operation to each DataFrame individually
    using the same requests configuration, then concatenates all results along
    the first axis (batch dimension).

    This approach is memory-efficient for processing large numbers of DataFrames
    as it processes them sequentially rather than loading all data into memory
    simultaneously.

    All DataFrames in the input list should have compatible structures to
    ensure successful concatenation.

    Examples
    --------
    >>> # Process multiple experiment files
    >>> dataframes = [pd.read_csv(f'exp_{i}.csv') for i in range(10)]
    >>> requests = DataManager()
    >>> requests['measurements'] = {
    ...     'columns': ['sensor1', 'sensor2'],
    ...     'post_operation': None
    ... }
    >>> combined_windows = listWindowing(dataframes, window=12, requests=requests)
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    assert requests is not None

    write_msg(f"Number of dfs: {len(dfs)}")
    write_msg(f"Requests: {requests}", level=LH.DEBUG)
    for df in dfs:
        sub_objects = windowing(df, requests, window, logger=logger)
        write_msg(f"Length of sub_objects: {len(sub_objects)}")

    for req in requests:
        requests[req].set_default(tf.concat(requests[req].dft(), 0))

    return requests

@action
def flag_anomaly_windows(df: pd.DataFrame,
                         anomaly_column: str = "anomaly",
                         window_column: str = "window",
                         new_clm_name: str = "anomaly_window",
                         drop_old_clm: bool = True) -> pd.DataFrame:
    """
    Flag windows that contain at least one anomaly for time series anomaly detection.

    This function creates a new binary column that flags entire windows as anomalous
    if they contain at least one anomalous data point. This is useful for window-based
    anomaly detection where the presence of any anomaly in a window makes the entire
    window suspicious.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing time series data with anomaly and window information.
    anomaly_column : str, default="anomaly"
        Name of the column containing binary anomaly indicators (0/1 or False/True).
    window_column : str, default="window"
        Name of the column containing window identifiers that group data points
        into windows.
    new_clm_name : str, default="anomaly_window"
        Name of the new column that will contain the window-level anomaly flags.
    drop_old_clm : bool, default=True
        Whether to drop the original anomaly column after creating the window flags.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new window-level anomaly flag column added.
        If drop_old_clm is True, the original anomaly column is removed.

    Notes
    -----
    The function uses pandas groupby with transform('max') to propagate any
    anomaly flag within a window to the entire window. This creates a conservative
    approach where any suspicious activity flags the whole window.

    The input anomaly and window columns are converted to integers before
    processing to ensure consistent data types.

    This approach is particularly useful in scenarios where anomalies might
    have temporal dependencies or where context around anomalous points is
    important for analysis.

    Examples
    --------
    >>> # Flag windows containing anomalies
    >>> flagged_df = flag_anomaly_windows(
    ...     df,
    ...     anomaly_column='is_anomaly',
    ...     window_column='time_window',
    ...     new_clm_name='window_anomaly'
    ... )
    >>>
    >>> # Count anomalous windows
    >>> anomaly_windows = flagged_df['window_anomaly'].sum()
    >>> print(f"Found {anomaly_windows} anomalous windows")
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
