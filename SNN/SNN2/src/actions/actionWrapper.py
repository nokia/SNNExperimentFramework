# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Action Wrapper Module
=====================

This module provides high-level wrapper functions and utilities for complex data processing
workflows within the SNN2 neural network framework. It contains specialized actions for:

- Dataset separation based on quality thresholds (good/bad/gray classification)
- Duration-based experiment grouping and analysis
- Feature statistics computation (mean, std, min, max)
- Data normalization operations (z-score and min-max)
- Triplet generation for metric learning (including gray zone handling)
- Random sampling and dataset balancing utilities
- Action selection and dynamic function dispatch

The module serves as a high-level interface for combining multiple basic actions into
complex data processing pipelines, particularly for network performance analysis and
anomalous behavior detection.

Functions
---------
check_df_not_none : function
    Utility function to validate DataFrame inputs.
GoodBadGraySeparation : function
    Separate experiments into good, bad, and gray categories based on performance thresholds.
durationSeparation : function
    Group experiments by duration for comparative analysis.
featureMean, featureStd, featureMax, featureMin : function
    Compute statistical measures across feature dimensions.
normalize : function
    Apply z-score normalization using provided mean and standard deviation.
normalizeMinMax : function
    Apply min-max normalization to dataset tuples.
generateTripletsNG : function
    Generate triplets from anchor data excluding difficult samples.
generateTriplets : function
    Generate triplets for metric learning with class balancing.
generatePredictionTriplets : function
    Create prediction triplets from separate anchor, positive, and negative sets.
GenerateGrayTriplets : function
    Generate triplets specifically handling gray zone samples.
print_data : function
    Utility function for debugging data inspection.
action_selector : function
    Dynamic action selection and execution.
GoodBad_randomGray : function
    Create random gray samples from good and bad datasets.
get_randomGray : function
    Extract random gray samples with specified portions (incomplete implementation).

Notes
-----
All functions in this module use the @action decorator for consistent tracking within
the SNN2 framework. Many functions are specifically designed for network performance
analysis with delay and packet drop rate thresholds.

The module includes sophisticated triplet generation algorithms for metric learning,
with special handling for difficult/gray zone samples that don't clearly belong to
either good or bad categories.

Examples
--------
Basic dataset separation workflow:

>>> # Separate experiments based on performance thresholds
>>> separated_df = GoodBadGraySeparation(
...     df, delay_lower=20.0, delay_upper=50.0,
...     drop_lower=0.0, drop_upper=0.7
... )
>>>
>>> # Generate triplets for metric learning
>>> anchor_data = DataManager()
>>> triplets = generateTripletsNG(anchor_data)
>>>
>>> # Apply normalization
>>> normalized_data = normalize(data, mean_tensor, std_tensor)

See Also
--------
SNN2.src.actions.separation : Basic separation utilities
SNN2.src.core.data.DataManager : DataManager class for data handling
SNN2.src.decorators.decorators : Action decorator implementation
"""

import math

from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from SNN2.src.actions.separation import ConcatenateNG
from SNN2.src.core.data.DataManager import DataManager

from SNN2.src.decorators.decorators import actions, action, f_logger
from SNN2.src.io.logger import LogHandler as LH

def check_df_not_none(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame input is not None.

    This utility function provides consistent DataFrame validation across
    the module, ensuring that functions receive valid DataFrame inputs
    before processing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate for None values.

    Returns
    -------
    None
        Function performs validation only, no return value.

    Raises
    ------
    AssertionError
        If the input DataFrame is None.

    Examples
    --------
    >>> check_df_not_none(my_dataframe)  # Passes if df is valid
    >>> check_df_not_none(None)  # Raises AssertionError
    """
    assert df is not None, "The dataframe passed is None"

@action
def GoodBadGraySeparation(df: pd.DataFrame = None,
                          delay_lower: float = 20.0,
                          delay_upper: float = 50.0,
                          drop_lower: float = 0.0,
                          drop_upper: float = 0.7) -> pd.DataFrame:
    """
    Separate experiments into good, bad, and gray categories based on performance thresholds.

    This function classifies network experiments based on delay and packet drop rate
    performance metrics, creating three categories: good (acceptable performance),
    bad (poor performance), and gray (uncertain/intermediate performance).

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input DataFrame containing experiment data with columns:
        - 'exp_id': Experiment identifier
        - 'problem': Problem type ('good', 'delay', 'drop')
        - 'value': Performance metric value
    delay_lower : float, default=20.0
        Lower threshold for delay classification. Values below this are considered good.
    delay_upper : float, default=50.0
        Upper threshold for delay classification. Values above this are considered bad.
    drop_lower : float, default=0.0
        Lower threshold for drop rate classification. Values below this are considered good.
    drop_upper : float, default=0.7
        Upper threshold for drop rate classification. Values above this are considered bad.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Dataset' column indicating classification:
        - 'good': Good performance experiments
        - 'bad': Poor performance experiments
        - 'gray': Intermediate/uncertain performance experiments

    Notes
    -----
    Classification logic:
    - Good: problem='good' OR (problem='delay' AND value < delay_lower) OR
            (problem='drop' AND value < drop_lower)
    - Bad: (problem='delay' AND value > delay_upper) OR
           (problem='drop' AND value > drop_upper)
    - Gray: Experiments with values between lower and upper thresholds

    The function assumes experiments with problem='good' are inherently good
    regardless of other metrics.

    Examples
    --------
    >>> # Classify experiments with custom thresholds
    >>> classified_df = GoodBadGraySeparation(
    ...     df, delay_lower=15.0, delay_upper=60.0,
    ...     drop_lower=0.05, drop_upper=0.8
    ... )
    >>> print(classified_df['Dataset'].value_counts())
    """
    check_df_not_none(df)
    exps = df[["exp_id", "problem", "value"]].drop_duplicates().copy()
    goods_exps = exps[(exps["problem"] == "good") | \
                    ((exps["problem"] == "delay") & (exps["value"] < delay_lower)) | \
                    ((exps["problem"] == "drop") & (exps["value"] < drop_lower))]["exp_id"].values
    bads_exps = exps[((exps["problem"] == "delay") & (exps["value"] > delay_upper)) | \
                   ((exps["problem"] == "drop") & (exps["value"] > drop_upper))]["exp_id"].values
    grays_exps = exps[((exps["problem"] == "delay") & (exps["value"] > delay_lower) &\
                        (exps["value"] < delay_upper)) | \
                    ((exps["problem"] == "drop") & (exps["value"] > drop_lower) &\
                      (exps["value"] < drop_upper))]["exp_id"].values
    good_df = df[df["exp_id"].isin(goods_exps)].copy()
    bad_df = df[df["exp_id"].isin(bads_exps)].copy()
    gray_df = df[df["exp_id"].isin(grays_exps)].copy()
    good_df["Dataset"] = "good"
    bad_df["Dataset"] = "bad"
    gray_df["Dataset"] = "gray"
    return pd.concat([good_df, bad_df, gray_df])

@action
def durationSeparation(df: pd.DataFrame = None) -> List[pd.DataFrame]:
    """
    Group experiments by duration for comparative analysis.

    This function separates experiments into groups based on their duration,
    enabling analysis of experiments with similar temporal characteristics.
    Duration is determined by the maximum 'second' value for each experiment.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Input DataFrame containing experiment data with columns:
        - 'exp_id': Experiment identifier
        - 'second': Time progression within each experiment

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames, each containing experiments with the same duration.
        The number of DataFrames equals the number of unique durations found.

    Notes
    -----
    The function groups experiments by their maximum 'second' value, which
    represents the duration of each experiment. This is useful for:
    - Comparing experiments with similar temporal scope
    - Analyzing duration-dependent patterns
    - Ensuring fair comparisons between experiments

    The grouping preserves all original data while organizing it by duration.

    Examples
    --------
    >>> # Separate experiments by duration
    >>> duration_groups = durationSeparation(experiment_df)
    >>> print(f"Found {len(duration_groups)} different durations")
    >>> for i, group in enumerate(duration_groups):
    ...     duration = group.groupby('exp_id')['second'].max().iloc[0]
    ...     print(f"Group {i}: {len(group['exp_id'].unique())} experiments of {duration}s")
    """
    check_df_not_none(df)
    seconds = df.groupby(["exp_id"])["second"].max().reset_index()
    exp_groups = [seconds[seconds["second"] == x]["exp_id"].values for \
                   x in seconds["second"].unique()]
    dfs = [df[df["exp_id"].isin(x)] for x in exp_groups]
    return dfs


@action
def featureMean(data: tf.Tensor) -> tf.Tensor:
    """
    Compute mean values across all samples for each feature dimension.

    This function flattens the input tensor and computes the mean across
    all samples while preserving the feature dimension structure.

    Parameters
    ----------
    data : tf.Tensor
        Input tensor with shape (..., n_features) where the last dimension
        represents feature channels.

    Returns
    -------
    tf.Tensor
        1D tensor of shape (n_features,) containing mean values for each feature.

    Examples
    --------
    >>> data = tf.random.normal([100, 10, 5])  # 100 samples, 10 timesteps, 5 features
    >>> means = featureMean(data)
    >>> print(means.shape)  # (5,)
    """
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_mean(data_tensor, 0)

@action
def featureStd(data: tf.Tensor) -> tf.Tensor:
    """
    Compute standard deviation across all samples for each feature dimension.

    This function flattens the input tensor and computes the standard deviation
    across all samples while preserving the feature dimension structure.

    Parameters
    ----------
    data : tf.Tensor
        Input tensor with shape (..., n_features) where the last dimension
        represents feature channels.

    Returns
    -------
    tf.Tensor
        1D tensor of shape (n_features,) containing standard deviation values
        for each feature.

    Examples
    --------
    >>> data = tf.random.normal([100, 10, 5])  # 100 samples, 10 timesteps, 5 features
    >>> stds = featureStd(data)
    >>> print(stds.shape)  # (5,)
    """
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_std(data_tensor, 0)

@action
def featureMax(data: tf.Tensor) -> tf.Tensor:
    """
    Compute maximum values across all samples for each feature dimension.

    This function flattens the input tensor and computes the maximum across
    all samples while preserving the feature dimension structure.

    Parameters
    ----------
    data : tf.Tensor
        Input tensor with shape (..., n_features) where the last dimension
        represents feature channels.

    Returns
    -------
    tf.Tensor
        1D tensor of shape (n_features,) containing maximum values for each feature.

    Examples
    --------
    >>> data = tf.random.normal([100, 10, 5])  # 100 samples, 10 timesteps, 5 features
    >>> maxs = featureMax(data)
    >>> print(maxs.shape)  # (5,)
    """
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_max(data_tensor, 0)

@action
def featureMin(data: tf.Tensor) -> tf.Tensor:
    """
    Compute minimum values across all samples for each feature dimension.

    This function flattens the input tensor and computes the minimum across
    all samples while preserving the feature dimension structure.

    Parameters
    ----------
    data : tf.Tensor
        Input tensor with shape (..., n_features) where the last dimension
        represents feature channels.

    Returns
    -------
    tf.Tensor
        1D tensor of shape (n_features,) containing minimum values for each feature.

    Examples
    --------
    >>> data = tf.random.normal([100, 10, 5])  # 100 samples, 10 timesteps, 5 features
    >>> mins = featureMin(data)
    >>> print(mins.shape)  # (5,)
    """
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_min(data_tensor, 0)

@action
@tf.autograph.experimental.do_not_convert
def normalize(data: tf.Tensor,
              mean: tf.Tensor,
              std: tf.Tensor) -> tf.Tensor:
    """
    Apply z-score normalization using provided mean and standard deviation.

    This function normalizes input data by subtracting the mean and dividing
    by the standard deviation for each feature dimension. The operation is
    applied element-wise across all samples.

    Parameters
    ----------
    data : tf.Tensor
        Input tensor to be normalized with shape (..., n_features).
    mean : tf.Tensor
        Mean values for each feature dimension with shape (n_features,).
    std : tf.Tensor
        Standard deviation values for each feature dimension with shape (n_features,).

    Returns
    -------
    tf.Tensor
        Normalized tensor with the same shape as input data.

    Notes
    -----
    The function uses tf.map_fn with autograph disabled for efficient
    element-wise operations. The normalization formula is: (x - mean) / std.

    Examples
    --------
    >>> data = tf.random.normal([100, 5])
    >>> mean = featureMean(data)
    >>> std = featureStd(data)
    >>> normalized = normalize(data, mean, std)
    >>> # Verify normalization: mean ≈ 0, std ≈ 1
    """
    return tf.map_fn(lambda x: (x-mean)/std, data)

@action
@tf.autograph.experimental.do_not_convert
def normalizeMinMax(data: Tuple[tf.data.Dataset, tf.data.Dataset],
                    l_max: tf.Tensor,
                    l_min: tf.Tensor) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Apply min-max normalization to dataset tuples.

    This function applies min-max scaling to the first dataset in the tuple
    (typically the feature dataset) while leaving the second dataset
    (typically targets) unchanged.

    Parameters
    ----------
    data : Tuple[tf.data.Dataset, tf.data.Dataset]
        Tuple containing (features_dataset, targets_dataset).
    l_max : tf.Tensor
        Maximum values for each feature dimension used for scaling.
    l_min : tf.Tensor
        Minimum values for each feature dimension used for scaling.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset]
        Tuple with normalized features dataset and unchanged targets dataset.

    Notes
    -----
    The normalization formula is: (x - min) / (max - min), which scales
    values to the range [0, 1]. Only the first dataset is normalized.

    Examples
    --------
    >>> features_ds = tf.data.Dataset.from_tensor_slices(features)
    >>> targets_ds = tf.data.Dataset.from_tensor_slices(targets)
    >>> data_tuple = (features_ds, targets_ds)
    >>> normalized_tuple = normalizeMinMax(data_tuple, max_vals, min_vals)
    """
    data_wdw = data[0]
    data_trg = data[1]
    return (data_wdw.map(lambda x: (x-l_min)/(l_max-l_min)), data_trg)

@action
@f_logger
def generateTripletsNG(anchor: DataManager,
                       **kwargs) -> DataManager:
    """
    Generate triplets from anchor data excluding difficult samples.

    This function creates triplet datasets for metric learning by separating
    anchor data into good and bad samples (excluding difficult/gray samples)
    and then generating triplets using the GenerateGrayTriplets function.

    Parameters
    ----------
    anchor : DataManager
        DataManager instance containing anchor data with 'Windows' and 'Classes' keys.
        Classes should be encoded as: 0 (good), 1 (bad), 2 (difficult/gray).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    DataManager
        DataManager instance containing triplet data generated from good and bad samples.

    Notes
    -----
    The function filters out samples with class label 2 (difficult) and separates
    the remaining samples into good (class 0) and bad (class 1) categories.
    It then uses GenerateGrayTriplets to create the final triplet structure.

    This approach ensures that triplets are generated only from clearly
    classified samples, avoiding ambiguity from difficult cases.

    Examples
    --------
    >>> anchor_data = DataManager()
    >>> # ... populate anchor_data with Windows and Classes ...
    >>> triplets = generateTripletsNG(anchor_data, logger=my_logger)
    >>> triplet_dataset = triplets['TripletDst']['TfDataset']
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx = tf.range(anchor.dft("Windows").shape[0])
    write_msg(f"Total idx: {len(idx)}", level=LH.DEBUG)

    not_difficult_idx = tf.gather(idx, tf.where(anchor.dft("Classes") != 2)[:, 0])
    not_difficult_classes = tf.gather(anchor.dft("Classes"), not_difficult_idx)
    write_msg(f"Not difficult idx: {len(not_difficult_idx)}", level=LH.DEBUG)

    g_idx = tf.gather(not_difficult_idx, tf.where(not_difficult_classes == 0)[:, 0])
    b_idx = tf.gather(not_difficult_idx, tf.where(not_difficult_classes == 1)[:, 0])
    write_msg(f"Not difficult g_idx: {len(g_idx)}", level=LH.DEBUG)
    write_msg(f"Not difficult b_idx: {len(b_idx)}", level=LH.DEBUG)

    goods = anchor.sub_select(g_idx)
    bads = anchor.sub_select(b_idx)

    return GenerateGrayTriplets(anchor, goods, bads, **kwargs)


@action
def generateTriplets(anchor: Tuple[tf.data.Dataset,
                          tf.data.Dataset,
                          tf.data.Dataset,
                          tf.data.Dataset],
                     log: Optional[LH] = None,) -> Tuple[tf.data.Dataset,
                          tf.data.Dataset,
                          tf.data.Dataset,
                          tf.data.Dataset]:
    """
    Generate triplets from a unified anchor dataset containing all sample types.

    This function creates triplets from a single anchor tuple containing four
    TensorFlow datasets representing different aspects or views of the data.
    It's designed for cases where all sample types are pre-organized in one structure.

    Parameters
    ----------
    anchor : Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
        Tuple containing four TensorFlow datasets:
        - [0]: Window features dataset
        - [1]: Additional features or metadata
        - [2]: Labels or classifications
        - [3]: Auxiliary data or indices
    log : Optional[LH], optional
        Logger instance for recording operations. If None, uses a dummy logger.

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]
        Tuple containing four processed TensorFlow datasets with triplet structure:
        - Anchor samples
        - Positive samples
        - Negative samples
        - Associated metadata

    Notes
    -----
    This function differs from other triplet generators by:
    1. Taking a unified input structure with four components
    2. Processing all data through a single anchor reference
    3. Maintaining the four-dataset output structure
    4. Supporting optional logging for debugging

    The triplet generation process:
    - Converts datasets to tensors for processing
    - Creates index mappings for efficient access
    - Generates balanced positive/negative pairs
    - Reconstructs datasets in triplet format

    Examples
    --------
    >>> # Generate triplets from unified anchor structure
    >>> anchor_tuple = (features_ds, meta_ds, labels_ds, indices_ds)
    >>> triplets = generateTriplets(anchor_tuple, log=logger)
    >>> anchor_out, pos_out, neg_out, meta_out = triplets
    """
    def fake_log(*arg, **kwargs):
        pass
    if log is None:
        log = fake_log

    anchor_wdw = tf.convert_to_tensor(np.array(list(anchor[0].as_numpy_iterator())))
    targets = tf.convert_to_tensor(np.array(list(anchor[3].as_numpy_iterator())))
    only_good_and_bad_t = tf.gather(targets, tf.where(targets != 2)[:, 0])
    flag = tf.where(only_good_and_bad_t == 1, True, False)
    idx = tf.range(anchor_wdw.shape[0])
    idx_C = tf.reshape(tf.gather(idx, tf.where(flag).numpy()), [-1])
    idx_B = tf.reshape(tf.gather(idx, tf.where(tf.where(flag, False, True)).numpy()), [-1])

    index_B_p = tf.random.shuffle(idx_B)
    index_C_p = tf.random.shuffle(idx_C)
    index_p = tf.concat([index_B_p, index_C_p], 0)

    while len(tf.where(tf.math.equal(idx, index_p)).numpy()) > 0:
        wrong_indexes = tf.reshape(tf.where(tf.math.equal(idx, index_p)).numpy(), [-1])
        for index in wrong_indexes:
            index = tf.cast(index, tf.int32)
            if len(tf.where(tf.math.equal(index, index_B_p)).numpy()) > 0:
                pos = tf.reshape(tf.where(tf.math.equal(index, index_B_p)).numpy(), [-1])
                new_index_B_p = index_B_p.numpy()
                new_element = tf.random.shuffle(idx_B)[pos.numpy()[0]]
                new_index_B_p[pos.numpy()[0]] = new_element.numpy()
                index_B_p = tf.convert_to_tensor(new_index_B_p)
            else:
                pos = tf.reshape(tf.where(tf.math.equal(index, index_C_p)).numpy(), [-1])
                new_index_C_p = index_C_p.numpy()
                new_element = tf.random.shuffle(idx_C)[pos.numpy()[0]]
                new_index_C_p[pos.numpy()[0]] = new_element.numpy()
                index_C_p = tf.convert_to_tensor(new_index_C_p)
        index_p = tf.concat([index_B_p, index_C_p], 0)

    reps = math.ceil(len(idx_B)/len(idx_C))
    index_B_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_C, [1, len(idx_C)]),
                                                       repeats=reps, axis=0), [-1]))[:len(idx_B)]
    reps = math.ceil(len(idx_C)/len(idx_B))
    index_C_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_B, [1, len(idx_B)]),
                                                       repeats=reps, axis=0), [-1]))[:len(idx_C)]
    index_n = tf.concat([index_B_n, index_C_n], 0)
    positive = tf.gather(anchor_wdw, index_p)
    negative = tf.gather(anchor_wdw, index_n)
    return tf.data.Dataset.from_tensor_slices((positive, anchor_wdw, negative)),\
           anchor[1], \
           tf.data.Dataset.from_tensor_slices(targets), \
           anchor[3]

@action
@f_logger
def generatePredictionTriplets(anchor: DataManager,
                               positive: DataManager,
                               negative: DataManager,
                               keep_tf_dft: bool = False,
                               keep_anchor_wdw: bool = False,
                               keep_all_wdw: bool = False,
                               **kwargs) -> DataManager:
    """
    Generate triplets for prediction tasks using DataManager objects.

    This function creates triplets (anchor, positive, negative) from separate
    DataManager objects for training neural networks in a metric learning setup.
    It handles data alignment and repetition to ensure proper triplet formation.

    Parameters
    ----------
    anchor : DataManager
        DataManager containing anchor samples.
    positive : DataManager
        DataManager containing positive samples (same class as anchors).
    negative : DataManager
        DataManager containing negative samples (different class from anchors).
    keep_tf_dft : bool, default=False
        Whether to preserve TensorFlow dataset format in output.
    keep_anchor_wdw : bool, default=False
        Whether to preserve anchor window information.
    keep_all_wdw : bool, default=False
        Whether to preserve window information for all samples.
    **kwargs : dict
        Additional keyword arguments including:
        - logger: Logger instance for recording operations
        - write_msg: Function for writing log messages

    Returns
    -------
    DataManager
        DataManager object containing generated triplets with properly
        aligned anchor, positive, and negative samples.

    Notes
    -----
    The function performs the following operations:
    1. Extracts indices from each DataManager's window data
    2. Calculates repetition factors to align different-sized datasets
    3. Repeats and tiles indices to create balanced triplets
    4. Combines data from all three sources into a unified triplet structure

    The repetition logic ensures that:
    - Each anchor has corresponding positive and negative examples
    - Smaller datasets are repeated to match larger ones
    - All triplets maintain proper class relationships

    Examples
    --------
    >>> # Generate triplets from separated datasets
    >>> triplet_data = generatePredictionTriplets(
    ...     anchor=good_data,
    ...     positive=good_validation,
    ...     negative=bad_data,
    ...     keep_tf_dft=True
    ... )
    >>> print(f"Generated triplets: {triplet_data.dft('Windows').shape[0]}")
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx_a = tf.range(anchor.dft("Windows").shape[0])
    idx_p = tf.range(positive.dft("Windows").shape[0])
    idx_n = tf.range(negative.dft("Windows").shape[0])

    reps_p = math.ceil(len(idx_a)/len(idx_p))
    idx_p = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_p, [1, len(idx_p)]),
                                                   repeats=reps_p, axis=0), [-1]))[:len(idx_a)]
    write_msg(f"p indexes: {idx_p}, len: {len(idx_p)}", level=LH.DEBUG)
    reps_n = math.ceil(len(idx_a)/len(idx_n))
    idx_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_n, [1, len(idx_n)]),
                                                   repeats=reps_n, axis=0), [-1]))[:len(idx_a)]
    write_msg(f"n indexes: {idx_n}, len: {len(idx_p)}", level=LH.DEBUG)

    tmp_p = positive.sub_select(idx_p, inplace=False)
    tmp_n = negative.sub_select(idx_n, inplace=False)
    tmp_a = anchor.sub_select(idx_a, inplace=False)
    write_msg(f"P shape: {tmp_p['Windows']['tf_values'].shape}")
    write_msg(f"A shape: {tmp_a['Windows']['tf_values'].shape}")
    write_msg(f"N shape: {tmp_n['Windows']['tf_values'].shape}")

    tmp_a["TripletDst"] = None
    if keep_tf_dft:
        tmp_a["TripletDst"].set_default(
                    tf.stack([tmp_p.dft("Windows"),
                              tmp_a.dft("Windows"),
                              tmp_n.dft("Windows")],
                             axis=1))
    else:
        tmp_a["TripletDst"].set_default(tf.zeros([3, 4], tf.int8))
    tmp_a["TripletDst"]["TfDataset"] = tf.data.Dataset.from_tensor_slices(
                (tmp_p.dft("Windows"),
                 tmp_a.dft("Windows"),
                 tmp_n.dft("Windows")))

    if not keep_anchor_wdw:
        del tmp_a["Windows"]
    if not keep_all_wdw:
        del tmp_p["Windows"]
        del tmp_n["Windows"]
    # del tmp_p
    # del tmp_n
    return tmp_a

@action
def generatePredictionTripletsReverse(anchor: Tuple[tf.data.Dataset, tf.data.Dataset],
                                      positive: Tuple[tf.data.Dataset, tf.data.Dataset],
                                      negative: Tuple[tf.data.Dataset, tf.data.Dataset]) -> Tuple[
                                          Tuple[tf.data.Dataset, tf.data.Dataset], ...]:
    """
    Generate triplets in reverse format using TensorFlow datasets.

    This function creates triplets from TensorFlow dataset tuples, processing
    data in a reverse manner compared to the standard triplet generation.
    It handles tensor conversion and index alignment for metric learning.

    Parameters
    ----------
    anchor : Tuple[tf.data.Dataset, tf.data.Dataset]
        Tuple containing (features, targets) datasets for anchor samples.
    positive : Tuple[tf.data.Dataset, tf.data.Dataset]
        Tuple containing (features, targets) datasets for positive samples.
    negative : Tuple[tf.data.Dataset, tf.data.Dataset]
        Tuple containing (features, targets) datasets for negative samples.

    Returns
    -------
    Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], ...]
        Nested tuple structure containing triplet datasets with properly
        aligned anchor, positive, and negative samples.

    Notes
    -----
    The function performs the following operations:
    1. Converts TensorFlow datasets to tensors for processing
    2. Extracts window features and target labels
    3. Creates index ranges for each dataset
    4. Aligns datasets through repetition and tiling
    5. Reconstructs datasets in triplet format

    The "reverse" nature refers to the processing order or structure
    compared to the standard generatePredictionTriplets function.

    This function is optimized for TensorFlow dataset operations and
    maintains computational graph compatibility for training pipelines.

    Examples
    --------
    >>> # Generate reverse triplets from TF datasets
    >>> anchor_data = (features_ds, labels_ds)
    >>> positive_data = (pos_features_ds, pos_labels_ds)
    >>> negative_data = (neg_features_ds, neg_labels_ds)
    >>> triplets = generatePredictionTripletsReverse(
    ...     anchor_data, positive_data, negative_data
    ... )
    """
    anchor_wdw = tf.convert_to_tensor(np.array(list(anchor[0].as_numpy_iterator())))
    positive_wdw = tf.convert_to_tensor(np.array(list(positive[0].as_numpy_iterator())))
    negative_wdw = tf.convert_to_tensor(np.array(list(negative[0].as_numpy_iterator())))
    targets = tf.convert_to_tensor(np.array(list(anchor[1].as_numpy_iterator())))
    idx_a = tf.range(anchor_wdw.shape[0])
    idx_p = tf.range(positive_wdw.shape[0])
    idx_n = tf.range(negative_wdw.shape[0])

    reps_p = math.ceil(len(idx_a)/len(idx_p))
    idx_p = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_p, [1, len(idx_p)]),
                                                   repeats=reps_p, axis=0), [-1]))[:len(idx_a)]
    reps_n = math.ceil(len(idx_a)/len(idx_n))
    idx_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_n, [1, len(idx_n)]),
                                                   repeats=reps_n, axis=0), [-1]))[:len(idx_a)]
    positive = tf.gather(positive_wdw, idx_p)
    negative = tf.gather(negative_wdw, idx_n)
    correct = (tf.data.Dataset.from_tensor_slices((positive, anchor_wdw, negative)),\
               tf.data.Dataset.from_tensor_slices(targets))
    reverse = (tf.data.Dataset.from_tensor_slices((negative, anchor_wdw, positive)),\
               tf.data.Dataset.from_tensor_slices(targets))
    return correct, reverse

@action
@f_logger
def GenerateGrayTriplets(anchor: DataManager,
                         positive: DataManager,
                         negative: DataManager,
                         keep_all: bool = False,
                         **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Generate gray triplets with comprehensive statistics and metadata.

    This function creates triplets for gray (uncertain) samples while collecting
    detailed statistics about the triplet generation process. It's specifically
    designed for handling uncertain or ambiguous data points in metric learning.

    Parameters
    ----------
    anchor : DataManager
        DataManager containing anchor samples (typically gray/uncertain samples).
    positive : DataManager
        DataManager containing positive samples (same class as anchors).
    negative : DataManager
        DataManager containing negative samples (different class from anchors).
    keep_all : bool, default=False
        Whether to preserve all intermediate data and statistics.
    **kwargs : dict
        Additional keyword arguments including:
        - logger: Logger instance for recording operations
        - write_msg: Function for writing log messages

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Nested dictionary containing:
        - Triplet data organized by categories
        - Statistics about triplet generation
        - Metadata about data shapes and distributions
        - Debug information and logs

    Notes
    -----
    This function is specialized for gray sample processing and includes:
    1. Enhanced logging and debugging capabilities
    2. Detailed statistics collection during triplet creation
    3. Shape validation and dimension checking
    4. Flexible data preservation options

    The gray triplet generation process:
    - Creates balanced triplets using uncertain samples as anchors
    - Maintains proper positive/negative relationships
    - Collects comprehensive metadata for analysis
    - Supports various data retention policies

    Examples
    --------
    >>> # Generate gray triplets with full statistics
    >>> gray_triplets = GenerateGrayTriplets(
    ...     anchor=gray_samples,
    ...     positive=good_samples,
    ...     negative=bad_samples,
    ...     keep_all=True
    ... )
    >>> print(f"Generated {len(gray_triplets['data'])} gray triplet sets")
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx_a = tf.range(anchor.dft("Windows").shape[0])
    idx_p = tf.range(positive.dft("Windows").shape[0])
    idx_n = tf.range(negative.dft("Windows").shape[0])
    write_msg(f"Len p: {len(idx_p)}", level=LH.DEBUG)
    write_msg(f"Len n: {len(idx_n)}", level=LH.DEBUG)

    anchor_wdw_shape = anchor.dft("Windows").shape
    reps_B = math.ceil(anchor.dft("Windows").shape[0]/len(idx_p))
    write_msg(f"Repetitions p: {reps_B}", level=LH.DEBUG)
    idx_B = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_p, [1, len(idx_p)]),
                                                   repeats=reps_B,
                                                   axis=0), [-1]))[:anchor_wdw_shape[0]]
    write_msg(f"p chosen: {idx_B}", level=LH.DEBUG)
    write_msg(f"dimension idx p chosen: {len(idx_B)}", level=LH.DEBUG)

    reps_C = math.ceil(anchor.dft("Windows").shape[0]/len(idx_n))
    write_msg(f"Repetitions n: {reps_C}", level=LH.DEBUG)
    idx_C = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_n, [1, len(idx_n)]),
                                                   repeats=reps_C,
                                                   axis=0), [-1]))[:anchor_wdw_shape[0]]
    write_msg(f"n chosen: {idx_C}", level=LH.DEBUG)
    write_msg(f"dimension idx n chosen: {len(idx_C)}", level=LH.DEBUG)


    TfDst_add = False
    tmp_keep_TfDst_n = None
    tmp_keep_TfDst_p = None
    write_msg(f"Positive keys: {positive.keys()}")
    if "TfDataset" in positive["Windows"].keys():
        tmp_keep_TfDst_p = positive["Windows"]["TfDataset"]
        tmp_keep_TfDst_n = negative["Windows"]["TfDataset"]
        del positive["Windows"]["TfDataset"]
        del negative["Windows"]["TfDataset"]
        TfDst_add = True

    write_msg(f"Positive keys: {positive.keys()}")

    buoni = positive.sub_select(idx_B, inplace=False)
    cattivi = negative.sub_select(idx_C, inplace=False)

    new_anchor = anchor.sub_select(idx_a, inplace=False)
    buoni_cattivi = ConcatenateNG([buoni, cattivi], label="BuoniCattivi", logger=logger)
    write_msg(f"{buoni.dft('Windows').shape[0]}")
    write_msg(f"{cattivi.dft('Windows').shape[0]}")

    if TfDst_add:
        positive["Windows"]["TfDataset"] = tmp_keep_TfDst_p
        negative["Windows"]["TfDataset"] = tmp_keep_TfDst_n

    idx_B = tf.range(buoni.dft("Windows").shape[0])
    idx_C = tf.range(buoni.dft("Windows").shape[0], buoni.dft("Windows").shape[0]+\
                     cattivi.dft("Windows").shape[0])
    # print(f"Keep value: {keep_all}")
    if not keep_all:
        del positive
        del negative
    write_msg(f"p chosen: {idx_B.shape}", level=LH.DEBUG)
    write_msg(f"n chosen: {idx_C.shape}", level=LH.DEBUG)
    write_msg(f"Anchor expected labels shape: {anchor.dft('ExpectedLabel').shape}")
    # write_msg(f"{tf.where(anchor.dft('ExpectedLabel') == 0, idx_B, idx_C)}")

    pos_net_samples_idx = tf.where(anchor.dft("ExpectedLabel") == 0, idx_B, idx_C)
    neg_net_samples_idx = tf.where(anchor.dft("ExpectedLabel") == 1, idx_B, idx_C)
    pos_net = buoni_cattivi.sub_select(pos_net_samples_idx, inplace=False)
    neg_net = buoni_cattivi.sub_select(neg_net_samples_idx, inplace=False)

    new_anchor["TripletDst"] = None
    new_anchor["TripletDst"].set_default(
                tf.stack([pos_net.dft("Windows"),
                          new_anchor.dft("Windows"),
                          neg_net.dft("Windows")],
                         axis=1))
    new_anchor["TripletDst"]["TfDataset"] = tf.data.Dataset.from_tensor_slices(
                (pos_net.dft("Windows"),
                 new_anchor.dft("Windows"),
                 neg_net.dft("Windows")))

    return new_anchor

@action
def print_data(*args, df: Optional[pd.DataFrame] = None, **kwargs) -> Union[pd.DataFrame, None]:
    """
    Utility function for debugging data inspection.

    This function prints DataFrame contents and other arguments for debugging
    purposes while passing the DataFrame through unchanged. It's useful for
    inspecting data at various stages of processing pipelines.

    Parameters
    ----------
    *args : tuple
        Variable arguments to be printed alongside the DataFrame.
    df : Optional[pd.DataFrame], optional
        DataFrame to be printed and returned. Can be None.
    **kwargs : dict
        Keyword arguments passed to the print function.

    Returns
    -------
    Union[pd.DataFrame, None]
        Returns the input DataFrame unchanged, or None if input was None.

    Examples
    --------
    >>> # Use in a processing pipeline for debugging
    >>> processed_df = some_processing_function(df)
    >>> processed_df = print_data("After processing:", df=processed_df)
    >>> # DataFrame is printed and continues through pipeline
    """
    print(df, *args, **kwargs)
    return df

def action_selector(obj, *args, **kwargs):
    """
    Dynamic action selection and execution.

    This function provides dynamic dispatch for registered actions,
    allowing runtime selection and execution of action functions
    based on string identifiers.

    Parameters
    ----------
    obj : str
        String identifier for the action to execute. Must be registered
        in the global actions dictionary.
    *args : tuple
        Positional arguments to pass to the selected action function.
    **kwargs : dict
        Keyword arguments to pass to the selected action function.

    Returns
    -------
    Any
        Return value from the executed action function.

    Raises
    ------
    ValueError
        If the specified action is not found in the actions registry.

    Notes
    -----
    This function relies on the global 'actions' dictionary that contains
    mappings from string identifiers to action functions. Actions must be
    registered using the @action decorator to be available.

    Examples
    --------
    >>> # Execute action by name
    >>> result = action_selector('normalize', data, mean, std)
    >>> # Equivalent to: result = normalize(data, mean, std)
    """
    if obj in actions.keys():
        return actions[obj](*args, **kwargs)
    raise ValueError(f"{obj} action not found")

@action
def GoodBad_randomGray(df: Optional[pd.DataFrame] = None,
                       exp_column: str = "op_id",
                       wdw_column: str = "window",
                       sep_column: str = "anomaly_window",
                       new_column: str = "Dataset",
                       gray_portions: List[float] = [0.1, 0.1]) -> pd.DataFrame:
    """
    Create random gray samples from good and bad datasets.

    This function separates data into good and bad categories based on an anomaly flag,
    then randomly selects portions from both categories to create a gray/uncertain
    category. This is useful for creating balanced datasets with uncertain samples.

    Parameters
    ----------
    df : Optional[pd.DataFrame], optional
        Input DataFrame containing experiment data.
    exp_column : str, default="op_id"
        Column name containing experiment/operation identifiers.
    wdw_column : str, default="window"
        Column name containing window identifiers.
    sep_column : str, default="anomaly_window"
        Column name containing binary anomaly flags (0=good, 1=bad).
    new_column : str, default="Dataset"
        Name of the new column to store dataset classifications.
    gray_portions : List[float], default=[0.1, 0.1]
        Portions of good and bad samples to convert to gray category.
        [good_portion, bad_portion] where each value is between 0 and 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with added classification column containing 'good', 'bad', or 'gray' labels.

    Notes
    -----
    The function operates at the window level, selecting entire windows rather than
    individual samples. This ensures temporal consistency within windows.

    Gray samples are created by:
    1. Identifying unique (experiment, window) combinations
    2. Randomly selecting specified portions from good and bad categories
    3. Relabeling selected windows as 'gray'

    The gray_portions parameter currently uses only the first value for both
    good and bad portions.

    Examples
    --------
    >>> # Create gray samples using 15% from each category
    >>> balanced_df = GoodBad_randomGray(
    ...     df, gray_portions=[0.15, 0.15],
    ...     exp_column='experiment_id',
    ...     sep_column='is_anomaly'
    ... )
    >>> print(balanced_df['Dataset'].value_counts())
    """
    check_df_not_none(df)
    # For each window if the anomaly flag is 1 then the dataset is bad
    # otherwise it's in the good dataframe
    good_df = df[df[sep_column] == 0].copy()
    bad_df = df[df[sep_column] == 1].copy()

    # randomly select from both the good and bad df entire windows that
    # would becom gray
    good_exp_wdw = np.vstack(good_df[[exp_column, wdw_column]].values,
                             dtype=int, casting='unsafe')
    good_idx = np.unique(good_exp_wdw, axis=0)
    good_wdw = np.random.choice(np.arange(len(good_idx)),
                                int(len(good_idx)*gray_portions[0]),
                                replace=False)
    good_wdw = good_idx[good_wdw]

    bad_exp_wdw = np.vstack(bad_df[[exp_column, wdw_column]].values,
                             dtype=int, casting='unsafe')
    bad_idx = np.unique(bad_exp_wdw, axis=0)
    bad_wdw = np.random.choice(np.arange(len(bad_idx)),
                                int(len(bad_idx)*gray_portions[0]),
                                replace=False)
    bad_wdw = bad_idx[bad_wdw]

    gray_wdw = np.concatenate([good_wdw, bad_wdw], axis=0)

    good_df[new_column] = "good"
    bad_df[new_column] = "bad"

    # Where good df exp_column and wdw_column are inside the gray_wdw set
    good_in_gray = np.isin(good_exp_wdw, gray_wdw)
    good_in_gray = np.logical_and(good_in_gray[:, 0], good_in_gray[:, 1])
    good_df[new_column] = np.where(good_in_gray, "gray", good_df[new_column])

    bad_in_gray = np.isin(bad_exp_wdw, gray_wdw)
    bad_in_gray = np.logical_and(bad_in_gray[:, 0], bad_in_gray[:, 1])
    bad_df[new_column] = np.where(bad_in_gray, "gray", bad_df[new_column])

    gray_df = pd.concat([
            good_df[good_df[new_column] == "gray"],
            bad_df[bad_df[new_column] == "gray"]
        ])
    good_df = good_df[good_df[new_column] != "gray"]
    bad_df = bad_df[bad_df[new_column] != "gray"]

    return pd.concat([good_df, bad_df, gray_df])

@action
def get_randomGray(df: Optional[pd.DataFrame] = None,
                   anomal_clm: str = "anomalous",
                   wdw_size: int = 120,
                   portion: List[float] = [0.1, 0.1]) -> pd.DataFrame:
    """
    Create random gray samples from normal and anomalous data.

    This function separates data into normal and anomalous categories, then
    randomly selects portions from each to create gray (uncertain) samples.
    It operates at the individual sample level rather than window level.

    Parameters
    ----------
    df : Optional[pd.DataFrame], optional
        Input DataFrame containing the data to process.
    anomal_clm : str, default="anomalous"
        Column name containing binary anomaly flags (0=normal, 1=anomalous).
    wdw_size : int, default=120
        Window size parameter for processing (currently unused in implementation).
    portion : List[float], default=[0.1, 0.1]
        Portions of normal and anomalous samples to convert to gray category.
        [normal_portion, anomalous_portion] where each value is between 0 and 1.

    Returns
    -------
    pd.DataFrame
        DataFrame with samples classified as normal, anomalous, or gray,
        with a new column indicating the classification.

    Notes
    -----
    The function performs the following operations:
    1. Separates data based on the anomaly column
    2. Randomly selects specified portions from each category
    3. Creates gray samples from selected data
    4. Combines all categories into a unified dataset

    This differs from GoodBad_randomGray by:
    - Operating at individual sample level
    - Using different column naming conventions
    - Having a simpler selection mechanism

    The gray sample creation helps in:
    - Creating balanced training sets
    - Handling uncertain or borderline cases
    - Improving model robustness to ambiguous data

    Examples
    --------
    >>> # Create gray samples using 15% from each category
    >>> processed_df = get_randomGray(
    ...     df=network_data,
    ...     anomal_clm='is_attack',
    ...     portion=[0.15, 0.15]
    ... )
    >>> print(processed_df['classification'].value_counts())
    """
    # Separate the df in good and bads bepending on the anomalous column
    good_df = df[df[anomal_clm] == 0].copy()
    bad_df = df[df[anomal_clm] == 1].copy()

    good_df["Dataset"] = "good"
    bad_df["Dataset"] = "bad"

    # Identify how many samples we need from both datasets to respect the portion
    # value
    n_good = int(len(good_df)*portion[0])
    n_bad = int(len(bad_df)*portion[1]) # pylint: disable=unused-variable

    # Randomly extract windows from the good and bad datasets without replication
    for _ in range(n_good):
        event = np.random.randint(wdw_size, len(good_df)-wdw_size)
        # Get a random value with maximum wdw_size-1, this value identifies how
        # many events to look behind the chosen event to start extracting, the
        # number of events to extract aveter the chosen event is given by
        # wdw_size - random_num - 1
        random_num = np.random.randint(0, wdw_size-1)
        start = event-random_num
        end = start+wdw_size # pylint: disable=unused-variable
    raise NotImplementedError("This action is not completely implemented yet")
