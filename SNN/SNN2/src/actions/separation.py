# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Data Separation and Balancing Module
====================================

This module provides functions for separating, balancing, and organizing neural network
training data based on various thresholds and criteria. It contains utilities for:

- Window-based data separation into good, bad, and difficult categories
- Dataset balancing and concatenation operations
- Train/validation/test split functionality
- Outlier detection and removal
- MVNO and VMAF-specific separation methods

The module is designed to work with TensorFlow tensors and DataManager objects,
providing comprehensive data preprocessing capabilities for neural network training.

Functions
---------
window_to_group : function
    Separate windows into good, bad, and gray categories based on feature thresholds.
window_to_group_mvno : function
    MVNO-specific window separation function.
windowsSeparation_mvno : function
    Complete MVNO window separation with class and label assignment.
windowsSeparation_vmaf : function
    VMAF-based window separation with quality thresholds.
windowsSeparation : function
    General window separation with multiple feature thresholds.
flag_windows : function
    Flag windows based on threshold criteria for difficult case identification.
windowDropOutliers : function
    Remove outlier samples from datasets based on target thresholds.
TrnValTstSeparation : function
    Split data into training, validation, and test sets.
balancingConcatenateNG : function
    Balance and concatenate multiple datasets according to specified ratios.
ConcatenateNG : function
    Concatenate multiple DataManager instances.
BalanceSeparationNG : function
    Combined portioning and balancing for train/validation/test splits.

Notes
-----
All functions in this module use the @action and @f_logger decorators for consistent
logging and action tracking within the SNN2 framework.
"""

import ast
import math
import functools

from typing import Any, Union, List, Tuple, Optional, Dict, Callable

import numpy as np
import tensorflow as tf
from SNN2.src.core.data.DataManager import DataManager

from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.decorators.decorators import action, f_logger


@f_logger
def window_to_group(wdw: tf.Tensor,
                    feature_thresholds: List[Tuple[float, float]],
                    **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate windows into good, bad, and gray categories based on feature thresholds.

    This function analyzes window data and categorizes each window based on multiple
    feature thresholds. Windows are classified as good (below good threshold),
    bad (above bad threshold), or gray/difficult (between thresholds).

    Parameters
    ----------
    wdw : tf.Tensor
        Input tensor containing window data with shape (n_windows, window_size, n_features).
    feature_thresholds : List[Tuple[float, float]]
        List of threshold tuples for each feature, where each tuple contains
        (good_threshold, bad_threshold).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from f_logger decorator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three numpy arrays:
        - good_indexes_intersection: Indices of windows classified as good
        - bad_indexes_intersection: Indices of windows classified as bad
        - gray_indexes_intersection: Indices of windows classified as gray/difficult

    Notes
    -----
    For the first feature (column 0), the function applies sum reduction across the
    window dimension before threshold comparison. This is designed for disaggregated
    mode processing.

    The function ensures that all windows are classified into exactly one category
    with no overlap between categories.
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    write_msg(f"Windows: {wdw}", level=LH.DEBUG)
    write_msg(f"Windows shp: {wdw.shape}", level=LH.DEBUG)
    columns = range(wdw.shape[-1])
    good_indexes = np.arange(wdw.shape[0])
    gray_indexes = []
    bad_indexes = []
    for column in columns:
        good_threshold, bad_threshold = feature_thresholds[column]
        write_msg(f"Column: {column}, thresholds: {(good_threshold, bad_threshold)}")
        # good_threshold = threshold[0]
        # bad_threshold = threshold[1]
        value = wdw[:, :, column]
        write_msg(f"Values taken into consideration: {value}", level=LH.DEBUG)
        write_msg(f"Specific value taken into consideration: {value[0]}", level=LH.DEBUG)
        # print("Running in disaggregated mode")
        if column == 0:
            value = tf.reduce_sum(value, axis=1)
        write_msg(f"Values taken into consideration: {value}", level=LH.DEBUG)
        write_msg(f"Specific value taken into consideration: {value[0]}", level=LH.DEBUG)

        write_msg(f"Example gray_indexes:"\
                  f"{tf.where((good_threshold < value) & (value < bad_threshold))[:, 0]}")
        write_msg(f"Example bad_indexes: {tf.where(value >= bad_threshold)[:, 0]}")
        gray_indexes.append(
                    tf.unique(tf.where((good_threshold < value) &\
                                       (value < bad_threshold))[:,0])[0].numpy()
                )
        bad_indexes.append(
                    tf.unique(tf.where(value > bad_threshold)[:,0])[0].numpy()
                )
        write_msg(f"last gray_indexes: {gray_indexes[-1]}")
        write_msg(f"last bad_indexes: {bad_indexes[-1]}")

    maybe_bad = np.unique(np.concatenate(bad_indexes))
    maybe_gray = np.unique(np.concatenate(gray_indexes))
    write_msg(f"Maybe bad: {maybe_bad} len: {len(maybe_bad)}")
    write_msg(f"Maybe gray: {maybe_gray} len: {len(maybe_gray)}")

    bad_indexes_intersection = np.intersect1d(maybe_bad, good_indexes)
    good_indexes = np.delete(good_indexes, np.argwhere(np.isin(good_indexes, maybe_bad)))
    gray_indexes_intersection = np.intersect1d(maybe_gray, good_indexes)
    good_indexes_intersection = np.delete(good_indexes,
                                          np.argwhere(np.isin(good_indexes, maybe_gray)))

    assert (len(good_indexes_intersection) +\
            len(bad_indexes_intersection) +\
            len(gray_indexes_intersection)) == wdw.shape[0]
    assert len(functools.reduce(np.intersect1d, [good_indexes_intersection,
                                                 bad_indexes_intersection,
                                                 gray_indexes_intersection])) == 0

    write_msg(f"good: {good_indexes_intersection} len: {len(good_indexes_intersection)}")
    write_msg(f"bad: {bad_indexes_intersection} len: {len(bad_indexes_intersection)}")
    write_msg(f"gray: {gray_indexes_intersection} len: {len(gray_indexes_intersection)}")
    return good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection

@f_logger
def window_to_group_mvno(wdw: tf.Tensor,
                    feature_thresholds: List[Tuple[float, float]],
                    **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MVNO-specific window separation into good, bad, and gray categories.

    This function is a specialized version of window_to_group designed for MVNO
    (Mobile Virtual Network Operator) data processing. It handles both 1D and
    multi-dimensional window data.

    Parameters
    ----------
    wdw : tf.Tensor
        Input tensor containing window data. Can be 1D or multi-dimensional.
    feature_thresholds : List[Tuple[float, float]]
        List of threshold tuples for each feature, where each tuple contains
        (good_threshold, bad_threshold).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from f_logger decorator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three numpy arrays:
        - good_indexes_intersection: Indices of windows classified as good
        - bad_indexes_intersection: Indices of windows classified as bad
        - gray_indexes_intersection: Indices of windows classified as gray/difficult

    Notes
    -----
    This function automatically detects if the input is 1D and adjusts the column
    processing accordingly. It maintains the same separation logic as window_to_group
    but is optimized for MVNO-specific data structures.
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    write_msg(f"Windows: {wdw}", level=LH.DEBUG)
    write_msg(f"Windows shp: {wdw.shape}", level=LH.DEBUG)
    if len(wdw.shape) == 1:
        columns = range(1)
    else:
        columns = range(wdw.shape[-1])
    write_msg(f"Columns: {columns}")
    good_indexes = np.arange(wdw.shape[0])
    gray_indexes = []
    bad_indexes = []
    for column in columns:
        good_threshold, bad_threshold = feature_thresholds[column]
        write_msg(f"Column: {column}, thresholds: {(good_threshold, bad_threshold)}")
        # good_threshold = threshold[0]
        # bad_threshold = threshold[1]
        value = wdw
        write_msg(f"Values taken into consideration: {value}", level=LH.DEBUG)
        write_msg(f"Specific value taken into consideration: {value[0]}", level=LH.DEBUG)
        # print("Running in disaggregated mode")

        write_msg(f"Example bad_indexes: {tf.where(value >= bad_threshold)[:, 0]}")
        write_msg(f"Example gray_indexes:"\
                  f"{tf.where((good_threshold < value) & (value < bad_threshold))[:, 0]}")
        gray_indexes.append(
                    tf.unique(tf.where((good_threshold < value) &\
                                       (value < bad_threshold))[:,0])[0].numpy()
                )
        bad_indexes.append(
                    tf.unique(tf.where(value > bad_threshold)[:,0])[0].numpy()
                )
        write_msg(f"last gray_indexes: {gray_indexes}")
        write_msg(f"last bad_indexes: {bad_indexes}")

    maybe_bad = np.unique(np.concatenate(bad_indexes))
    maybe_gray = np.unique(np.concatenate(gray_indexes))
    write_msg(f"Maybe bad: {maybe_bad} len: {len(maybe_bad)}")
    write_msg(f"Maybe gray: {maybe_gray} len: {len(maybe_gray)}")

    bad_indexes_intersection = np.intersect1d(maybe_bad, good_indexes)
    good_indexes = np.delete(good_indexes, np.argwhere(np.isin(good_indexes, maybe_bad)))
    gray_indexes_intersection = np.intersect1d(maybe_gray, good_indexes)
    good_indexes_intersection = np.delete(good_indexes,
                                          np.argwhere(np.isin(good_indexes, maybe_gray)))

    assert (len(good_indexes_intersection) +\
            len(bad_indexes_intersection) +\
            len(gray_indexes_intersection)) == wdw.shape[0]
    assert len(functools.reduce(np.intersect1d, [good_indexes_intersection,
                                                 bad_indexes_intersection,
                                                 gray_indexes_intersection])) == 0

    write_msg(f"good: {good_indexes_intersection} len: {len(good_indexes_intersection)}")
    write_msg(f"bad: {bad_indexes_intersection} len: {len(bad_indexes_intersection)}")
    write_msg(f"gray: {gray_indexes_intersection} len: {len(gray_indexes_intersection)}")
    return good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection


@action
@f_logger
def windowsSeparation_mvno(properties: Dict[str, Dict[str, tf.Tensor]] = {},
                           anomalous_threshold: Optional[Union[Dict[str, Tuple[float, float]],
                                                               str]] = None,
                           **kwargs) -> Tuple[Dict[str, Dict[str, tf.Tensor]], ...]:
    """
    MVNO-specific window separation into positive, negative, and difficult groups.

    This function separates MVNO (Mobile Virtual Network Operator) network data
    windows into three categories based on anomalous behavior thresholds. It's
    specifically designed for network anomaly detection scenarios.

    Parameters
    ----------
    properties : Dict[str, Dict[str, tf.Tensor]], default={}
        Dictionary containing window and target data organized by property names.
        Must contain 'Windows' and 'Targets' keys with TensorFlow tensors.
    anomalous_threshold : Optional[Union[Dict[str, Tuple[float, float]], str]], optional
        Dictionary or string representation of anomalous behavior thresholds.
        Should contain an 'anomalous' key mapping to (good_threshold, bad_threshold).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Tuple[Dict[str, Dict[str, tf.Tensor]], ...]
        A tuple containing three property dictionaries:
        - goods_properties: Properties for normal/good samples (class 0, expected label 0)
        - bads_properties: Properties for anomalous/bad samples (class 1, expected label 1)
        - grays_properties: Properties for difficult/uncertain samples (class 2, mixed labels)

    Raises
    ------
    Exception
        If window or target tensors are None in the properties dictionary.

    Notes
    -----
    Expected labels for good and bad samples are fixed (0 and 1 respectively),
    while difficult samples get labels based on whether their target values are
    positive (1) or non-positive (0).

    The function adds 'Classes' and 'ExpectedLabel' properties to each output
    group and logs detailed statistics about the separation results.

    Examples
    --------
    >>> goods, bads, grays = windowsSeparation_mvno(
    ...     properties=network_data,
    ...     anomalous_threshold={'anomalous': (0.1, 0.8)}
    ... )
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    write_msg("Window Separation MVNO")

    wdw = properties.dft("Windows")
    trg = properties["Targets"]["tf_values"]

    feature_thresholds = anomalous_threshold
    if isinstance(anomalous_threshold, str):
        if 'dict(' in anomalous_threshold:
            anomalous_threshold = anomalous_threshold.replace('dict(','{')
            anomalous_threshold = '}'.join(anomalous_threshold.rsplit(')', 1))
        feature_thresholds = ast.literal_eval(anomalous_threshold)
    write_msg(f"Threshold that needs to be applied: {feature_thresholds}")

    if wdw is None:
        raise ValueError("The window tensor passed is None")
    if trg is None:
        raise ValueError("The target tensor passed is None")

    good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection = window_to_group_mvno(
        trg, [feature_thresholds["anomalous"]], logger=logger)

    def sub_properties(idx, obj_class, exp_label = None):
        tmp_property = properties.sub_select(idx)
        tmp_property["Classes"] = None
        tmp_property["ExpectedLabel"] = None

        # write_msg(f"Indexes applied: {tmp_property.dft('OriginIndexes')}")
        write_msg(f"len(tmp_property['Targets']['tf_values']):"\
                  f"{len(tmp_property['Targets']['tf_values'])}")
        write_msg(f"tmp_property['Targets']['tf_values'] shape:"\
                  f"{tmp_property['Targets']['tf_values'].shape}")
        tmp_property["Classes"].set_default(tf.cast(
                tf.fill(dims=(len(tmp_property["Targets"]["tf_values"])),
                        value=obj_class),
                tf.int8))
        write_msg(f"Classes applied: {tmp_property['Classes']['tf_values']}")
        exp_l = tf.cast(tf.where(tmp_property.dft("Targets") > 0, 1, 0), tf.int8)
        if exp_label is not None:
            exp_l = tf.cast(tf.fill(dims=(len(tmp_property.dft("Targets"))), value=exp_label),
                            tf.int8)
        tmp_property["ExpectedLabel"].set_default(exp_l)
        write_msg(f"Expected Labels applied: {tmp_property['ExpectedLabel']['tf_values']}")
        return tmp_property


    write_msg(f"Good count: {len(good_indexes_intersection)}")
    write_msg(f"Bads count: {len(bad_indexes_intersection)}")
    write_msg(f"Difficult count: {len(gray_indexes_intersection)}")
    goods_properties = sub_properties(good_indexes_intersection, 0, exp_label=0)
    bads_properties = sub_properties(bad_indexes_intersection, 1, exp_label=1)
    grays_properties = sub_properties(gray_indexes_intersection, 2)

    write_msg(f"Good tf shape: {goods_properties.dft('Windows').shape}")
    write_msg(f"Good tf shape: {goods_properties.dft('Targets').shape}")
    write_msg(f"Good tf shape: {goods_properties.dft('Classes').shape}")
    write_msg(f"Good tf shape: {goods_properties.dft('ExpectedLabel').shape}")
    write_msg(f"Bad tf shape: {bads_properties.dft('Windows').shape}")
    write_msg(f"Bad tf shape: {bads_properties.dft('Targets').shape}")
    write_msg(f"Bad tf shape: {bads_properties.dft('Classes').shape}")
    write_msg(f"Bad tf shape: {bads_properties.dft('ExpectedLabel').shape}")
    write_msg(f"Difficult tf shape: {grays_properties.dft('Windows').shape}")
    write_msg(f"Difficult tf shape: {grays_properties.dft('Targets').shape}")
    write_msg(f"Difficult tf shape: {grays_properties.dft('Classes').shape}")
    write_msg(f"Difficult tf shape: {grays_properties.dft('ExpectedLabel').shape}")

    idi, _, count = tf.unique_with_counts(grays_properties["ExpectedLabel"]["tf_values"])
    write_msg(f"Difficult count: {(idi.numpy(), count.numpy())}", level=LH.DEBUG)

    return goods_properties, bads_properties, grays_properties

@action
@f_logger
def windowsSeparation_vmaf(properties: Dict[str, Dict[str, tf.Tensor]] = {},
                           vmaf_threshold: Optional[Union[float, str]] = None,
                           thresholds: Optional[Union[Dict[str, Tuple[float, float]], str]] = None,
                           **kwargs) -> Tuple[Dict[str, Dict[str, tf.Tensor]], ...]:
    """
    Separate windows into three groups using VMAF-based quality thresholds.

    This function separates video quality assessment windows into good, bad, and
    difficult categories based on VMAF (Video Multi-Method Assessment Fusion)
    thresholds and feature-specific criteria.

    Parameters
    ----------
    properties : Dict[str, Dict[str, tf.Tensor]], default={}
        Dictionary containing window and target data organized by property names.
    vmaf_threshold : Optional[Union[float, str]], optional
        VMAF quality threshold used to determine expected labels for samples.
        Higher VMAF scores indicate better video quality.
    thresholds : Optional[Union[Dict[str, Tuple[float, float]], str]], optional
        Dictionary or string representation of feature thresholds, where each key
        maps to a tuple of (good_threshold, bad_threshold) values.

    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Tuple[Dict[str, Dict[str, tf.Tensor]], ...]
        A tuple containing three property dictionaries:
        - goods_properties: Properties for samples classified as good (class 0)
        - bads_properties: Properties for samples classified as bad (class 1)
        - grays_properties: Properties for samples classified as difficult (class 2)

    Raises
    ------
    Exception
        If thresholds parameter is None, or if window/target tensors are None.

    Notes
    -----
    Expected labels are determined by comparing targets to the VMAF threshold:
    - Label 0: Target > vmaf_threshold (good quality)
    - Label 1: Target <= vmaf_threshold (poor quality)

    The function ensures feature threshold count matches window feature dimensions
    and logs detailed statistics about the separation process.

    Examples
    --------
    >>> goods, bads, grays = windowsSeparation_vmaf(
    ...     properties=data_props,
    ...     vmaf_threshold=30.0,
    ...     thresholds={'feature1': (10, 50), 'feature2': (5, 25)}
    ... )
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    wdw = properties.dft("Windows")
    trg = properties.dft("Targets")

    if thresholds is None:
        raise ValueError("thresholds must be defined!")

    feature_thresholds = thresholds
    if isinstance(thresholds, str):
        if 'dict(' in thresholds:
            thresholds = thresholds.replace('dict(','{')
            thresholds = '}'.join(thresholds.rsplit(')', 1))
        feature_thresholds = ast.literal_eval(thresholds)
    write_msg(f"Threshold that needs to be applied: {feature_thresholds}")

    if wdw is None:
        raise ValueError("The window tensor passed is None")
    if trg is None:
        raise ValueError("The target tensor passed is None")
    feature_thresholds = list(feature_thresholds.values())

    write_msg(f"Feature thresholds {feature_thresholds}", level=LH.DEBUG)
    assert len(feature_thresholds) == wdw.shape[2]
    write_msg(f"Feature threshold len: {len(feature_thresholds)}, wdw shape: {wdw.shape}")
    properties.log_dump("PropertiesToSubset-1")

    good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection = window_to_group(
            wdw, feature_thresholds, logger=logger)
    properties.log_dump("PropertiesToSubset-2")

    def sub_properties(idx, obj_class, exp_label = None):
        tmp_property = properties.sub_select(idx)
        tmp_property["Classes"] = None
        tmp_property["ExpectedLabel"] = None

        # write_msg(f"Indexes applied: {tmp_property.dft('OriginIndexes')}")
        tmp_property["Classes"].set_default(tf.cast(
                tf.fill(dims=(len(tmp_property["Targets"]["tf_values"])),
                        value=obj_class),
                tf.int8))
        write_msg(f"Classes applied: {tmp_property['Classes']['tf_values']}")
        exp_l = tf.cast(tf.where(tmp_property.dft("Targets") > vmaf_threshold, 0, 1), tf.int8)
        if exp_label is not None:
            exp_l = tf.cast(tf.fill(dims=(len(tmp_property.dft("Targets"))), value=exp_label),
                            tf.int8)
        tmp_property["ExpectedLabel"].set_default(exp_l)
        write_msg(f"Expected Labels applied: {tmp_property['ExpectedLabel']['tf_values']}")
        return tmp_property


    write_msg(f"Good count: {len(good_indexes_intersection)}")
    write_msg(f"Bads count: {len(bad_indexes_intersection)}")
    write_msg(f"Difficult count: {len(gray_indexes_intersection)}")
    goods_properties = sub_properties(good_indexes_intersection, 0, exp_label=0)
    bads_properties = sub_properties(bad_indexes_intersection, 1, exp_label=1)
    grays_properties = sub_properties(gray_indexes_intersection, 2)

    idi, _, count = tf.unique_with_counts(grays_properties["ExpectedLabel"]["tf_values"])
    write_msg(f"Difficult count: {(idi.numpy(), count.numpy())}", level=LH.DEBUG)

    return goods_properties, bads_properties, grays_properties


@action
def windowsSeparation(wdw: tf.Tensor, trg: tf.Tensor,
                       pdr_threshold: Optional[Union[Tuple[float, float], str]] = None,
                       bdr_threshold: Optional[Union[Tuple[float, float], str]] = None,
                       avg_ipt_threshold: Optional[Union[Tuple[float, float], str]] = None,
                       std_ipt_threshold: Optional[Union[Tuple[float, float], str]] = None,
                       skw_ipt_threshold: Optional[Union[Tuple[float, float], str]] = None,
                       kur_ipt_threshold: Optional[Union[Tuple[float, float], str]] = None,
                       difficult_pdr_threshold: Optional[Union[float, str]] = None,
                       difficult_bdr_threshold: Optional[Union[float, str]] = None,
                       difficult_avg_ipt_threshold: Optional[Union[float, str]] = None,
                       difficult_std_ipt_threshold: Optional[Union[float, str]] = None,
                       difficult_skw_ipt_threshold: Optional[Union[float, str]] = None,
                       difficult_kur_ipt_threshold: Optional[Union[float, str]] = None,
                       logger: Optional[LH] = None) -> Tuple[List[tf.Tensor],
                                                             List[tf.Tensor],
                                                             List[tf.Tensor],
                                                             List[tf.Tensor]]:
    """
    Separate windows into three groups using multiple network feature thresholds.

    This function performs comprehensive window separation based on multiple network
    performance features including packet delivery ratios, bit delivery ratios,
    and inter-packet time statistics. It categorizes windows as good, bad, or difficult
    based on feature-specific threshold ranges.

    Parameters
    ----------
    wdw : tf.Tensor
        Input tensor containing window data with shape (n_windows, window_size, n_features).
        Features are expected in order: PDR, BDR, avg_IPT, std_IPT, skw_IPT, kur_IPT.
    trg : tf.Tensor
        Target tensor containing ground truth values for each window.
    pdr_threshold : Optional[Union[Tuple[float, float], str]], optional
        Packet Delivery Ratio thresholds as (good_threshold, bad_threshold).
    bdr_threshold : Optional[Union[Tuple[float, float], str]], optional
        Bit Delivery Ratio thresholds as (good_threshold, bad_threshold).
    avg_ipt_threshold : Optional[Union[Tuple[float, float], str]], optional
        Average Inter-Packet Time thresholds as (good_threshold, bad_threshold).
    std_ipt_threshold : Optional[Union[Tuple[float, float], str]], optional
        Standard deviation of Inter-Packet Time thresholds as (good_threshold, bad_threshold).
    skw_ipt_threshold : Optional[Union[Tuple[float, float], str]], optional
        Skewness of Inter-Packet Time thresholds as (good_threshold, bad_threshold).
    kur_ipt_threshold : Optional[Union[Tuple[float, float], str]], optional
        Kurtosis of Inter-Packet Time thresholds as (good_threshold, bad_threshold).
    difficult_pdr_threshold : Optional[Union[float, str]], optional
        PDR threshold for labeling difficult samples.
    difficult_bdr_threshold : Optional[Union[float, str]], optional
        BDR threshold for labeling difficult samples.
    difficult_avg_ipt_threshold : Optional[Union[float, str]], optional
        Average IPT threshold for labeling difficult samples.
    difficult_std_ipt_threshold : Optional[Union[float, str]], optional
        Standard deviation IPT threshold for labeling difficult samples.
    difficult_skw_ipt_threshold : Optional[Union[float, str]], optional
        Skewness IPT threshold for labeling difficult samples.
    difficult_kur_ipt_threshold : Optional[Union[float, str]], optional
        Kurtosis IPT threshold for labeling difficult samples.
    logger : Optional[LH], optional
        Logger instance for logging separation statistics and debug information.

    Returns
    -------
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]
        A tuple containing four lists:
        - windows: [good_windows, gray_windows, bad_windows]
        - targets: [good_targets, gray_targets, bad_targets]
        - classes: [good_classes, gray_classes, bad_classes] (0, 2, 1 respectively)
        - expectations: [good_expectations, gray_expectations, bad_expectations]

    Raises
    ------
    Exception
        If any required threshold is None, or if window/target tensors are None.

    Notes
    -----
    The function uses feature_eval to parse string thresholds and ensures all
    required thresholds are provided. Difficult samples use flag_windows to
    determine their expected labels based on the difficult_* thresholds.

    String thresholds are parsed using ast.literal_eval for flexibility.

    Examples
    --------
    >>> windows, targets, classes, expectations = windowsSeparation(
    ...     wdw, trg,
    ...     pdr_threshold=(0.8, 0.95),
    ...     bdr_threshold=(0.7, 0.9),
    ...     # ... other thresholds
    ...     logger=my_logger
    ... )
    """
    def feature_eval(features: List[Any]) -> List[Any]:
        if None in features:
            raise ValueError("All thresholds must be provided: pdr_threshold,"\
                             "bdr_threshold, avg_ipt_threshold, std_ipt_threshold,"\
                                "skw_ipt_threshold, kur_ipt_threshold")
        for i, threshold in enumerate(features):
            if isinstance(threshold, str):
                features[i] = ast.literal_eval(threshold)
        return features

    log = logger
    if log is None:
        raise ValueError("Logger instance must be provided")

    if wdw is None:
        raise ValueError("The window tensor passed is None")
    if trg is None:
        raise ValueError("The target tensor passed is None")
    feature_thresholds = [pdr_threshold,
                          bdr_threshold,
                          avg_ipt_threshold,
                          std_ipt_threshold,
                          skw_ipt_threshold,
                          kur_ipt_threshold]
    difficult_feature_thresholds = [difficult_pdr_threshold,
                                    difficult_bdr_threshold,
                                    difficult_avg_ipt_threshold,
                                    difficult_std_ipt_threshold,
                                    difficult_skw_ipt_threshold,
                                    difficult_kur_ipt_threshold]
    feature_thresholds = feature_eval(feature_thresholds)
    difficult_feature_thresholds = feature_eval(difficult_feature_thresholds)

    log("WindowSeparation", f"Feature thresholds {feature_thresholds}", level=LH.DEBUG)
    log("WindowSeparation", f"Difficult feature thresholds"\
                            f"{difficult_feature_thresholds}", level=LH.DEBUG)

    columns = range(wdw.shape[2])
    good_indexes = np.arange(wdw.shape[0])
    gray_indexes = []
    bad_indexes = []
    for column in columns:
        threshold = feature_thresholds[column]
        good_threshold = threshold[0]
        bad_threshold = threshold[1]
        gray_indexes.append(
                    tf.unique(tf.where((good_threshold < wdw[:,:,column]) &\
                                       (wdw[:,:,column] < bad_threshold))[:,0])[0].numpy()
                )
        bad_indexes.append(
                    tf.unique(tf.where(wdw[:,:,column] > bad_threshold)[:,0])[0].numpy()
                )

    maybe_bad = np.unique(np.concatenate(bad_indexes))
    maybe_gray = np.unique(np.concatenate(gray_indexes))

    bad_indexes_intersection = np.intersect1d(maybe_bad, good_indexes)
    good_indexes = np.delete(good_indexes, np.argwhere(np.isin(good_indexes, maybe_bad)))
    gray_indexes_intersection = np.intersect1d(maybe_gray, good_indexes)
    good_indexes_intersection = np.delete(good_indexes,
                                          np.argwhere(np.isin(good_indexes, maybe_gray)))

    assert (len(good_indexes_intersection) +\
            len(bad_indexes_intersection) +\
            len(gray_indexes_intersection)) == wdw.shape[0]
    assert len(functools.reduce(np.intersect1d,
                                [good_indexes_intersection,
                                 bad_indexes_intersection,
                                 gray_indexes_intersection])) == 0
    goods_windows = tf.gather(wdw, good_indexes_intersection)
    goods_targets = tf.gather(trg, good_indexes_intersection)
    goods_classes = tf.cast(tf.fill(dims=(len(goods_targets)), value=0), tf.int8)
    goods_exp_l = tf.cast(tf.fill(dims=(len(goods_targets)), value=0), tf.int8)
    bads_windows = tf.gather(wdw, bad_indexes_intersection)
    bads_targets = tf.gather(trg, bad_indexes_intersection)
    bads_classes = tf.cast(tf.fill(dims=(len(bads_targets)), value=1), tf.int8)
    bads_exp_l = tf.cast(tf.fill(dims=(len(bads_targets)), value=1), tf.int8)
    grays_windows = tf.gather(wdw, gray_indexes_intersection)
    grays_targets = tf.gather(trg, gray_indexes_intersection)
    grays_classes = tf.cast(tf.fill(dims=(len(grays_targets)), value=2), tf.int8)
    grays_exp_l = tf.cast(flag_windows(grays_windows,
                                       pdr_threshold = difficult_feature_thresholds[0],
                                       bdr_threshold = difficult_feature_thresholds[1],
                                       avg_ipt_threshold = difficult_feature_thresholds[2],
                                       std_ipt_threshold = difficult_feature_thresholds[3],
                                       skw_ipt_threshold = difficult_feature_thresholds[4],
                                       kur_ipt_threshold = difficult_feature_thresholds[5],
                                       log=log), tf.int8)

    log("WindowSeparation", f"Goods targets: {goods_targets}", level=LH.DEBUG)
    log("WindowSeparation", f"Goods classes: {goods_classes}", level=LH.DEBUG)
    log("WindowSeparation", f"Goods expected_label: {goods_exp_l}", level=LH.DEBUG)
    log("WindowSeparation", f"Bads targets: {bads_targets}", level=LH.DEBUG)
    log("WindowSeparation", f"Bads classes: {bads_classes}", level=LH.DEBUG)
    log("WindowSeparation", f"Bads expected_label: {bads_exp_l}", level=LH.DEBUG)
    log("WindowSeparation", f"Difficult targets: {grays_targets}", level=LH.DEBUG)
    log("WindowSeparation", f"Difficult classes: {grays_classes}", level=LH.DEBUG)
    log("WindowSeparation", f"Difficult expected_label: {grays_exp_l}", level=LH.DEBUG)
    idi, _, count = tf.unique_with_counts(grays_exp_l)
    log("WindowSeparation", f"Difficult count: {(idi.numpy(), count.numpy())}", level=LH.DEBUG)

    windows = [goods_windows, grays_windows, bads_windows]
    targets = [goods_targets, grays_targets, bads_targets]
    classes = [goods_classes, grays_classes, bads_classes]
    expectations = [goods_exp_l, grays_exp_l, bads_exp_l]

    return windows, targets, classes, expectations

@action
def flag_windows(wdw: tf.Tensor,
                 pdr_threshold: Optional[Union[float, str]] = None,
                 bdr_threshold: Optional[Union[float, str]] = None,
                 avg_ipt_threshold: Optional[Union[float, str]] = None,
                 std_ipt_threshold: Optional[Union[float, str]] = None,
                 skw_ipt_threshold: Optional[Union[float, str]] = None,
                 kur_ipt_threshold: Optional[Union[float, str]] = None,
                 log: Union[LH, Callable] = None) -> tf.Tensor:
    """
    Flag windows as positive (1) or negative (0) based on multiple feature thresholds.

    This function evaluates windows against multiple feature thresholds and returns
    binary flags indicating whether each window exceeds any of the specified thresholds.
    It's primarily used for labeling difficult/gray windows in the separation process.

    Parameters
    ----------
    wdw : tf.Tensor
        Input tensor containing window data with shape (n_windows, window_size, n_features).
    pdr_threshold : Optional[Union[float, str]], optional
        Threshold for Packet Delivery Ratio feature.
    bdr_threshold : Optional[Union[float, str]], optional
        Threshold for Bit Delivery Ratio feature.
    avg_ipt_threshold : Optional[Union[float, str]], optional
        Threshold for average Inter-Packet Time feature.
    std_ipt_threshold : Optional[Union[float, str]], optional
        Threshold for standard deviation of Inter-Packet Time feature.
    skw_ipt_threshold : Optional[Union[float, str]], optional
        Threshold for skewness of Inter-Packet Time feature.
    kur_ipt_threshold : Optional[Union[float, str]], optional
        Threshold for kurtosis of Inter-Packet Time feature.
    log : Union[LH, Callable], optional
        Logger instance or callable for logging operations.

    Returns
    -------
    tf.Tensor
        Binary tensor of shape (n_windows,) where 1 indicates the window exceeds
        at least one threshold and 0 indicates it doesn't exceed any threshold.

    Raises
    ------
    Exception
        If any of the required thresholds is None.

    Notes
    -----
    String thresholds are parsed using ast.literal_eval. The function checks each
    feature column against its corresponding threshold and flags windows that
    exceed any threshold value.
    """
    log("flag_windows", "Required generation of true windows", level=LH.DEBUG)
    feature_thresholds = [pdr_threshold,
                          bdr_threshold,
                          avg_ipt_threshold,
                          std_ipt_threshold,
                          skw_ipt_threshold,
                          kur_ipt_threshold]
    if None in feature_thresholds:
        raise ValueError("All thresholds must be provided")

    for i, threshold in enumerate(feature_thresholds):
        if isinstance(threshold, str):
            feature_thresholds[i] = ast.literal_eval(threshold)

    log("flag_windows", f"Feature thresholds {feature_thresholds}", level=LH.DEBUG)

    good_indexes = np.arange(wdw.shape[0])
    bad_indexes = []
    log("flag_windows", f"Indexes length before cycle, good:"\
              f"{len(good_indexes)} bads: {len(bad_indexes)}", level=LH.DEBUG)
    for column in range(wdw.shape[2]):
        bad_threshold = feature_thresholds[column]
        bad_indexes.append(
                    tf.unique(tf.where(wdw[:,:,column] > bad_threshold)[:,0])[0].numpy()
                )

    bad_idx = np.unique(np.concatenate(bad_indexes))
    log("flag_windows", f"Indexes length after cycle, good:"\
              f"{len(good_indexes)} bads: {len(bad_idx)}", level=LH.DEBUG)

    return tf.cast(tf.where(np.isin(good_indexes, bad_idx), 1, 0), tf.int32)


@action
@f_logger
def windowDropOutliers(data: DataManager,
                       threshold: Union[float, str] = 101.0,
                       **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Remove outlier samples from a dataset based on target value thresholds.

    This function filters out data samples where the target values exceed a specified
    threshold, effectively removing outliers from the dataset.

    Parameters
    ----------
    data : DataManager
        The DataManager instance containing the dataset to be filtered.
    threshold : Union[float, str], default=101.0
        The threshold value above which samples are considered outliers and removed.
        If string, it will be parsed using ast.literal_eval.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        The filtered DataManager instance with outliers removed.

    Notes
    -----
    The function modifies the input DataManager in-place by selecting only the indices
    of samples that meet the threshold criteria. The original data structure is preserved
    but with reduced sample count.

    Examples
    --------
    >>> filtered_data = windowDropOutliers(data, threshold=100.0)
    >>> # Removes all samples where target values >= 100.0
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    if isinstance(threshold, str):
        threshold = ast.literal_eval(threshold)

    trg = data.dft("Targets")
    write_msg(f"Length before the outlier drop: {len(trg)}")
    idx = tf.reshape(tf.where(trg >= threshold), [-1])
    write_msg(f"Length after the outlier drop: {len(idx)}")

    data.sub_select(idx, inplace=True)
    return data

@action
@f_logger
def TrnValTstSeparation(data: DataManager,
                        training_portion: float = 0.7,
                        validation_portion: float = 0.1,
                        test_portion: float = 0.2,
                        to_dataset: bool = True,
                        **kwargs) -> Tuple[Dict[str, Dict[str, Any]], ...]:
    """
    Split dataset into training, validation, and test sets with specified proportions.

    This function randomly splits a dataset into three subsets for machine learning
    model training, validation, and testing. It ensures no overlap between the sets
    and provides statistics about positive and negative samples.

    Parameters
    ----------
    data : DataManager
        The input DataManager instance containing the complete dataset.
    training_portion : float, default=0.7
        Fraction of data to use for training (0.0 to 1.0).
    validation_portion : float, default=0.1
        Fraction of data to use for validation (0.0 to 1.0).
    test_portion : float, default=0.2
        Fraction of data to use for testing (0.0 to 1.0).
    to_dataset : bool, default=True
        Whether to convert the resulting data to TensorFlow datasets.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], ...]
        A tuple containing three DataManager instances:
        - training: Training dataset
        - validation: Validation dataset
        - test: Test dataset

    Raises
    ------
    AssertionError
        If the sum of portions exceeds 1.0 or if there are overlapping indices.

    Notes
    -----
    The function preserves TensorFlow dataset objects if they exist in the original
    data and randomly shuffles indices to ensure unbiased splits. It also logs
    statistics about positive and negative target value ranges.

    Examples
    --------
    >>> train, val, test = TrnValTstSeparation(data, 0.8, 0.1, 0.1)
    >>> # Splits data into 80% training, 10% validation, 10% test
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    windows = data.dft("Windows")
    targets = data.dft("Targets")
    expectl = data.dft("ExpectedLabel")

    write_msg(f"Positive targets minimum:"\
              f"{tf.math.reduce_min(tf.gather(targets, tf.where(expectl == 0)[:, 0]))}")
    write_msg(f"Negative targets maximum:"\
              f"{tf.math.reduce_max(tf.gather(targets, tf.where(expectl == 1)[:, 0]))}")

    assert training_portion + validation_portion + test_portion <= 1.0
    dimension_wdw = windows.shape[0]

    training_val = math.floor(dimension_wdw*training_portion)
    validation_val = math.floor(dimension_wdw*validation_portion)
    test_val = math.floor(dimension_wdw*test_portion)
    assert training_val + validation_val + test_val <= dimension_wdw
    write_msg(f"Training dimension: {training_val}")
    write_msg(f"Validation dimension: {validation_val}")
    write_msg(f"Test dimension: {test_val}")

    idx = tf.range(windows.shape[0])
    idx = tf.random.shuffle(idx)

    trn_idx = idx[:training_val]
    val_idx = idx[training_val:training_val+validation_val]
    tst_idx = idx[training_val+validation_val:training_val+validation_val+test_val]

    write_msg(f"Train idx: {trn_idx}")
    write_msg(f"Validation idx: {val_idx}")
    write_msg(f"Test idx: {tst_idx}")
    trn_val_int = np.intersect1d(trn_idx.numpy(), val_idx.numpy())
    trn_tst_int = np.intersect1d(trn_idx.numpy(), tst_idx.numpy())
    val_tst_int = np.intersect1d(val_idx.numpy(), tst_idx.numpy())
    intersections = [trn_val_int, trn_tst_int, val_tst_int]
    write_msg(f"Train - Val intersection: {trn_val_int}")
    write_msg(f"Train - Test intersection: {trn_tst_int}")
    write_msg(f"Val - Test intersection: {val_tst_int}")
    assert all(len(x) == 0 for x in intersections)

    TfDst_add = False
    tmp_keep_TfDst_data = None
    write_msg(f"Positive keys: {data.keys()}")
    if "TfDataset" in data["Windows"].keys():
        tmp_keep_TfDst_data = data["Windows"]["TfDataset"]
        del data["Windows"]["TfDataset"]
        TfDst_add = True

    training = data.sub_select(trn_idx)
    validation = data.sub_select(val_idx)
    test = data.sub_select(tst_idx)

    if TfDst_add:
        data["Windows"]["TfDataset"] = tmp_keep_TfDst_data

    training.log_dump("Training")
    validation.log_dump("Validation")
    test.log_dump("test")

    if to_dataset:
        training.transform_toDataset(inplace=True)
        validation.transform_toDataset(inplace=True)
        test.transform_toDataset(inplace=True)

    return training, validation, test

@action
@f_logger
def balancingConcatenateNG(datasets: List[DataManager],
                           balancing: Tuple[float, ...],
                           **kwargs) -> Tuple[Dict[str, Dict[str, Any]], ...]:
    """
    Balance and concatenate multiple datasets according to specified ratios.

    This function takes multiple datasets and balances them according to the provided
    balancing ratios, then concatenates them into merged datasets. It handles cases
    where some datasets might be empty and ensures proper proportional sampling.

    Parameters
    ----------
    datasets : List[DataManager]
        List of DataManager objects containing the datasets to be balanced and concatenated.
    balancing : Tuple[float, ...]
        Tuple of float values specifying the desired proportion for each dataset.
        Must have the same length as datasets.
    **kwargs : dict
        Additional keyword arguments containing:
        - logger : Logger object for logging operations
        - write_msg : Function for writing messages/logs

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], ...]
        A tuple containing two dictionaries:
        - merged_data: Concatenated data from the balanced sampling
        - merged_not_used_data: Concatenated data that wasn't used in balancing

    Raises
    ------
    AssertionError
        If the length of balancing tuple doesn't match the number of datasets.

    Notes
    -----
    - The function handles empty datasets by removing them from consideration
    - Uses the minimum dataset size relative to its balancing ratio to determine total elements
    - Randomly shuffles indices before sampling to ensure randomness
    - At least 1 element is required from each dataset even if balancing suggests 0
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]
    assert len(balancing) == len(datasets)

    dimensions = np.array([len(data.dft('Windows')) for data in datasets])
    write_msg(f"Dimensions {dimensions}")
    balancing = np.array(balancing).flatten()
    write_msg(f"balancing requirements: {balancing}")

    if any(dimensions == 0):
        write_msg("One of the dst dimensions is 0")
        zero_idx_dim = np.where(dimensions == 0)[0]
        zero_idx_bal = np.where(balancing == 0.)[0]
        np.testing.assert_equal(zero_idx_dim, zero_idx_bal)
        dimensions = np.delete(dimensions, zero_idx_dim)
        balancing = np.delete(balancing, zero_idx_dim)
        write_msg(f"Dimensions {dimensions}")
        write_msg(f"balancing requirements: {balancing}")

    write_msg(f"Dimensions {dimensions}")
    write_msg(f"balancing requirements: {balancing}")
    min_idx = np.argmin(dimensions)
    max_idx = np.argmax(dimensions)
    total_elements = math.floor(dimensions[min_idx]/balancing[min_idx])
    total_elements = min(total_elements, dimensions[max_idx])
    write_msg(f"Total elements required: {total_elements}")

    idx, not_used_idx = [], []
    for i, (balance, dim) in enumerate(zip(balancing, dimensions)):
        write_msg(f"Current balancing cycle: {str((i, balance, dim))}")
        elements_required = math.floor(total_elements * balance)
        elements_required = max(elements_required, 1)

        indices = tf.range(start=0, limit=dim, dtype=tf.int32)
        shuffled_idx = tf.random.shuffle(indices)
        idx.append(shuffled_idx[:elements_required])
        write_msg(f"Elements required: {len(idx[-1])} vs {len(shuffled_idx)}")
        if elements_required < len(shuffled_idx):
            not_used_idx.append(shuffled_idx[elements_required:])
        else:
            not_used_idx.append(None)

    # write_msg(f"Lend index used: {[len(id) for id in idx]}")
    # write_msg(f"Lend index not used: {[len(id) for id in not_used_idx]}")
    used_data, not_used_data = [], []
    for idi, id_not_used, dst in zip(idx, not_used_idx, datasets):
        used_data.append(dst.sub_select(idi))
        if id_not_used is not None:
            not_used_data.append(dst.sub_select(id_not_used))

    merged_data = ConcatenateNG(used_data, label="UsedData", logger=logger)
    merged_not_used_data = ConcatenateNG(not_used_data, label="NotUsedData", logger=logger)

    write_msg(f"Total elements used returned: {len(merged_data['Targets']['tf_values'])}")
    write_msg(f"Total elements not used returned:"\
              f"{len(merged_not_used_data['Targets']['tf_values'])}")
    return merged_data, merged_not_used_data

@action
@f_logger
def ConcatenateNG(datasets: List[DataManager],
                  label: str = "MergedData",
                  **kwargs) -> DataManager:
    """
    Concatenate multiple datasets into a single DataManager instance.

    This function merges a list of DataManager objects into one consolidated dataset
    and logs the merged data with a specified label.

    Parameters
    ----------
    datasets : List[DataManager]
        A list of DataManager objects to be concatenated/merged together.
    label : str, optional
        Label to use when logging the merged dataset. Default is "MergedData".
    **kwargs
        Arbitrary keyword arguments. Must contain:
        - logger: Logger instance for logging operations
        - write_msg: Message writing function for logging

    Returns
    -------
    DataManager
        A new DataManager instance containing the merged data from all input datasets.

    Notes
    -----
    The function uses DataManager.merge() to perform the actual concatenation and
    automatically logs the result using the provided label.
    """
    _, _ = kwargs["logger"], kwargs["write_msg"]

    merged_data = DataManager.merge(datasets)
    merged_data.log_dump(label)

    return merged_data

@action
@f_logger
def BalanceSeparationNG(data: List[DataManager],
                        portion: Tuple[Tuple[float, float, float], ...] = ((0.7, 0.2, 0.1),
                                                                           (0.7, 0.2, 0.1),
                                                                           (0.0, 0.0, 0.0)),
                        balancing: Tuple[Tuple[float, float, float], ...] = ((0.98, 0.02, 0.0),
                                                                             (0.5, 0.5, 0.0),
                                                                             (0.5, 0.5, 0.0)),
                        test_get_everything: bool = False,
                        **kwargs) -> Tuple[Dict[str, Dict[str, Any]], ...]:
    """
    Apply portioning and balancing to multiple datasets for train/validation/test splits.

    This function performs comprehensive dataset preparation by first splitting each
    input dataset into training, validation, and test portions, then applying
    balancing ratios to create final mixed datasets with desired class distributions.

    Parameters
    ----------
    data : List[DataManager]
        List of DataManager instances representing different dataset categories
        (e.g., good, bad, difficult samples).
    portion : Tuple[Tuple[float, float, float], ...],
        default=((0.7, 0.2, 0.1), (0.7, 0.2, 0.1), (0.0, 0.0, 0.0))
        Tuple of portion tuples specifying how to split each dataset.
        Each inner tuple contains (train_portion, validation_portion, test_portion)
        and must sum to <= 1.0. Length must match number of input datasets.
    balancing : Tuple[Tuple[float, float, float], ...],
        default=((0.98, 0.02, 0.0), (0.5, 0.5, 0.0), (0.5, 0.5, 0.0))
        Tuple of balancing ratios for train, validation, and test sets.
        Each inner tuple specifies the desired proportion of each dataset category.
        Contains exactly 3 tuples for (train_balance, val_balance, test_balance).
    test_get_everything : bool, default=False
        If True, test set includes all unused data from training and validation.
        If False, test set uses balanced sampling like train and validation.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Tuple[Dict[str, Dict[str, Any]], ...]
        A tuple containing three DataManager instances:
        - training: Balanced training dataset
        - validation: Balanced validation dataset
        - test: Balanced test dataset (or everything if test_get_everything=True)

    Raises
    ------
    AssertionError
        If input validation fails for data/portion/balancing length matching.

    Notes
    -----
    The function first applies TrnValTstSeparation to each input dataset according
    to the specified portions, then uses balancingConcatenateNG to create balanced
    final datasets according to the balancing ratios.

    When test_get_everything=True, unused samples from training and validation
    balancing are added to the test set, providing comprehensive test coverage.

    Examples
    --------
    >>> train, val, test = BalanceSeparationNG(
    ...     data=[good_data, bad_data, gray_data],
    ...     portion=((0.8, 0.1, 0.1), (0.7, 0.15, 0.15), (0.6, 0.2, 0.2)),
    ...     balancing=((0.7, 0.3, 0.0), (0.5, 0.5, 0.0), (0.4, 0.4, 0.2)),
    ...     test_get_everything=True
    ... )
    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    assert len(data) == len(portion)
    assert len(balancing) == 3
    for value in balancing:
        assert len(data) == len(value)

    write_msg(f"Portion request: {portion}")
    write_msg(f"Balancing request: {balancing}")

    datasets = {'trn': [], 'val': [], 'tst': []}
    for dataset, portions in zip(data, portion):
        trn, val, tst = TrnValTstSeparation(dataset,
                                    training_portion=portions[0],
                                    validation_portion=portions[1],
                                    test_portion=portions[2],
                                    logger=logger,
                                    to_dataset=False)
        datasets['trn'].append(trn)
        datasets['val'].append(val)
        datasets['tst'].append(tst)

    for key, item in datasets.items():
        write_msg(f"---- {key} ----")
        write_msg(str([len(x['Windows']['tf_values']) for x in item]))

    training, training_not_used = balancingConcatenateNG(
                datasets['trn'],
                balancing[0],
                logger=logger
            )
    validation, validation_not_used = balancingConcatenateNG(
                datasets['val'],
                balancing[1],
                logger=logger
            )
    if test_get_everything:
        write_msg("Test get everything")
        datasets['tst'].extend([training_not_used, validation_not_used])
        test = ConcatenateNG(
                    datasets['tst'],
                    label="TestDST",
                    logger=logger
                )
    else:
        write_msg("Test get balanced")
        test, _ = balancingConcatenateNG(
                    datasets['tst'],
                    balancing[2],
                    logger=logger
                )

    return training, validation, test
