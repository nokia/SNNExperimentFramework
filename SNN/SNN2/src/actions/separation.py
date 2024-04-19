# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import ast
from copy import deepcopy
import math
import functools
from unittest import expectedFailure
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.unconnected_gradients import enum
from SNN2.src.core.data.DataManager import DataManager

from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.decorators.decorators import action, f_logger
from SNN2.src.util.helper import dst2tensor

from typing import Any, Union, List, Tuple, Optional, Dict, Callable

@f_logger
def window_to_group(wdw: tf.Tensor,
                    feature_thresholds: List[Tuple[float, float]],
                    **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    columns = range(wdw.shape[2])
    good_indexes = np.arange(wdw.shape[0])
    gray_indexes = []
    bad_indexes = []
    write_msg(f"Windows: {wdw}", level=LH.DEBUG)
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

        write_msg(f"Example gray_indexes: {tf.where((good_threshold < value) & (value < bad_threshold))[:, 0]}")
        write_msg(f"Example bad_indexes: {tf.where(value >= bad_threshold)[:, 0]}")
        gray_indexes.append(
                    tf.unique(tf.where((good_threshold < value) & (value < bad_threshold))[:,0])[0].numpy()
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
    good_indexes_intersection = np.delete(good_indexes, np.argwhere(np.isin(good_indexes, maybe_gray)))

    assert (len(good_indexes_intersection) + len(bad_indexes_intersection) + len(gray_indexes_intersection)) == wdw.shape[0]
    assert len(functools.reduce(np.intersect1d, [good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection])) == 0

    write_msg(f"good: {good_indexes_intersection} len: {len(good_indexes_intersection)}")
    write_msg(f"bad: {bad_indexes_intersection} len: {len(bad_indexes_intersection)}")
    write_msg(f"gray: {gray_indexes_intersection} len: {len(gray_indexes_intersection)}")
    return good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection


@action
@f_logger
def windowsSeparation_vmaf(properties: Dict[str, Dict[str, tf.Tensor]] = {},
                           vmaf_threshold: Optional[Union[float, str]] = None,
                           thresholds: Optional[Union[Dict[str, Tuple[float, float]], str]] = None,
                           **kwargs) -> Tuple[Dict[str, Dict[str, tf.Tensor]], ...]:
    """windowsSeparation.
    Function used to separate the windows in the 3 groups, certainly positive,
    certainly negative and difficult.
    The result of the separation will be given back as a tuple contaning 4 elements
    - List of windows tensors, [good.windows, difficult.windows, bad.windows]
    - List of target tensors, [good.targets, difficult.targets, bad.targets]
    - List of classes tensors, [good.classes, difficult.classes, bad.classes]
      o For the good samples the class is 0, 2 for difficult cases and 1 for bad ones
    - List of exp-labels tensors, [good.exp-labels, difficult.exp-labels, bad.exp-labels]
      o Label is 0 if the sample is expected to be positive or 1 otherwise
      The exp-label is decided using the VMAF threshold

    Parameters
    ----------

    Returns
    -------
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]

    """
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    wdw = properties.dft("Windows")
    trg = properties.dft("Targets")

    if thresholds is None:
        raise Exception(f"thresholds must be defined!")

    feature_thresholds = thresholds
    if isinstance(thresholds, str):
        if 'dict(' in thresholds:
            thresholds = thresholds.replace('dict(','{')
            thresholds = '}'.join(thresholds.rsplit(')', 1))
        feature_thresholds = ast.literal_eval(thresholds)
    write_msg(f"Threshold that needs to be applied: {feature_thresholds}")

    if wdw is None:
        raise Exception("The window tensor passed is None")
    if trg is None:
        raise Exception("The target tensor passed is None")
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

        write_msg(f"Indexes applied: {tmp_property.dft('OriginIndexes')}")
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

    id, _, count = tf.unique_with_counts(grays_properties["ExpectedLabel"]["tf_values"])
    write_msg(f"Difficult count: {(id.numpy(), count.numpy())}", level=LH.DEBUG)

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
                       logger: Optional[LH] = None,
                       **kwargs) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
    """windowsSeparation.
    Function used to separate the windows in the 3 groups, certainly positive,
    certainly negative and difficult.
    The result of the separation will be given back as a tuple contaning 4 elements
    - List of windows tensors, [good.windows, difficult.windows, bad.windows]
    - List of target tensors, [good.targets, difficult.targets, bad.targets]
    - List of classes tensors, [good.classes, difficult.classes, bad.classes]
      o For the good samples the class is 0, 2 for difficult cases and 1 for bad ones
    - List of exp-labels tensors, [good.exp-labels, difficult.exp-labels, bad.exp-labels]
      o Label is 0 if the sample is expected to be positive or 1 otherwise

    Parameters
    ----------
    wdw : tf.Tensor
        wdw
    trg : tf.Tensor
        trg
    pdr_threshold : Optional[Union[Tuple[float, float], str]]
        pdr_threshold
    bdr_threshold : Optional[Union[Tuple[float, float], str]]
        bdr_threshold
    avg_ipt_threshold : Optional[Union[Tuple[float, float], str]]
        avg_ipt_threshold
    std_ipt_threshold : Optional[Union[Tuple[float, float], str]]
        std_ipt_threshold
    skw_ipt_threshold : Optional[Union[Tuple[float, float], str]]
        skw_ipt_threshold
    kur_ipt_threshold : Optional[Union[Tuple[float, float], str]]
        kur_ipt_threshold
    logger : Optional[LH]
        logger
    kwargs :
        kwargs

    Returns
    -------
    Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]

    """
    def feature_eval(features: List[Any]) -> List[Any]:
        if None in features:
            raise Exception("All the threshold must be given")
        for i, threshold in enumerate(features):
            if isinstance(threshold, str):
                features[i] = ast.literal_eval(threshold)
        return features

    log = logger
    if log is None:
        raise Exception

    if wdw is None:
        raise Exception("The window tensor passed is None")
    if trg is None:
        raise Exception("The target tensor passed is None")
    feature_thresholds = [pdr_threshold, bdr_threshold, avg_ipt_threshold, std_ipt_threshold, skw_ipt_threshold, kur_ipt_threshold]
    difficult_feature_thresholds = [difficult_pdr_threshold, difficult_bdr_threshold, difficult_avg_ipt_threshold, difficult_std_ipt_threshold, difficult_skw_ipt_threshold, difficult_kur_ipt_threshold]
    feature_thresholds = feature_eval(feature_thresholds)
    difficult_feature_thresholds = feature_eval(difficult_feature_thresholds)

    log("WindowSeparation", f"Feature thresholds {feature_thresholds}", level=LH.DEBUG)
    log("WindowSeparation", f"Difficult feature thresholds {difficult_feature_thresholds}", level=LH.DEBUG)

    columns = range(wdw.shape[2])
    good_indexes = np.arange(wdw.shape[0])
    gray_indexes = []
    bad_indexes = []
    for column in columns:
        threshold = feature_thresholds[column]
        good_threshold = threshold[0]
        bad_threshold = threshold[1]
        gray_indexes.append(
                    tf.unique(tf.where((good_threshold < wdw[:,:,column]) & (wdw[:,:,column] < bad_threshold))[:,0])[0].numpy()
                )
        bad_indexes.append(
                    tf.unique(tf.where(wdw[:,:,column] > bad_threshold)[:,0])[0].numpy()
                )

    maybe_bad = np.unique(np.concatenate(bad_indexes))
    maybe_gray = np.unique(np.concatenate(gray_indexes))

    bad_indexes_intersection = np.intersect1d(maybe_bad, good_indexes)
    good_indexes = np.delete(good_indexes, np.argwhere(np.isin(good_indexes, maybe_bad)))
    gray_indexes_intersection = np.intersect1d(maybe_gray, good_indexes)
    good_indexes_intersection = np.delete(good_indexes, np.argwhere(np.isin(good_indexes, maybe_gray)))

    assert (len(good_indexes_intersection) + len(bad_indexes_intersection) + len(gray_indexes_intersection)) == wdw.shape[0]
    assert len(functools.reduce(np.intersect1d, [good_indexes_intersection, bad_indexes_intersection, gray_indexes_intersection])) == 0
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
    id, _, count = tf.unique_with_counts(grays_exp_l)
    log("WindowSeparation", f"Difficult count: {(id.numpy(), count.numpy())}", level=LH.DEBUG)

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
    log("flag_windows", "Required generation of true windows", level=LH.DEBUG)
    feature_thresholds = [pdr_threshold, bdr_threshold, avg_ipt_threshold, std_ipt_threshold, skw_ipt_threshold, kur_ipt_threshold]
    if None in feature_thresholds:
        raise Exception("All the threshold must be given")

    for i, threshold in enumerate(feature_thresholds):
        if isinstance(threshold, str):
            feature_thresholds[i] = ast.literal_eval(threshold)

    log("flag_windows", f"Feature thresholds {feature_thresholds}", level=LH.DEBUG)

    good_indexes = np.arange(wdw.shape[0])
    bad_indexes = []
    log("flag_windows", f"Indexes length before cycle, good: {len(good_indexes)} bads: {len(bad_indexes)}", level=LH.DEBUG)
    for column in range(wdw.shape[2]):
        bad_threshold = feature_thresholds[column]
        bad_indexes.append(
                    tf.unique(tf.where(wdw[:,:,column] > bad_threshold)[:,0])[0].numpy()
                )

    bad_idx = np.unique(np.concatenate(bad_indexes))
    log("flag_windows", f"Indexes length after cycle, good: {len(good_indexes)} bads: {len(bad_idx)}", level=LH.DEBUG)

    return tf.cast(tf.where(np.isin(good_indexes, bad_idx), 1, 0), tf.int32)


@action
@f_logger
def windowDropOutliers(data: DataManager,
                       *args,
                       threshold: Union[float, str] = 101.0,
                       **kwargs) -> Dict[str, Dict[str, Any]]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

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
                        *args,
                        training_portion: float = 0.7,
                        validation_portion: float = 0.1,
                        test_portion: float = 0.2,
                        to_dataset: bool = True,
                        **kwargs) -> Tuple[Dict[str, Dict[str, Any]], ...]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    windows = data.dft("Windows")
    targets = data.dft("Targets")
    expectl = data.dft("ExpectedLabel")

    write_msg(f"Positive targets minimum: {tf.math.reduce_min(tf.gather(targets, tf.where(expectl == 0)[:, 0]))}")
    write_msg(f"Negative targets maximum: {tf.math.reduce_max(tf.gather(targets, tf.where(expectl == 1)[:, 0]))}")

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
    assert all([len(x) == 0 for x in intersections])

    def sub_data(idx):
        tmp_data = deepcopy(data)
        for key in tmp_data:
            write_msg(f"Gathering property: {key}")
            tmp_data[key]["tf_values"] = tf.gather(tmp_data[key]["tf_values"], idx)
            write_msg(f"{key}: {tmp_data[key]['tf_values']}")
        return tmp_data

    training = data.sub_select(trn_idx)
    validation = data.sub_select(val_idx)
    test = data.sub_select(tst_idx)

    training.log_dump("Training")
    validation.log_dump("Validation")
    test.log_dump("test")

    if to_dataset:
        training.transform_toDataset(inplace=True)
        validation.transform_toDataset(inplace=True)
        test.transform_toDataset(inplace=True)
        raise Exception("Check")

    return training, validation, test

@action
@f_logger
def balancingConcatenateNG(datasets: List[DataManager],
                           balancing: Tuple[float, ...],
                           *args,
                           **kwargs) -> Tuple[Dict[str, Dict[str, Any]], ...]:
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
    if total_elements > dimensions[max_idx]:
        total_elements = dimensions[max_idx]
    write_msg(f"Total elements required: {total_elements}")

    idx, not_used_idx = [], []
    for i, (balance, dim) in enumerate(zip(balancing, dimensions)):
        write_msg(f"Current balancing cycle: {str((i, balance, dim))}")
        elements_required = math.floor(total_elements * balance)
        if elements_required < 1:
            elements_required = 1

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
    for id, id_not_used, dst in zip(idx, not_used_idx, datasets):
        used_data.append(dst.sub_select(id))
        if id_not_used is not None:
            not_used_data.append(dst.sub_select(id_not_used))

    merged_data = ConcatenateNG(used_data, label="UsedData", logger=logger)
    merged_not_used_data = ConcatenateNG(not_used_data, label="NotUsedData", logger=logger)

    write_msg(f"Total elements used returned: {len(merged_data['Targets']['tf_values'])}")
    write_msg(f"Total elements not used returned: {len(merged_not_used_data['Targets']['tf_values'])}")
    return merged_data, merged_not_used_data

@action
@f_logger
def ConcatenateNG(datasets: List[DataManager],
                  *args,
                  label: str = "MergedData",
                  **kwargs) -> DataManager:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    merged_data = DataManager.merge(datasets)
    merged_data.log_dump(label)

    return merged_data

@action
@f_logger
def BalanceSeparationNG(data: List[DataManager],
                        *args,
                        portion: Tuple[Tuple[float, float, float], ...] = ((0.7, 0.2, 0.1),
                                                                           (0.7, 0.2, 0.1),
                                                                           (0.0, 0.0, 0.0)),
                        balancing: Tuple[Tuple[float, float, float], ...] = ((0.98, 0.02, 0.0),
                                                                             (0.5, 0.5, 0.0),
                                                                             (0.5, 0.5, 0.0)),
                        test_get_everything: bool = False,
                        **kwargs) -> Tuple[Dict[str, Dict[str, Any]], ...]:
    """BalanceSeparationNG.
    Function used to both apply portioning to datasets and also balancing.
    Then mixes the datasets as required in train, validation and test.

    Parameters
    ----------
    data : List[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]]
        data
    args :
        args
    portion : Tuple[Tuple[float, float, float], ...]
        portion that should be used for each dataset in training-validaton-test;
        Data should be organized as follows, this array must contain one tuple
        for each Tuple in the data arg.
        Each tuple must contain 3 portion elements, where each portion must
        sum to 1.
        Each element portion refers to the portion that should be used respectively
        for Training, validation and test.
    balancing : Tuple[Tuple[float, float, float], ...]
        balancing
    logger : LH
        logger
    kwargs :
        kwargs

    Returns
    -------
    Tuple[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], ...]

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
        test, test_not_used = balancingConcatenateNG(
                    datasets['tst'],
                    balancing[2],
                    logger=logger
                )

    return training, validation, test

