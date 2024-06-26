# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from SNN2.src.actions.separation import ConcatenateNG
from SNN2.src.core.data.DataManager import DataManager

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from SNN2.src.decorators.decorators import actions, action, f_logger
from SNN2.src.io.logger import LogHandler as LH

from typing import Any, Dict, Optional, Union, List, Tuple

@action
def GoodBadGraySeparation(*args,
                          df: pd.DataFrame = None,
                          delay_lower: float = 20.0,
                          delay_upper: float = 50.0,
                          drop_lower: float = 0.0,
                          drop_upper: float = 0.7,
                          **kwargs) -> pd.DataFrame:
    if df is None:
        raise Exception("The dataframe passed is None")
    exps = df[["exp_id", "problem", "value"]].drop_duplicates().copy()
    goods_exps = exps[(exps["problem"] == "good") | \
                    ((exps["problem"] == "delay") & (exps["value"] < delay_lower)) | \
                    ((exps["problem"] == "drop") & (exps["value"] < drop_lower))]["exp_id"].values
    bads_exps = exps[((exps["problem"] == "delay") & (exps["value"] > delay_upper)) | \
                   ((exps["problem"] == "drop") & (exps["value"] > drop_upper))]["exp_id"].values
    grays_exps = exps[((exps["problem"] == "delay") & (exps["value"] > delay_lower) & (exps["value"] < delay_upper)) | \
                    ((exps["problem"] == "drop") & (exps["value"] > drop_lower) & (exps["value"] < drop_upper))]["exp_id"].values
    good_df = df[df["exp_id"].isin(goods_exps)].copy()
    bad_df = df[df["exp_id"].isin(bads_exps)].copy()
    gray_df = df[df["exp_id"].isin(grays_exps)].copy()
    good_df["Dataset"] = "good"
    bad_df["Dataset"] = "bad"
    gray_df["Dataset"] = "gray"
    return pd.concat([good_df, bad_df, gray_df])

@action
def durationSeparation(*args,
                       df: pd.DataFrame = None,
                       **kwargs) -> List[pd.DataFrame]:
    if df is None:
        raise Exception("The dataframe passed is None")
    seconds = df.groupby(["exp_id"])["second"].max().reset_index()
    exp_groups = [seconds[seconds["second"] == x]["exp_id"].values for x in seconds["second"].unique()]
    dfs = [df[df["exp_id"].isin(x)] for x in exp_groups]
    return dfs


@action
def featureMean(data: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_mean(data_tensor, 0)

@action
def featureStd(data: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_std(data_tensor, 0)

@action
def featureMax(data: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_max(data_tensor, 0)

@action
def featureMin(data: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    data_tensor = tf.reshape(data, [-1, data.shape[-1]])
    return tf.math.reduce_min(data_tensor, 0)

@action
@tf.autograph.experimental.do_not_convert
def normalize(data: tf.Tensor,
              mean: tf.Tensor,
              std: tf.Tensor, *args, **kwargs) -> tf.Tensor:
    return tf.map_fn(lambda x: (x-mean)/std, data)

@action
@tf.autograph.experimental.do_not_convert
def normalizeMinMax(data: Tuple[tf.data.Dataset, tf.data.Dataset],
                    max: tf.Tensor,
                    min: tf.Tensor, *args, **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    data_wdw = data[0]
    data_trg = data[1]
    return (data_wdw.map(lambda x: (x-min)/(max-min)), data_trg)

@action
@f_logger
def generateTripletsNG(anchor: DataManager,
                       *args,
                       **kwargs) -> DataManager:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

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

    return GenerateGrayTriplets(anchor, goods, bads, logger=logger)


@action
def generateTriplets(anchor: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                     *args,
                     log: Optional[LH] = None,
                     **kwargs) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
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
    index_B_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_C, [1, len(idx_C)]), repeats=reps, axis=0), [-1]))[:len(idx_B)]
    reps = math.ceil(len(idx_C)/len(idx_B))
    index_C_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_B, [1, len(idx_B)]), repeats=reps, axis=0), [-1]))[:len(idx_C)]
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
                               *args,
                               **kwargs) -> DataManager:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx_a = tf.range(anchor.dft("Windows").shape[0])
    idx_p = tf.range(positive.dft("Windows").shape[0])
    idx_n = tf.range(negative.dft("Windows").shape[0])

    reps_p = math.ceil(len(idx_a)/len(idx_p))
    idx_p = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_p, [1, len(idx_p)]), repeats=reps_p, axis=0), [-1]))[:len(idx_a)]
    write_msg(f"p indexes: {idx_p}, len: {len(idx_p)}", level=LH.DEBUG)
    reps_n = math.ceil(len(idx_a)/len(idx_n))
    idx_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_n, [1, len(idx_n)]), repeats=reps_n, axis=0), [-1]))[:len(idx_a)]
    write_msg(f"n indexes: {idx_n}, len: {len(idx_p)}", level=LH.DEBUG)

    new_positive = positive.sub_select(idx_p)
    new_negative = negative.sub_select(idx_n)
    new_anchor = anchor.sub_select(idx_a)
    write_msg(f"P shape: {new_positive['Windows']['tf_values'].shape}")
    write_msg(f"A shape: {new_anchor['Windows']['tf_values'].shape}")
    write_msg(f"N shape: {new_negative['Windows']['tf_values'].shape}")

    new_anchor["TripletDst"] = None
    new_anchor["TripletDst"].set_default(
                tf.stack([new_positive.dft("Windows"),
                          new_anchor.dft("Windows"),
                          new_negative.dft("Windows")],
                         axis=1))
    new_anchor["TripletDst"]["TfDataset"] = tf.data.Dataset.from_tensor_slices(
                (new_positive.dft("Windows"),
                 new_anchor.dft("Windows"),
                 new_negative.dft("Windows")))
    return new_anchor

@action
def generatePredictionTripletsReverse(anchor: Tuple[tf.data.Dataset, tf.data.Dataset],
                                      positive: Tuple[tf.data.Dataset, tf.data.Dataset],
                                      negative: Tuple[tf.data.Dataset, tf.data.Dataset],
                                      *args, **kwargs) -> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], ...]:
    anchor_wdw = tf.convert_to_tensor(np.array(list(anchor[0].as_numpy_iterator())))
    positive_wdw = tf.convert_to_tensor(np.array(list(positive[0].as_numpy_iterator())))
    negative_wdw = tf.convert_to_tensor(np.array(list(negative[0].as_numpy_iterator())))
    targets = tf.convert_to_tensor(np.array(list(anchor[1].as_numpy_iterator())))
    idx_a = tf.range(anchor_wdw.shape[0])
    idx_p = tf.range(positive_wdw.shape[0])
    idx_n = tf.range(negative_wdw.shape[0])

    reps_p = math.ceil(len(idx_a)/len(idx_p))
    idx_p = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_p, [1, len(idx_p)]), repeats=reps_p, axis=0), [-1]))[:len(idx_a)]
    reps_n = math.ceil(len(idx_a)/len(idx_n))
    idx_n = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_n, [1, len(idx_n)]), repeats=reps_n, axis=0), [-1]))[:len(idx_a)]
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
                         *args,
                         **kwargs) -> Dict[str, Dict[str, Any]]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx_a = tf.range(anchor.dft("Windows").shape[0])
    idx_p = tf.range(positive.dft("Windows").shape[0])
    idx_n = tf.range(negative.dft("Windows").shape[0])
    write_msg(f"Len p: {len(idx_p)}", level=LH.DEBUG)
    write_msg(f"Len n: {len(idx_n)}", level=LH.DEBUG)

    anchor_wdw_shape = anchor.dft("Windows").shape
    reps_B = math.ceil(anchor.dft("Windows").shape[0]/len(idx_p))
    write_msg(f"Repetitions p: {reps_B}", level=LH.DEBUG)
    idx_B = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_p, [1, len(idx_p)]), repeats=reps_B, axis=0), [-1]))[:anchor_wdw_shape[0]]
    write_msg(f"p chosen: {idx_B}", level=LH.DEBUG)
    write_msg(f"dimension idx p chosen: {len(idx_B)}", level=LH.DEBUG)

    reps_C = math.ceil(anchor.dft("Windows").shape[0]/len(idx_n))
    write_msg(f"Repetitions n: {reps_C}", level=LH.DEBUG)
    idx_C = tf.random.shuffle(tf.reshape(tf.repeat(tf.reshape(idx_n, [1, len(idx_n)]), repeats=reps_C, axis=0), [-1]))[:anchor_wdw_shape[0]]
    write_msg(f"n chosen: {idx_C}", level=LH.DEBUG)
    write_msg(f"dimension idx n chosen: {len(idx_C)}", level=LH.DEBUG)

    buoni = positive.sub_select(idx_B, inplace=False)
    cattivi = negative.sub_select(idx_C, inplace=False)
    new_anchor = anchor.sub_select(idx_a, inplace=False)
    buoni_cattivi = ConcatenateNG([buoni, cattivi], label="BuoniCattivi", logger=logger)
    write_msg(f"{buoni.dft('Windows').shape[0]}")
    write_msg(f"{cattivi.dft('Windows').shape[0]}")

    idx_B = tf.range(buoni.dft("Windows").shape[0])
    idx_C = tf.range(buoni.dft("Windows").shape[0], buoni.dft("Windows").shape[0]+cattivi.dft("Windows").shape[0])
    write_msg(f"p chosen: {idx_B.shape}", level=LH.DEBUG)
    write_msg(f"n chosen: {idx_C.shape}", level=LH.DEBUG)
    write_msg(f"Anchor expected labels shape: {anchor.dft('ExpectedLabel').shape}")
    write_msg(f"{tf.where(anchor.dft('ExpectedLabel') == 0, idx_B, idx_C)}")

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
    # new_anchor["Windows"]["tf_values"] = tf.stack(
    #             [pos_net["Windows"]["tf_values"],
    #              new_anchor["Windows"]["tf_values"],
    #              neg_net["Windows"]["tf_values"]],
    #             axis=1)

    return new_anchor

@action
def print_data(*args, df: pd.DataFrame = None, **kwargs) -> None:
    print(df, *args, **kwargs)
    return df

def action_selector(obj, *args, **kwargs):
    if obj in actions.keys():
        return actions[obj](*args, **kwargs)
    else:
        raise Exception(f"{obj} action not found")

