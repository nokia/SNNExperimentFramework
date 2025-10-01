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
from tensorflow import keras
from tensorflow.python.ops.array_ops import inplace_add
from tensorflow.python.util.deprecation import inspect
from SNN2.src.core.data.DataManager import DataManager

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from SNN2.src.decorators.decorators import actions, action, f_logger
from SNN2.src.io.logger import LogHandler as LH

from typing import Any, Callable, Dict, Optional, Union, List, Tuple

@f_logger
def apply_post_operations(request: Dict[str, Any],
                          tf_values: tf.Tensor,
                          **kwargs) -> tf.Tensor:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    for operation, o_arg, o_kwarg in zip(request["post_operation"],
                                         request["post_operation_args"],
                                         request["post_operation_kwargs"]):
        write_msg(f"Appling {operation.__qualname__}")
        tf_values = operation(tf_values, *o_arg, **o_kwarg)

    return tf_values


@action
@f_logger
def kaggleDataVsLabelSeparation(df: pd.DataFrame,
                                *args,
                                requests: Union[DataManager, None] = None,
                                **kwargs) -> DataManager:
    if requests is None:
        raise Exception

    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    write_msg(f"DF: \n{df}")
    counts = df['web_service'].value_counts()
    df = df[~df['web_service'].isin(counts[counts < 100].index)]
    df = df.drop_duplicates()
    write_msg(f"Web services: {df['web_service'].unique()}")
    write_msg(f"len ws: {len(df['web_service'].unique())}")
    write_msg(f"len ws: {df['web_service'].value_counts()}")
    write_msg(f"len ws: {df['web_service'].value_counts()}")
    none_elem = df.isna().sum()
    write_msg(f"None elements: {none_elem}")
    dup_elem = df.duplicated().sum()
    write_msg(f"Dup elements: {dup_elem}")

    write_msg(f"Requests: {requests}", level=LH.DEBUG)
    for key, req in requests.items():
        write_msg(f"{key} - Requests columns: {req['columns']}")
        req["Values"] = df[req["columns"]].values
        write_msg(f"Including {req['columns']} into {key}['Values'] request object")
        write_msg(f"Dtype required for the tensor: {req['dtype']}")

        tf_values = tf.convert_to_tensor(df[req["columns"]].values, dtype=req["dtype"])

        if req["post_operation"] is not None:
            tf_values = apply_post_operations(req, tf_values, logger=logger)

        req.append_default(tf_values)

    for req in requests:
        requests[req].set_default(tf.concat(requests[req].dft(), 0))

    return requests

@action
@f_logger
def kaggleShuffle(data: DataManager, *args,
                  **kwargs) -> None:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    features = data.dft("Samples")
    ids = tf.range(features.shape[0])
    ids = tf.random.shuffle(ids)

    data.sub_select(ids, inplace=True)


@action
@f_logger
def kaggleTrnValTestSeparation(data: DataManager, *args,
                               trn_portion: float = 0.6,
                               val_portion: float = 0.2,
                               **kwargs) -> Tuple[DataManager, DataManager, DataManager]:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    features = data.dft("Samples")
    ids = tf.range(features.shape[0])
    write_msg(f"Available ids: {ids}")

    train_n = math.floor(features.shape[0]*trn_portion)
    validation_n = math.floor(features.shape[0]*val_portion)
    trn_ids = ids[:train_n]
    val_ids = ids[train_n:train_n+validation_n]
    tst_ids = ids[train_n+validation_n:]

    trn_data = data.sub_select(trn_ids)
    val_data = data.sub_select(val_ids)
    tst_data = data.sub_select(tst_ids)

    return trn_data, val_data, tst_data

@action
@f_logger
def generateKaggleTriplets(anchor: DataManager,
                           *args,
                           sample_name: str = "Samples",
                           target_name: str = "Targets",
                           **kwargs) -> DataManager:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx = tf.range(anchor.dft(sample_name).shape[0])
    write_msg(f"Total idx: {len(idx)}", level=LH.DEBUG)

    targets = anchor.dft(target_name)
    u_targets, _ = tf.unique(tf.reshape(targets, [-1]))

    p_idx = None
    n_idx = None

    for tg in u_targets.numpy():
        write_msg(f"Looking for target {tg}")
        positive_idx = tf.gather(idx, tf.where(targets == tg)[:, 0])
        negative_idx = tf.gather(idx, tf.where(targets != tg)[:, 0])

        write_msg(f"Positive idx: {positive_idx.shape}")
        write_msg(f"Negative idx: {negative_idx.shape}")

        reps = math.ceil(len(idx)/len(positive_idx))
        positive_idx = tf.repeat(tf.random.shuffle(positive_idx), repeats=reps, axis=0)[:targets.shape[0]]
        reps = math.ceil(len(idx)/len(negative_idx))
        negative_idx = tf.repeat(tf.random.shuffle(negative_idx), repeats=reps, axis=0)[:targets.shape[0]]

        write_msg(f"Positive idx: {positive_idx.shape}")
        write_msg(f"Negative idx: {negative_idx.shape}")

        if p_idx is None:
            p_idx = tf.where(tf.reshape(targets, [-1]) == tg, positive_idx, [-1])
            n_idx = tf.where(tf.reshape(targets, [-1]) == tg, negative_idx, -1)
        else:
            p_idx = tf.where(tf.reshape(targets, [-1]) == tg, positive_idx, p_idx)
            n_idx = tf.where(tf.reshape(targets, [-1]) == tg, negative_idx, n_idx)

    write_msg(f"Positive idx shape: {p_idx.shape}")
    write_msg(f"Negative idx shape: {n_idx.shape}")
    write_msg(f"Positive idx: {p_idx}")
    write_msg(f"Negative idx: {n_idx}")
    write_msg(f"Any -1 in p_idx: {tf.where(p_idx == -1)}")
    write_msg(f"Any -1 in n_idx: {tf.where(n_idx == -1)}")

    assert len(tf.where(p_idx == -1)) == 0
    assert len(tf.where(n_idx == -1)) == 0

    p_samples = anchor.sub_select(p_idx)
    n_samples = anchor.sub_select(n_idx)
    a_samples = anchor.sub_select(idx)

    a_samples["TripletDst"] = None
    a_samples["TripletDst"].set_default(
                tf.stack([p_samples.dft(sample_name),
                          a_samples.dft(sample_name),
                          n_samples.dft(sample_name)],
                         axis=1))
    a_samples["TripletDst"]["TfDataset"] = tf.data.Dataset.from_tensor_slices(
                (p_samples.dft(sample_name),
                 a_samples.dft(sample_name),
                 n_samples.dft(sample_name)))

    return a_samples

@action
@f_logger
def generateKaggleCategorical(anchor: DataManager,
                              *args,
                              sample_name: str = "Samples",
                              target_name: str = "Targets",
                              dst_name: str = "SamplesDst",
                              num_classes: int = 141,
                              **kwargs) -> DataManager:
    logger, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx = tf.range(anchor.dft(sample_name).shape[0])
    write_msg(f"Total idx: {len(idx)}", level=LH.DEBUG)

    targets = anchor.dft(target_name)
    categorical_targets = keras.utils.to_categorical(targets, num_classes)
    write_msg(f"Categorical targets: {categorical_targets}")

    samples = anchor.sub_select(idx)
    samples[dst_name] = None
    samples[dst_name].set_default(samples.dft(sample_name))
    samples[dst_name]["TfDataset"] = tf.data.Dataset.from_tensor_slices((samples.dft(sample_name), categorical_targets))
    samples[dst_name]["CategoricalTargets"] = categorical_targets

    return samples
