#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the graphNU grapheneral Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# graphNU grapheneral Public License for more details.
#
# You should have received a copy of the graphNU grapheneral Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2021 Mattia Milani <mattia.milani@nokia.com>

"""
Fit function Wrapper
====================


"""

import math
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model

from SNN2.src.io.files import FileHandler as FH
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.actions.actionWrapper import action_selector as AS
from SNN2.src.util.helper import dst2tensor
from SNN2.src.decorators.decorators import train_enhancement, grays

from typing import Generator, Tuple, Callable, Union, Any, Optional

def __where(*args, flat=True, **kwargs) -> tf.Tensor:
    if flat:
        return tf.where(*args, **kwargs)[:,0]
    return tf.where(*args, **kwargs)

def validation_flag(distances: Tuple[np.ndarray, np.ndarray],
                    expected_flags: tf.Tensor,
                    margin: float = 0.) -> tf.Tensor:
    correctness, _ = prediction_flag(distances, margin=margin)
    expected_flags  = tf.cast(expected_flags, tf.int8)
    reverse_expected_flags = tf.cast(tf.where(expected_flags == 0, 1, 0), tf.int8)
    flags = tf.where(correctness == 0, expected_flags, correctness)
    flags = tf.where(correctness == 1, reverse_expected_flags, flags)
    return flags

def prediction_flag(distances: Tuple[np.ndarray, np.ndarray],
                    margin: float = 0.) -> Tuple[tf.Tensor, Tuple[np.ndarray, np.ndarray]]:
    ap: np.ndarray = distances[0]
    an: np.ndarray = distances[1]
    diff = tf.convert_to_tensor(ap - an)
    flags = tf.where(diff <= -margin, 0, 2)
    flags = tf.where(diff > margin, 1, flags)
    flags = tf.cast(flags, tf.int8)
    return flags, distances

def save_gray_stats(correct: tf.Tensor,
                    wrong: tf.Tensor,
                    output: str,
                    undecided: bool = False,
                    frozen_predictions: Optional[int] = None) -> None:
    tp = tf.where(correct == 0).shape[0]
    tn = tf.where(correct == 1).shape[0]
    fp = tf.where(wrong == 0).shape[0]
    fn = tf.where(wrong == 1).shape[0]
    n_undec = tf.where(wrong == 2).shape[0]
    accuracy = (tp+tn)/(len(correct) + len(wrong))

    df = pd.DataFrame({'TP': [tp],
                       'FP': [fp],
                       'TN': [tn],
                       'FN': [fn],
                       'Accuracy': [accuracy]})
    if undecided:
        df["Undecided"] = n_undec

    if frozen_predictions is not None:
        df["Frozen"] = frozen_predictions

    if FH.exists(output):
        df.to_csv(output, mode="a", header=False, index=False)
    else:
        df.to_csv(output, mode="a", header=True, index=False)

def save_gray_predictions(predictions: tf.Tensor,
                          distances: Tuple[np.ndarray, np.ndarray],
                          output: str) -> None:
    df = None
    if FH.exists(output):
        df = AS("load", output)

    sample_id = np.repeat(np.arange(len(predictions)), 3)
    iteration = 0
    if df is not None:
        iteration = df["Iteration"].unique()[-1] + 1

    kpi = np.repeat(np.array([["AP", "AN", "Flag"]]), len(predictions), axis=0).flatten()
    values = np.stack((distances[0], distances[1], predictions.numpy()), axis=-1).flatten()
    tmp_df = pd.DataFrame({
            "SampleID": sample_id,
            "Iteration": iteration,
            "KPI": kpi,
            "Values": values
        })
    df = pd.concat([df, tmp_df]) if df is not None else tmp_df
    AS("write", output, df=df, mode="w+", header=True, index=False)

def predict(model: Model, dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    ap, an = model.predict(dataset)
    return ap, an

def flag_windows(wdw: tf.Tensor,
                 pdr_threshold: float,
                 bdr_threshold: float,
                 avg_ipt_threshold: float,
                 std_ipt_threshold: float,
                 skw_ipt_threshold: float,
                 kur_ipt_threshold: float,
                 log: Union[LH, Callable] = None) -> tf.Tensor:

    log("flag_windows", "Required generation of true windows", level=LH.DEBUG)
    feature_thresholds = [pdr_threshold, bdr_threshold, avg_ipt_threshold, std_ipt_threshold, skw_ipt_threshold, kur_ipt_threshold]
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

    return tf.cast(tf.where(np.isin(good_indexes, bad_idx), 1, 0), tf.int8)

def convert(val: str, caster: Callable) -> Any:
    return caster(val)

def log_none(obj: str, msg: str, level: int = 0):
    pass

def subIdx(idx: tf.Tensor, flag: tf.Tensor, reverse: bool = False) -> tf.Tensor:
    if not reverse:
        subidx = tf.reshape(tf.gather(idx, tf.where(flag).numpy()), [-1])
    else:
        subidx = tf.reshape(tf.gather(idx, tf.where(tf.where(flag, False, True)).numpy()), [-1])
    return subidx

def separate_training_triplets(training_data_samples: tf.data.Dataset,
                               training_data_classes: tf.data.Dataset,
                               log: Optional[LH] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    classes = dst2tensor(training_data_classes)
    id, _, count = tf.unique_with_counts(classes)
    if log is not None:
        log(f"SeparateTrainingTriplets", f"classes: {id.numpy(), count.numpy()}", level=LH.DEBUG)

    original_training_wdw = dst2tensor(training_data_samples)

    original_training_C = tf.data.Dataset.from_tensor_slices(
                tf.gather(original_training_wdw, __where(classes == 1)))
    if log is not None:
        log(f"SeparateTrainingTriplets", f"# Negative training samples: {original_training_C.cardinality().numpy()}")

    original_training_B = tf.data.Dataset.from_tensor_slices(
                tf.gather(original_training_wdw, __where(classes == 0)))
    if log is not None:
        log(f"SeparateTrainingTriplets", f"# Positive training samples: {original_training_B.cardinality().numpy()}")

    return (original_training_B, original_training_C)

def remove_undecided(samples: tf.Tensor,
                     flags: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    return tf.gather(samples, __where(flags != 2)), tf.gather(flags, __where(flags != 2))

def regenerate_gray_triplets(predicted_flags: tf.Tensor,
                             gray_samples: tf.Tensor,
                             tripletGenerator: Callable,
                             goodSamples: tf.data.Dataset,
                             badSamples: tf.data.Dataset,
                             portion: float = 1.,
                             log: Optional[LH] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    if log is not None:
        id, _, count = tf.unique_with_counts(predicted_flags)
        log(f"regenerate_gray_triplets", f"predicted flags: {id.numpy(), count.numpy()}", level=LH.DEBUG)
    gray_samples, predicted_flags = remove_undecided(gray_samples, predicted_flags)
    if log is not None:
        id, _, count = tf.unique_with_counts(predicted_flags)
        log(f"regenerate_gray_triplets", f"predicted flags: {id.numpy(), count.numpy()}", level=LH.DEBUG)

    gray_fake_trg = tf.where(predicted_flags == 0, 100., 0.)
    # samples_to_keep = math.ceil(gray_fake_trg.shape[0]*portion)
    # random_gray_indexes = tf.random.shuffle(tf.range(gray_fake_trg.shape[0]))[:samples_to_keep]
    gray_training = (
                tf.data.Dataset.from_tensor_slices(gray_samples),
                tf.data.Dataset.from_tensor_slices(gray_fake_trg),
                tf.data.Dataset.from_tensor_slices(tf.cast(tf.fill(dims=(len(gray_samples)), value=2), tf.int8)),
                tf.data.Dataset.from_tensor_slices(predicted_flags)
            )

    return tripletGenerator(gray_training, (goodSamples,), (badSamples,), log=log)

def merge_triplets(new_triplets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                   old_triplets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                   batch_size: Optional[int] = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    new_triplets_samples = new_triplets[0]
    new_triplets_targets = tf.data.Dataset.from_tensor_slices(tf.cast(dst2tensor(new_triplets[1]), tf.float64))
    new_triplets_classes = new_triplets[2]
    new_triplets_exp_l = new_triplets[3]
    trn_tri_samples, trn_tri_targets, trn_tri_classes, trn_tri_exp_l = old_triplets

    trn_tri_samples = trn_tri_samples.concatenate(new_triplets_samples)
    trn_tri_targets = trn_tri_targets.concatenate(new_triplets_targets)
    trn_tri_classes = trn_tri_classes.concatenate(new_triplets_classes)
    trn_tri_classes = trn_tri_exp_l.concatenate(new_triplets_exp_l)

    trn_tri_samples = trn_tri_samples.shuffle(50000)

    if batch_size is not None:
        trn_tri_samples = trn_tri_samples.batch(batch_size)
        trn_tri_targets = trn_tri_targets.batch(batch_size)
        trn_tri_classes = trn_tri_classes.batch(batch_size)
        trn_tri_exp_l = trn_tri_exp_l.batch(batch_size)
    return trn_tri_samples, trn_tri_targets, trn_tri_classes, trn_tri_exp_l

@train_enhancement
def gray_update(model: Model,
                pred_gray_triplets: tf.data.Dataset,
                gray_targets: tf.data.Dataset,
                training_data: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                training_triplets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                gray_samples: tf.Tensor,
                GrayTripletsGenerator: Callable,
                pdr_threshold: Optional[Union[float, str]] = None,
                bdr_threshold: Optional[Union[float, str]] = None,
                avg_ipt_threshold: Optional[Union[float, str]] = None,
                std_ipt_threshold: Optional[Union[float, str]] = None,
                skw_ipt_threshold: Optional[Union[float, str]] = None,
                kur_ipt_threshold: Optional[Union[float, str]] = None,
                gray_stats_output: Optional[str] = None,
                batch_size: Union[int, str] = 0,
                logger: LH = None
                ) -> Generator:
    if gray_stats_output is None:
        raise Exception("An output CSV file for the grays stats must be provided")

    log = log_none if logger is None else logger

    batch_size: int = convert(batch_size, int) if isinstance(batch_size, str) else batch_size
    pdr_threshold: float = convert(pdr_threshold, float) if isinstance(pdr_threshold, str) else pdr_threshold
    bdr_threshold: float = convert(bdr_threshold, float) if isinstance(bdr_threshold, str) else bdr_threshold
    avg_ipt_threshold: float = convert(avg_ipt_threshold, float) if isinstance(avg_ipt_threshold, str) else avg_ipt_threshold
    std_ipt_threshold: float = convert(std_ipt_threshold, float) if isinstance(std_ipt_threshold, str) else std_ipt_threshold
    skw_ipt_threshold: float = convert(skw_ipt_threshold, float) if isinstance(skw_ipt_threshold, str) else skw_ipt_threshold
    kur_ipt_threshold: float = convert(kur_ipt_threshold, float) if isinstance(kur_ipt_threshold, str) else kur_ipt_threshold

    log("gray_update", "parameters conversion compleated", level=LH.DEBUG)

    true_flags = flag_windows(gray_samples,
                              pdr_threshold, bdr_threshold,
                              avg_ipt_threshold, std_ipt_threshold,
                              skw_ipt_threshold, kur_ipt_threshold,
                              log=log)

    log("gray_update", "True flags generated", level=LH.DEBUG)


    while True:
        print("\n---- Grays evaluation, training dataset enhancement----\n")

        # Predict the grays distances and obtain the class flags
        predicted_flags, _ = prediction_flag(predict(model, pred_gray_triplets))
        assert len(tf.where(predicted_flags == 2)) == 0

        correct = tf.gather(predicted_flags, tf.where(predicted_flags == true_flags))
        wrong = tf.gather(predicted_flags, tf.where(predicted_flags != true_flags))
        save_gray_stats(correct, wrong, gray_stats_output)

        # Regenerate the training dataset with also this new gray data
        original_training_B, original_training_C = separate_training_triplets(
                    training_data[0], training_data[2]
                )

        gray_triplets = regenerate_gray_triplets(predicted_flags, gray_samples, GrayTripletsGenerator,
                                                 original_training_B, original_training_C)

        # Return the new training dataset.
        yield merge_triplets(gray_triplets, training_triplets, batch_size)

@train_enhancement
def gray_update_margin(model: Model,
                       pred_gray_triplets: tf.data.Dataset,
                       training_data: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                       training_triplets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                       gray_samples: tf.Tensor,
                       gray_exp_l: tf.Tensor,
                       GrayTripletsGenerator: Callable,
                       margin: Union[float, str] = 0.0,
                       gray_stats_output: Optional[str] = None,
                       gray_prediction_output: Optional[str] = None,
                       batch_size: Union[int, str] = 0,
                       verbose: int = 1,
                       logger: Optional[LH] = None) -> Generator:
    if gray_stats_output is None:
        raise Exception("An output CSV file for the grays stats must be provided")

    log = log_none if logger is None else logger

    batch_size: int = convert(batch_size, int) if isinstance(batch_size, str) else batch_size
    margin: float = convert(margin, float) if isinstance(margin, str) else margin

    true_flags = gray_exp_l

    log("gray_update", "True flags obtained", level=LH.DEBUG)

    # Regenerate the training dataset with also this new gray data
    original_training_B, original_training_C = separate_training_triplets(
                training_data[0], training_data[2],
                log=log
            )

    while True:
        log("gray_update", "---- Grays evaluation, training dataset enhancement----")

        # Predict the grays distances and obtain the class flags
        predicted_flags, predicted_distances = prediction_flag(predict(model, pred_gray_triplets, verbose=verbose), margin=margin)

        correct = tf.reshape(tf.gather(predicted_flags, tf.where(predicted_flags == true_flags)), [-1])
        wrong = tf.reshape(tf.gather(predicted_flags, tf.where(predicted_flags != true_flags)), [-1])

        log("gray_update", f"Correct flags: {len(correct)}", level=LH.DEBUG)
        id, _, count = tf.unique_with_counts(correct)
        log("gray_update", f"Correct labels: {(id.numpy(), count.numpy())}")
        log("gray_update", f"Wrong flags: {len(wrong)}", level=LH.DEBUG)
        id, _, count = tf.unique_with_counts(wrong)
        log("gray_update", f"Wrong labels: {(id.numpy(), count.numpy())}")

        save_gray_stats(correct, wrong, gray_stats_output, undecided=True)
        if gray_prediction_output is not None:
            save_gray_predictions(predicted_flags, predicted_distances, gray_prediction_output)

        gray_triplets = regenerate_gray_triplets(predicted_flags,
                                                 gray_samples,
                                                 GrayTripletsGenerator,
                                                 original_training_B,
                                                 original_training_C,
                                                 log=log)

        # Return the new training dataset.
        yield merge_triplets(gray_triplets, training_triplets, batch_size)

@train_enhancement
def gray_update_margin_general(model: Model,
                               pred_gray_triplets: tf.data.Dataset,
                               good_dst: tf.data.Dataset,
                               bad_dst: tf.data.Dataset,
                               training_triplets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                               gray_samples: tf.Tensor,
                               gray_exp_l: tf.Tensor,
                               GrayTripletsGenerator: Callable,
                               margin: Union[float, str] = 0.0,
                               gray_stats_output: Optional[str] = None,
                               gray_prediction_output: Optional[str] = None,
                               batch_size: Union[int, str] = 0,
                               verbose: int = 1,
                               logger: Optional[LH] = None) -> Generator:
    if gray_stats_output is None:
        raise Exception("An output CSV file for the grays stats must be provided")

    log = log_none if logger is None else logger

    batch_size: int = convert(batch_size, int) if isinstance(batch_size, str) else batch_size
    margin: float = convert(margin, float) if isinstance(margin, str) else margin

    while True:
        log("gray_update", "---- Grays evaluation, training dataset enhancement----")

        # Predict the grays distances and obtain the class flags
        dst = pred_gray_triplets[0].batch(100)
        predicted_flags, predicted_distances = prediction_flag(predict(model, dst, verbose=verbose), margin=margin)

        correct = tf.reshape(tf.gather(predicted_flags, tf.where(predicted_flags == gray_exp_l)), [-1])
        wrong = tf.reshape(tf.gather(predicted_flags, tf.where(predicted_flags != gray_exp_l)), [-1])

        id, _, count = tf.unique_with_counts(correct)
        log("gray_update", f"Correct labels - {len(correct)}: {(id.numpy(), count.numpy())}", level=LH.DEBUG)
        id, _, count = tf.unique_with_counts(wrong)
        log("gray_update", f"Wrong labels - {len(wrong)}: {(id.numpy(), count.numpy())}", level=LH.DEBUG)

        save_gray_stats(correct, wrong, gray_stats_output, undecided=True)
        if gray_prediction_output is not None:
            save_gray_predictions(predicted_flags, predicted_distances, gray_prediction_output)

        np.set_printoptions(formatter={'float_kind':'{:f}'.format})
        log("gray_update", f"Example of triplets pred_gray_triplets:\n{np.round(dst2tensor(pred_gray_triplets[0].take(2)).numpy(), 3)}", level=LH.DEBUG)

        gray_triplets = regenerate_gray_triplets(predicted_flags,
                                                 gray_samples,
                                                 GrayTripletsGenerator,
                                                 good_dst[0],
                                                 bad_dst[0],
                                                 log=log)

        log("gray_update", f"Example of triplets gray_triplets:\n{np.round(dst2tensor(gray_triplets[0].take(2)).numpy(),3)}", level=LH.DEBUG)
        np.set_printoptions()
        # Return the new training dataset.
        yield merge_triplets(gray_triplets, training_triplets, batch_size)

@train_enhancement
def gray_update_margin_freeze(model: Model,
                              pred_gray_triplets: tf.data.Dataset,
                              gray_targets: tf.data.Dataset,
                              training_data: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                              training_triplets: Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
                              gray_samples: tf.Tensor,
                              GrayTripletsGenerator: Callable,
                              margin: Union[float, str] = 0.0,
                              sequence_threshold: Union[int, str] = 5,
                              pdr_threshold: Optional[Union[float, str]] = None,
                              bdr_threshold: Optional[Union[float, str]] = None,
                              avg_ipt_threshold: Optional[Union[float, str]] = None,
                              std_ipt_threshold: Optional[Union[float, str]] = None,
                              skw_ipt_threshold: Optional[Union[float, str]] = None,
                              kur_ipt_threshold: Optional[Union[float, str]] = None,
                              gray_stats_output: Optional[str] = None,
                              gray_prediction_output: Optional[str] = None,
                              batch_size: Union[int, str] = 0,
                              logger: LH = None) -> Generator:
    if gray_stats_output is None:
        raise Exception("An output CSV file for the grays stats must be provided")

    log = log_none if logger is None else logger

    batch_size: int = convert(batch_size, int) if isinstance(batch_size, str) else batch_size
    margin: float = convert(margin, float) if isinstance(margin, str) else margin
    sequence_threshold: int = convert(sequence_threshold, int) if isinstance(sequence_threshold, str) else sequence_threshold
    pdr_threshold: float = convert(pdr_threshold, float) if isinstance(pdr_threshold, str) else pdr_threshold
    bdr_threshold: float = convert(bdr_threshold, float) if isinstance(bdr_threshold, str) else bdr_threshold
    avg_ipt_threshold: float = convert(avg_ipt_threshold, float) if isinstance(avg_ipt_threshold, str) else avg_ipt_threshold
    std_ipt_threshold: float = convert(std_ipt_threshold, float) if isinstance(std_ipt_threshold, str) else std_ipt_threshold
    skw_ipt_threshold: float = convert(skw_ipt_threshold, float) if isinstance(skw_ipt_threshold, str) else skw_ipt_threshold
    kur_ipt_threshold: float = convert(kur_ipt_threshold, float) if isinstance(kur_ipt_threshold, str) else kur_ipt_threshold

    log("gray_update", "parameters conversion compleated", level=LH.DEBUG)

    true_flags = flag_windows(gray_samples,
                              pdr_threshold, bdr_threshold,
                              avg_ipt_threshold, std_ipt_threshold,
                              skw_ipt_threshold, kur_ipt_threshold,
                              log=log)
    log("gray_update", "True flags generated", level=LH.DEBUG)

    counter_flags = tf.zeros(len(true_flags))
    previous_flags = None

    while True:
        print("\n---- Grays evaluation, training dataset enhancement----\n")

        # Predict the grays distances and obtain the class flags
        predicted_flags, predicted_distances = prediction_flag(predict(model, pred_gray_triplets), margin=margin)

        if gray_prediction_output is not None:
            save_gray_predictions(predicted_flags, predicted_distances, gray_prediction_output)

        # Update the predicted flags fixing those that are equal to the threshold
        if previous_flags is not None:
            predicted_flags = tf.where(counter_flags == sequence_threshold, previous_flags, predicted_flags)
        frozen_p = len(tf.where(counter_flags == sequence_threshold))

        correct = tf.reshape(tf.gather(predicted_flags, tf.where(predicted_flags == true_flags)), [-1])
        wrong = tf.reshape(tf.gather(predicted_flags, tf.where(predicted_flags != true_flags)), [-1])
        save_gray_stats(correct, wrong, gray_stats_output, undecided=True, frozen_predictions=frozen_p)

        # Update the counters
        if previous_flags is None:
            previous_flags = predicted_flags
        counter_flags = tf.where(previous_flags == predicted_flags, counter_flags+1, 1)
        previous_flags = predicted_flags
        counter_flags = tf.where(counter_flags > sequence_threshold, sequence_threshold, counter_flags)
        print(len(tf.where(counter_flags == sequence_threshold)))

        # Regenerate the training dataset with also this new gray data
        original_training_B, original_training_C = separate_training_triplets(
                    training_data[0], training_data[2]
                )

        gray_triplets = regenerate_gray_triplets(predicted_flags, gray_samples, GrayTripletsGenerator,
                                                 original_training_B, original_training_C)

        # Return the new training dataset.
        yield merge_triplets(gray_triplets, training_triplets, batch_size)


def Gray_Selector(function, *args, **kwargs):
    if function in grays.keys():
        return grays[function](*args, **kwargs)
    else:
        raise ValueError(f"Gray enhancement \"{function}\" not available")

