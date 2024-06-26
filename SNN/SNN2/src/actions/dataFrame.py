# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
import tensorflow as tf

from SNN2.src.decorators.decorators import action, f_logger, timeit
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Union, Any
from SNN2.src.io.logger import LogHandler as LH

def df_none(df: Any) -> None:
    if df is None:
        raise Exception("The dataframe passed is None")

@action
@timeit
def load(*args, **kwargs) -> pd.DataFrame:
    return pd.read_csv(*args, **kwargs)

@action
def loadMNIST(*args, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    return tf.keras.datasets.mnist.load_data()

def MNIST_unpack(mnist: Tuple[Tuple[np.ndarray, ...], ...]) -> Tuple[np.ndarray, ...]:
    objects = list(mnist)
    objects = [list(obj) for obj in objects]
    flat_ls = [item for sublist in objects for item in sublist]
    return tuple(flat_ls)

def MNIST_pack(mnist: List[np.ndarray]) -> Tuple[Tuple[Any, ...], ...]:
    i = range(0, len(mnist), 2)
    j = range(1, len(mnist), 2)
    return ((mnist[ii], mnist[jj]) for ii, jj in zip(i, j))

@action
def reshapeMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    df_none(df)
    train_images, train_labels, test_images, test_labels = MNIST_unpack(df)
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    return MNIST_pack([train_images, train_labels, test_images, test_labels])

@action
def normalizeMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    df_none(df)
    train_images, train_labels, test_images, test_labels = MNIST_unpack(df)
    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255
    return MNIST_pack([train_images, train_labels, test_images, test_labels])

@action
def trn_val_test_splitMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    df_none(df)
    train_images, train_labels, test_images, test_labels = MNIST_unpack(df)
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2, shuffle=True, random_state=1)
    return MNIST_pack([train_images, train_labels, validation_images, validation_labels, test_images, test_labels])

def generateImgTriplets(images, labels) -> np.ndarray:
    classes = 10
    same_class = [np.where(labels==i)[0] for i in range(classes)]
    different_class = [np.where(labels!=i)[0] for i in range(classes)]

    positive_idx = np.array([np.random.choice(same_class[l]) for l in labels])
    negative_idx = np.array([np.random.choice(different_class[l]) for l in labels])

    positive_imgs = np.take(images, positive_idx, axis=0)
    negative_imgs = np.take(images, negative_idx, axis=0)
    result = np.stack([positive_imgs, images, negative_imgs], axis=1)
    return result

def generateImgUnrelatedGroups(images, labels) -> np.ndarray:
    classes = 10
    class_indexes = [np.where(labels==i)[0] for i in range(classes)]

    matrix_indexes = np.array([
                np.array([np.random.choice(class_indexes[i]) for l in labels])
                for i in range(classes)])

    matrix_images = np.take(images, matrix_indexes, axis=0)
    # result = np.stack([images, matrix_images], axis=1)
    images = np.expand_dims(images, 0)
    result = np.concatenate((matrix_images, images), 0)
    return result

@action
def generateTripletsMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    df_none(df)
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = MNIST_unpack(df)
    train_triplet_imgs = generateImgTriplets(train_images, train_labels)
    validation_triplet_imgs = generateImgTriplets(validation_images, validation_labels)
    test_triplet_imgs = generateImgTriplets(test_images, test_labels)
    return MNIST_pack([train_triplet_imgs, train_labels, validation_triplet_imgs, validation_labels, test_triplet_imgs, test_labels])

@action
def generateUnrelatedGroupsMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    df_none(df)
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = MNIST_unpack(df)
    train_triplet_imgs = generateImgUnrelatedGroups(train_images, train_labels)
    validation_triplet_imgs = generateImgUnrelatedGroups(validation_images, validation_labels)
    test_triplet_imgs = generateImgUnrelatedGroups(test_images, test_labels)
    return MNIST_pack([train_triplet_imgs, train_labels, validation_triplet_imgs, validation_labels, test_triplet_imgs, test_labels])

def generateExpectationMatrix(labels) -> np.ndarray:
    classes = 10
    result = np.full((len(labels), classes), -1, dtype=np.int8)
    result[range(result.shape[0]), labels] = 1
    print(labels[:10])
    print(result[:10, :])
    raise Exception
    return result

@action
def generateExpectationMatrixMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[np.ndarray, ...], ...]:
    df_none(df)
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = MNIST_unpack(df)
    train_triplet_imgs = generateExpectationMatrix(train_labels)
    validation_triplet_imgs = generateExpectationMatrix(validation_labels)
    test_triplet_imgs = generateExpectationMatrix(test_labels)
    return MNIST_pack([train_triplet_imgs, train_labels, validation_triplet_imgs, validation_labels, test_triplet_imgs, test_labels])

@action
def toTfDatasetsMNIST(*args, df: Tuple[Tuple[np.ndarray, ...], ...]=None, **kwargs) -> Tuple[Tuple[tf.data.Dataset, tf.Tensor], ...]:
    df_none(df)
    train_images, train_labels, validation_images, validation_labels, test_images, test_labels = MNIST_unpack(df)
    train_images = tf.data.Dataset.from_tensor_slices((train_images[:, 0], train_images[:, 1], train_images[:, 2]))
    train_labels = tf.convert_to_tensor(train_labels)
    validation_images = tf.data.Dataset.from_tensor_slices((validation_images[:, 0], validation_images[:, 1], validation_images[:, 2]))
    validation_labels = tf.convert_to_tensor(validation_labels)
    test_images = tf.data.Dataset.from_tensor_slices((test_images[:, 0], test_images[:, 1], test_images[:, 2]))
    test_labels = tf.convert_to_tensor(test_labels)
    return MNIST_pack([train_images, train_labels, validation_images, validation_labels, test_images, test_labels])

@action
def write(*args, df: pd.DataFrame = None, **kwargs) -> None:
    df_none(df)
    df.to_csv(*args, **kwargs)

@action
def dropColumns(*args, df: pd.DataFrame = None, axis=1, **kwargs) -> Union[pd.DataFrame, None]:
    df_none(df)
    return df.drop(*args, axis=axis, **kwargs)

@action
def dropOutliers(*args,
                 df: pd.DataFrame = None,
                 threshold: float = 90.0,
                 **kwargs) -> Union[pd.DataFrame, None]:
    df_none(df)
    tmp_df = df[df["problem"] == "good"].copy()
    vmafs = tmp_df.groupby(["exp_id"])["vmaf"].mean().reset_index()
    outliers = vmafs[vmafs["vmaf"] < threshold]["exp_id"].values
    return df[~df["exp_id"].isin(outliers)]

@action
def removeNaN(*args,
              df: pd.DataFrame = None,
              **kwargs) -> Union[pd.DataFrame, None]:
    df_none(df)
    exp_ids = df[df.isnull().any(axis=1)]["exp_id"].values
    return df[~df["exp_id"].isin(exp_ids)]

@action
def remove_over_threshold(*args,
                          df: Optional[pd.DataFrame] = None,
                          column: Optional[str] = None,
                          threshold: Optional[float] = None,
                          **kwargs) -> pd.DataFrame:
    df_none(df)
    df_none(column)
    df_none(threshold)
    over_exp_ids = df[df[column] > threshold]["exp_id"].unique()
    return df[~df["exp_id"].isin(over_exp_ids)]

