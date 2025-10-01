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

import numpy as np
import pandas as pd
import tensorflow as tf

from SNN2.src.decorators.decorators import action, f_logger, timeit
from sklearn.model_selection import train_test_split
from typing import List, Optional, Tuple, Union, Any, Dict, Callable
from SNN2.src.io.logger import LogHandler as LH
from scipy.signal import welch
from scipy.stats import entropy

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
def dropNaN(*args, df: pd.DataFrame = None, **kwargs) -> Union[pd.DataFrame, None]:
    df_none(df)
    return df.dropna(*args, **kwargs)

@action
def keepColumns(df: pd.DataFrame = None, columns: List[str]=[]) -> Union[pd.DataFrame, None]:
    df_none(df)
    return df[columns]

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


def rolling_feature(df: pd.DataFrame,
                    feature:str,
                    frm_col:list,
                    *args,
                    function: Callable = None,
                    group_id: str = 'id',
                    **kwargs) -> pd.DataFrame:
    """
    Compute rolling feature for each id in the DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to compute the feature on
    feature : str
        The name of the feature to compute
    frm_col : list
        The list of columns to compute the feature on
    *args : list
        Additional arguments to pass to the rolling function
    **kwargs : dict
        Additional keyword arguments to pass to the rolling function

    Returns
    -------
    pd.DataFrame
        The DataFrame with the new feature columns added
    """
    if function is None:
        function = globals()[feature]
    # Copy the dataframe in a new instance
    new_df = df.copy()
    new_col = [f"roll_{feature}_{col}" for col in frm_col]
    new_df[new_col] = new_df.groupby(group_id).rolling(*args, **kwargs)[frm_col].apply(
        function,
        engine='cython',
        raw=False
    ).values
    if feature == "entropy":
        new_df[new_col] = new_df[new_col].replace(np.nan, 0)
    return new_df

@action
def compute_roll_features(*args,
                          df: Optional[pd.DataFrame] = None,
                          operations: Optional[Dict[str, List[str]]] = None,
                          window_size: Optional[int] = 120,
                          time_col: Optional[str] = 'timestamp',
                          groupby_col: Optional[str] = 'op_id') -> pd.DataFrame:
    df_none(df)
    if operations is None:
        return df
    for key, value in operations.items():
        if key == "medcross":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.sum((np.diff(np.sign(np.array(x) - np.median(np.array(x)))) != 0).astype(int)),
                                group_id = groupby_col, on=time_col)
        if key == "spec_entr":
            def spec_entropy(x):
                freqs, psd = welch(np.array(x), fs=1.0, nperseg=window_size)
                return np.trapz(psd, freqs)
            df = rolling_feature(df, key, value, window_size,
                                function = spec_entropy,
                                group_id = groupby_col, on=time_col)
        if key == "mean":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.mean(np.array(x)),
                                group_id = groupby_col, on=time_col)
        if key == "std":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.std(np.array(x)),
                                group_id = groupby_col, on=time_col)
        if key == "entropy":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.clip(entropy(x), 0.0, 1.0),
                                group_id = groupby_col, on=time_col)
        if key == "quan95":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.quantile(np.array(x), 0.95),
                                group_id = groupby_col, on=time_col)
        if key == "quan99":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.quantile(np.array(x), 0.99),
                                group_id = groupby_col, on=time_col)
        if key == "tot_connline":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.sum(np.abs(np.diff(np.array(x)))),
                                group_id = groupby_col, on=time_col)
        if key == "tot_energy":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.sum(np.square(np.array(x))),
                                group_id = groupby_col, on=time_col)
        if key == "max":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.max(np.array(x)),
                                group_id = groupby_col, on=time_col)
        if key == "meddiffmin":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.median(x) - np.min(x),
                                group_id = groupby_col, on=time_col)
        if key == "meddiffmax":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.median(np.array(x)) - np.max(np.array(x)),
                                group_id = groupby_col, on=time_col)
        if key == "medquan90":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.median(np.array(x)) - np.quantile(np.array(x), 0.90),
                                group_id = groupby_col, on=time_col)
        if key == "median":
            df = rolling_feature(df, key, value, window_size,
                                function = lambda x: np.median(np.array(x)),
                                group_id = groupby_col, on=time_col)
    return df

@action
def compute_lag_features(*args,
                          df: Optional[pd.DataFrame] = None,
                          lag_col: List[str] = None,
                          window_size: Optional[int] = 120,
                          time_col: Optional[str] = 'timestamp',
                          groupby_col: Optional[str] = 'op_id') -> pd.DataFrame:
    df_none(df)
    df_none(lag_col)
    new_df = df.copy()
    new_col = [f"lag_{window_size}_{col}" for col in lag_col]
    new_df[new_col] = new_df.groupby(groupby_col).shift(window_size)[lag_col].values
    return new_df

@action
def compute_dayofweek(*args,
                       df: Optional[pd.DataFrame] = None,
                       in_col: str = None,
                       out_col: str = None) -> pd.DataFrame:
    df_none(df)
    df_none(in_col)
    df_none(out_col)
    df[out_col] = df[in_col].dt.dayofweek
    return df

@action
def add_window_column(*args,
                      df: Optional[pd.DataFrame] = None,
                      window_size: int = 120,
                      time_col: str = 'timestamp',
                      groupby_col: Optional[str] = 'op_id') -> pd.DataFrame:
    df_none(df)
    tmp_df = df.copy()
    # For each group defined by groupby_col add a window column that
    # starts from 1 and increments of 1 unity every window_size rows
    tmp_df['window'] = tmp_df.groupby(groupby_col).cumcount() // window_size + 1
    return tmp_df

@action
def group_standardize(*args,
                      df: Optional[pd.DataFrame] = None,
                      standardize_columns: List[str] = ['n_ul'],
                      groupby_col: Optional[str] = 'op_id') -> pd.DataFrame:
    df_none(df)
    normalized = df.copy()
    groups = normalized.groupby(groupby_col)[standardize_columns]
    mean = groups.transform("mean").values
    std = groups.transform("std").values
    std_zero_idx = np.where(std[0] == 0)[0]
    print(std_zero_idx)
    std_zero_clm = [standardize_columns[i] for i in std_zero_idx]
    mean_zero_clm = mean[:, std_zero_idx]
    del standardize_columns[std_zero_idx]
    del std[:, std_zero_idx]
    del mean[:, std_zero_idx]

    normalized[standardize_columns] = (normalized[standardize_columns].values - mean) / std
    normalized[std_zero_clm] = normalized[std_zero_clm].values - mean_zero_clm
    return normalized
