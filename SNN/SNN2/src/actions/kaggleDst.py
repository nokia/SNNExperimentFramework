# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Kaggle Dataset Actions Module
=============================

This module provides specialized actions for processing and manipulating Kaggle datasets
within the SNN2 neural network framework. It contains utilities for:

- Data separation and label extraction from Kaggle datasets
- Dataset shuffling and train/validation/test splitting
- Triplet generation for metric learning tasks
- Categorical target preparation for classification
- Post-processing operations on tensor data

The module is specifically designed to handle common Kaggle dataset formats and
workflows, providing seamless integration with the SNN2 data processing pipeline.

Functions
---------
apply_post_operations : function
    Apply a sequence of post-processing operations to tensor data.
kaggleDataVsLabelSeparation : function
    Extract and separate data and labels from Kaggle DataFrames.
kaggleShuffle : function
    Randomly shuffle dataset samples for training preparation.
kaggleTrnValTestSeparation : function
    Split Kaggle datasets into training, validation, and test sets.
generateKaggleTriplets : function
    Generate triplet datasets (anchor, positive, negative) for metric learning.
generateKaggleCategorical : function
    Prepare categorical targets and datasets for classification tasks.

Notes
-----
All functions in this module use the @action and @f_logger decorators for consistent
logging and action tracking within the SNN2 framework. The module handles pandas
DataFrames, TensorFlow tensors, and DataManager objects for comprehensive data processing.

The module includes specific handling for web service classification tasks commonly
found in Kaggle competitions, with built-in data cleaning and preprocessing steps.

Examples
--------
Basic Kaggle dataset processing workflow:

>>> # Load and process Kaggle DataFrame
>>> processed_data = kaggleDataVsLabelSeparation(df, requests=data_requests)
>>>
>>> # Shuffle and split the data
>>> kaggleShuffle(processed_data)
>>> train, val, test = kaggleTrnValTestSeparation(processed_data, trn_portion=0.7)
>>>
>>> # Generate triplets for metric learning
>>> triplet_data = generateKaggleTriplets(train, sample_name="Features")

See Also
--------
SNN2.src.core.data.DataManager : DataManager class for data handling
SNN2.src.decorators.decorators : Action and logging decorators
SNN2.src.actions.separation : General data separation utilities
"""

import math
from typing import Any, Dict, Union, Tuple

import pandas as pd
import tensorflow as tf
from SNN2.src.core.data.DataManager import DataManager

from SNN2.src.decorators.decorators import action, f_logger
from SNN2.src.io.logger import LogHandler as LH


@f_logger
def apply_post_operations(request: Dict[str, Any],
                          tf_values: tf.Tensor,
                          **kwargs) -> tf.Tensor:
    """
    Apply a sequence of post-processing operations to tensor data.

    This function sequentially applies a list of operations to a TensorFlow tensor,
    using the operation functions, arguments, and keyword arguments specified in the
    request dictionary. Each operation is applied in order with its corresponding
    arguments.

    Parameters
    ----------
    request : Dict[str, Any]
        Dictionary containing post-processing configuration with keys:
        - 'post_operation': List of callable functions to apply
        - 'post_operation_args': List of argument tuples for each operation
        - 'post_operation_kwargs': List of keyword argument dicts for each operation
    tf_values : tf.Tensor
        Input tensor to which operations will be applied sequentially.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from f_logger decorator.

    Returns
    -------
    tf.Tensor
        Transformed tensor after applying all specified post-processing operations.

    Notes
    -----
    The function iterates through the operation lists in parallel, applying each
    operation with its corresponding arguments and keyword arguments. The operations
    are applied in the order they appear in the lists.

    The function logs each operation being applied using the qualified name of the
    operation function for debugging and tracking purposes.

    Examples
    --------
    >>> request = {
    ...     'post_operation': [tf.nn.relu, tf.nn.l2_normalize],
    ...     'post_operation_args': [(), (1,)],
    ...     'post_operation_kwargs': [{}, {'epsilon': 1e-12}]
    ... }
    >>> processed_tensor = apply_post_operations(request, input_tensor)
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    for operation, o_arg, o_kwarg in zip(request["post_operation"],
                                         request["post_operation_args"],
                                         request["post_operation_kwargs"]):
        write_msg(f"Appling {operation.__qualname__}")
        tf_values = operation(tf_values, *o_arg, **o_kwarg)

    return tf_values


@action
@f_logger
def kaggleDataVsLabelSeparation(df: pd.DataFrame,
                                requests: Union[DataManager, None] = None,
                                **kwargs) -> DataManager:
    """
    Extract and separate data and labels from Kaggle DataFrames with preprocessing.

    This function processes Kaggle DataFrames by performing data cleaning operations
    (removing low-frequency classes, duplicates) and extracting specified columns
    into tensors according to the provided requests configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Input Kaggle DataFrame containing the dataset to be processed.
        Expected to have a 'web_service' column for classification tasks.
    requests : Union[DataManager, None], optional
        DataManager instance containing column extraction requests.
        Each request should specify 'columns', 'dtype', and optional 'post_operation'.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    DataManager
        Updated DataManager instance with extracted and processed tensor data.
        Each request key contains the corresponding processed tensor data.

    Raises
    ------
    Exception
        If requests parameter is None.

    Notes
    -----
    The function performs several preprocessing steps:
    1. Removes web services with fewer than 100 occurrences
    2. Removes duplicate rows
    3. Logs data quality statistics (missing values, duplicates)
    4. Extracts specified columns and converts to tensors
    5. Applies post-processing operations if specified
    6. Concatenates multiple tensor batches if needed

    Examples
    --------
    >>> requests = DataManager()
    >>> requests['features'] = {
    ...     'columns': ['feature1', 'feature2'],
    ...     'dtype': tf.float32,
    ...     'post_operation': None
    ... }
    >>> processed_data = kaggleDataVsLabelSeparation(kaggle_df, requests=requests)
    """
    if requests is None:
        raise AttributeError("No requests provided")

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
def kaggleShuffle(data: DataManager,
                  **kwargs) -> None:
    """
    Randomly shuffle dataset samples for training preparation.

    This function performs in-place shuffling of all samples in the DataManager
    by generating random indices and reordering the data accordingly. This is
    commonly used before training to ensure random sample ordering.

    Parameters
    ----------
    data : DataManager
        DataManager instance containing the dataset to be shuffled.
        Must have a 'Samples' key with tensor data.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    None
        Function modifies the input DataManager in-place.

    Notes
    -----
    The function uses tf.random.shuffle to generate random indices based on the
    number of samples in the 'Samples' tensor, then applies these indices to
    reorder all data in the DataManager using sub_select with inplace=True.

    This ensures that all related data (samples, targets, etc.) are shuffled
    consistently while maintaining their relationships.

    Examples
    --------
    >>> # Shuffle training data before model training
    >>> kaggleShuffle(training_data)
    >>> # Data is now randomly ordered
    """
    _, _= kwargs["logger"], kwargs["write_msg"]

    features = data.dft("Samples")
    ids = tf.range(features.shape[0])
    ids = tf.random.shuffle(ids)

    data.sub_select(ids, inplace=True)


@action
@f_logger
def kaggleTrnValTestSeparation(data: DataManager,
                               trn_portion: float = 0.6,
                               val_portion: float = 0.2,
                               **kwargs) -> Tuple[DataManager, DataManager, DataManager]:
    """
    Split Kaggle datasets into training, validation, and test sets.

    This function divides a DataManager instance into three separate datasets
    for training, validation, and testing based on the specified proportions.
    The remaining portion (1 - trn_portion - val_portion) is used for testing.

    Parameters
    ----------
    data : DataManager
        DataManager instance containing the complete dataset to be split.
        Must have a 'Samples' key with tensor data.
    trn_portion : float, default=0.6
        Fraction of data to use for training (0.0 to 1.0).
    val_portion : float, default=0.2
        Fraction of data to use for validation (0.0 to 1.0).
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    Tuple[DataManager, DataManager, DataManager]
        A tuple containing three DataManager instances:
        - trn_data: Training dataset
        - val_data: Validation dataset
        - tst_data: Test dataset

    Notes
    -----
    The function uses sequential index slicing (not random) to create the splits:
    - Training: indices [0 : train_n]
    - Validation: indices [train_n : train_n + validation_n]
    - Test: indices [train_n + validation_n : end]

    For random splits, use kaggleShuffle before calling this function.
    The test portion is automatically calculated as the remainder.

    Examples
    --------
    >>> # Split with custom proportions (70% train, 20% val, 10% test)
    >>> train, val, test = kaggleTrnValTestSeparation(
    ...     data, trn_portion=0.7, val_portion=0.2
    ... )
    >>>
    >>> # For random splits, shuffle first
    >>> kaggleShuffle(data)
    >>> train, val, test = kaggleTrnValTestSeparation(data)
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

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
                           sample_name: str = "Samples",
                           target_name: str = "Targets",
                           **kwargs) -> DataManager:
    """
    Generate triplet datasets (anchor, positive, negative) for metric learning.

    This function creates triplet datasets suitable for metric learning by generating
    positive and negative sample pairs for each anchor sample based on target labels.
    For each unique target class, positive samples are from the same class and
    negative samples are from different classes.

    Parameters
    ----------
    anchor : DataManager
        DataManager instance containing the base dataset from which triplets are generated.
    sample_name : str, default="Samples"
        Key name for sample data in the DataManager.
    target_name : str, default="Targets"
        Key name for target/label data in the DataManager.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    DataManager
        Updated DataManager instance with triplet data in 'TripletDst' key containing:
        - Stacked tensor: [positive, anchor, negative] samples
        - TfDataset: TensorFlow dataset with (positive, anchor, negative) tuples

    Notes
    -----
    The function handles class imbalance by repeating samples as needed:
    - For each target class, positive/negative indices are repeated to match dataset size
    - Uses tf.repeat with calculated repetitions to ensure sufficient samples
    - Applies random shuffling to positive/negative indices for variety

    The resulting triplets maintain the property that:
    - Anchor and positive samples have the same target label
    - Anchor and negative samples have different target labels

    Examples
    --------
    >>> # Generate triplets for metric learning
    >>> triplet_data = generateKaggleTriplets(
    ...     training_data,
    ...     sample_name="Features",
    ...     target_name="Labels"
    ... )
    >>>
    >>> # Access triplet dataset
    >>> triplet_dataset = triplet_data['TripletDst']['TfDataset']
    >>> for pos, anc, neg in triplet_dataset.take(1):
    ...     print(f"Shapes: pos={pos.shape}, anc={anc.shape}, neg={neg.shape}")
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

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
        positive_idx = tf.repeat(tf.random.shuffle(positive_idx),
                                 repeats=reps, axis=0)[:targets.shape[0]]
        reps = math.ceil(len(idx)/len(negative_idx))
        negative_idx = tf.repeat(tf.random.shuffle(negative_idx),
                                 repeats=reps, axis=0)[:targets.shape[0]]

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
                              sample_name: str = "Samples",
                              target_name: str = "Targets",
                              dst_name: str = "SamplesDst",
                              num_classes: int = 141,
                              **kwargs) -> DataManager:
    """
    Prepare categorical targets and datasets for classification tasks.

    This function converts integer target labels to one-hot categorical encoding
    and prepares the data for classification training by creating appropriate
    TensorFlow datasets with samples and categorical targets.

    Parameters
    ----------
    anchor : DataManager
        DataManager instance containing the dataset to be prepared for classification.
    sample_name : str, default="Samples"
        Key name for sample data in the DataManager.
    target_name : str, default="Targets"
        Key name for target/label data in the DataManager.
    dst_name : str, default="SamplesDst"
        Key name for the output dataset in the DataManager.
    num_classes : int, default=141
        Number of classes for categorical encoding. Should match the maximum
        class index + 1 in the target data.
    **kwargs : dict
        Additional keyword arguments containing logger and write_msg from decorators.

    Returns
    -------
    DataManager
        Updated DataManager instance with categorical classification data in dst_name key
        containing:
        - Sample data copy
        - TfDataset: TensorFlow dataset with (samples, categorical_targets) pairs
        - CategoricalTargets: One-hot encoded target vectors

    Notes
    -----
    The function uses keras.utils.to_categorical to convert integer labels to
    one-hot encoded vectors. The num_classes parameter should be set to accommodate
    all possible class indices in the dataset.

    The resulting dataset is suitable for training classification models with
    categorical crossentropy loss functions.

    Examples
    --------
    >>> # Prepare data for 141-class classification
    >>> categorical_data = generateKaggleCategorical(
    ...     training_data,
    ...     sample_name="Features",
    ...     target_name="Labels",
    ...     dst_name="ClassificationDst",
    ...     num_classes=141
    ... )
    >>>
    >>> # Access categorical dataset
    >>> dataset = categorical_data['ClassificationDst']['TfDataset']
    >>> for samples, targets in dataset.take(1):
    ...     print(f"Sample shape: {samples.shape}, Target shape: {targets.shape}")
    """
    _, write_msg = kwargs["logger"], kwargs["write_msg"]

    idx = tf.range(anchor.dft(sample_name).shape[0])
    write_msg(f"Total idx: {len(idx)}", level=LH.DEBUG)

    targets = anchor.dft(target_name)
    categorical_targets = tf.keras.utils.to_categorical(targets, num_classes)
    write_msg(f"Categorical targets: {categorical_targets}")

    samples = anchor.sub_select(idx)
    samples[dst_name] = None
    samples[dst_name].set_default(samples.dft(sample_name))
    samples[dst_name]["TfDataset"] = tf.data.Dataset.from_tensor_slices((samples.dft(sample_name),
                                                                         categorical_targets))
    samples[dst_name]["CategoricalTargets"] = categorical_targets

    return samples
