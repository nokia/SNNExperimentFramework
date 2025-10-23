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
# Copyright (C) 2025 Mattia Milani <mattia.milani@nokia.com>

"""
Embeddings Actions Module
=========================

This module provides actions for loading, manipulating, and processing neural network
embeddings within the SNN2 framework. It contains utilities for:

- Loading embeddings from pickle files
- Computing centroids from embedding tensors
- Loading sample data for embedding analysis
- Converting between different tensor formats

The module is designed to work with TensorFlow tensors and PickleHandler objects,
providing essential functionality for embedding-based neural network operations
and analysis workflows.

Functions
---------
load_embeddings : function
    Load embeddings from pickle files and convert to TensorFlow tensors.
compute_centroids : function
    Compute centroid vectors from embedding tensors using mean reduction.
load_samples : function
    Load sample data from pickle files for embedding analysis.

Notes
-----
All functions in this module use the @action decorator for consistent
action tracking and logging within the SNN2 framework. The module handles
TensorFlow tensor operations with proper type conversion and shape management.

Examples
--------
Basic embedding loading and centroid computation:

>>> embeddings = load_embeddings(emb_path="path/to/embeddings.pkl", pkl=pickle_handler)
>>> centroids = compute_centroids(embeddings)
>>> samples = load_samples(samples_path="path/to/samples.pkl", pkl=pickle_handler)

See Also
--------
SNN2.src.io.pickleHandler : PickleHandler class for file operations
SNN2.src.decorators.decorators : Action decorator for function tracking
"""

from typing import List

import tensorflow as tf

from SNN2.src.decorators.decorators import action
from SNN2.src.io.pickleHandler import PickleHandler as PkH

@action
def load_embeddings(emb_path: str = None,
                    pkl: PkH = None,
                    **kwargs) -> tf.Tensor:
    """
    Load embeddings from pickle files and convert to TensorFlow tensors.

    This function loads embedding data from pickle files using the provided
    PickleHandler, converts the data to TensorFlow tensors with float32 dtype,
    and applies squeeze operation to remove unnecessary dimensions.

    Parameters
    ----------
    emb_path : str, optional
        Path to the pickle file containing the embeddings data.
        Must be a valid file path accessible by the PickleHandler.
    pkl : PickleHandler, optional
        PickleHandler instance used for loading the pickle file.
        Must be properly initialized with appropriate configuration.
    **kwargs : dict
        Additional keyword arguments passed to the PickleHandler.load() method.
        Can include loading options, compression settings, etc.

    Returns
    -------
    tf.Tensor
        TensorFlow tensor containing the loaded embeddings with float32 dtype.
        Unnecessary dimensions are removed using tf.squeeze().

    Raises
    ------
    ValueError
        If emb_path is None or if pkl (PickleHandler) is None.

    Notes
    -----
    The function automatically converts loaded data to float32 dtype for
    compatibility with TensorFlow operations. The squeeze operation removes
    dimensions of size 1, which is useful for standardizing tensor shapes.

    Examples
    --------
    >>> pkl_handler = PickleHandler(io_handler, "appendix", logger)
    >>> embeddings = load_embeddings(
    ...     emb_path="embeddings/model_embeddings.pkl",
    ...     pkl=pkl_handler
    ... )
    >>> print(embeddings.shape)
    TensorShape([1000, 128])
    """
    if emb_path is None:
        raise ValueError("No embedding path provided")

    if pkl is None:
        raise ValueError("No PickleHandler provided")

    emb_l = pkl.load(emb_path, **kwargs)
    emb_tf = tf.convert_to_tensor(emb_l, dtype=tf.float32)
    emb_tf = tf.squeeze(emb_tf)
    return emb_tf

@action
def compute_centroids(emb_tf: tf.Tensor) -> tf.Tensor:
    """
    Compute centroid vectors from embedding tensors using mean reduction.

    This function calculates the centroid (mean) of embedding vectors along
    the first axis (typically the batch/sample dimension) and expands the
    result to maintain tensor rank consistency.

    Parameters
    ----------
    emb_tf : tf.Tensor
        Input tensor containing embedding vectors with shape (n_samples, embedding_dim).
        Each row represents an individual embedding vector.

    Returns
    -------
    tf.Tensor
        Centroid tensor with shape (1, embedding_dim) representing the mean
        of all input embedding vectors. The first dimension is expanded to
        maintain consistency with batch operations.

    Notes
    -----
    The centroid computation uses tf.reduce_mean along axis=0 to average
    across all samples while preserving the embedding dimension. The result
    is expanded using tf.expand_dims to add a batch dimension of size 1.

    This is commonly used in clustering algorithms, prototype learning,
    and similarity computations where a representative vector is needed
    for a group of embeddings.

    Examples
    --------
    >>> embeddings = tf.random.normal([100, 64])  # 100 samples, 64-dim embeddings
    >>> centroid = compute_centroids(embeddings)
    >>> print(centroid.shape)
    TensorShape([1, 64])

    >>> # Use centroid for similarity computation
    >>> similarities = tf.keras.utils.cosine_similarity(embeddings, centroid)
    """
    return tf.expand_dims(tf.reduce_mean(emb_tf, axis=0), axis=0)

@action
def load_samples(samples_path: str = None,
                 pkl: PkH = None,
                 **kwargs) -> List[tf.Tensor]:
    """
    Load sample data from pickle files for embedding analysis.

    This function loads sample data (typically embedding vectors or related
    tensors) from pickle files using the provided PickleHandler. The loaded
    data is returned as a list of TensorFlow tensors without additional
    processing.

    Parameters
    ----------
    samples_path : str, optional
        Path to the pickle file containing the sample data.
        Must be a valid file path accessible by the PickleHandler.
    pkl : PickleHandler, optional
        PickleHandler instance used for loading the pickle file.
        Must be properly initialized with appropriate configuration.
    **kwargs : dict
        Additional keyword arguments passed to the PickleHandler.load() method.
        Can include loading options, compression settings, etc.

    Returns
    -------
    List[tf.Tensor]
        List containing the loaded sample data as TensorFlow tensors.
        The structure and content depend on what was originally saved
        in the pickle file.

    Raises
    ------
    ValueError
        If samples_path is None or if pkl (PickleHandler) is None.

    Notes
    -----
    Unlike load_embeddings, this function doesn't perform automatic type
    conversion or shape manipulation. The loaded data is returned as-is,
    allowing for flexible handling of different sample data formats.

    This function is commonly used for loading reference samples, test data,
    or pre-computed embeddings that need to be compared or analyzed.

    Examples
    --------
    >>> pkl_handler = PickleHandler(io_handler, "appendix", logger)
    >>> samples = load_samples(
    ...     samples_path="samples/reference_samples.pkl",
    ...     pkl=pkl_handler
    ... )
    >>> print(f"Loaded {len(samples)} sample tensors")
    Loaded 50 sample tensors

    >>> # Access individual samples
    >>> first_sample = samples[0]
    >>> print(f"First sample shape: {first_sample.shape}")
    """
    if samples_path is None:
        raise ValueError("No samples path provided")

    if pkl is None:
        raise ValueError("No PickleHandler provided")

    samples = pkl.load(samples_path, **kwargs)
    return samples
