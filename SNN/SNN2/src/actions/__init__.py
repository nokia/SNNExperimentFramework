"""
Actions Package
===============

This package provides a collection of action modules for data processing,
manipulation, and analysis within the SNN2 neural network framework.

The actions package contains specialized modules for different aspects of
data processing workflows, including data frame operations, dataset separation,
windowing techniques, Kaggle dataset handling, and embedding operations.

Modules
-------
dataFrame : module
    DataFrame operations and manipulations for structured data processing.
separation : module
    Data separation and balancing utilities for train/validation/test splits
    and threshold-based categorization.
windowing : module
    Time series and sequence windowing operations for neural network input preparation.
kaggleDst : module
    Kaggle dataset specific operations and preprocessing utilities.
embeddings : module
    Embedding loading, manipulation, and centroid computation operations.

Notes
-----
All action modules in this package use the @action decorator for consistent
function tracking and logging within the SNN2 framework. The modules are
designed to work together to provide a complete data processing pipeline
for neural network training and evaluation.

Examples
--------
Import specific action modules:

>>> from SNN2.src.actions import separation
>>> from SNN2.src.actions import embeddings

Access action functions:

>>> # Use separation functions
>>> train, val, test = separation.TrnValTstSeparation(data)
>>>
>>> # Use embedding functions
>>> embeddings_tensor = embeddings.load_embeddings(path, pkl_handler)
>>> centroids = embeddings.compute_centroids(embeddings_tensor)

See Also
--------
SNN2.src.decorators.decorators : Action decorator implementation
SNN2.src.core.data : Core data management classes
SNN2.src.io : Input/output utilities and handlers
"""

import SNN2.src.actions.dataFrame
import SNN2.src.actions.separation
import SNN2.src.actions.windowing
import SNN2.src.actions.kaggleDst
import SNN2.src.actions.embeddings
