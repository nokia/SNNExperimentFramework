Configuration Settings
======================

This page documents all the default configuration values available in the SNN2 framework. These configurations are organized into multiple INI files located in the ``SNN2/default_ini/`` directory.

Overview
--------

The SNN2 framework uses a hierarchical configuration system with parameter interpolation. Configuration values can reference other values using the ``{parameter_name}`` syntax, allowing for flexible and reusable configurations.

Core Configuration (default.ini)
--------------------------------

Basic folder structure and timing settings:

**Folder Structure:**
    - ``pkg_folder``: Path to the package folder
    - ``main_folder``: Main working directory
    - ``uxtime_sec``: Unix timestamp in seconds for unique run identification

Model Configuration (model.ini)
-------------------------------

Neural network model parameters:

**Model Architecture:**
    - ``weights_file``: Path to pre-trained model weights
    - ``checkpoint_file``: Path for saving model checkpoints
    - ``model``: Model architecture specification
    - ``loss``: Loss function configuration
    - ``metrics``: List of metrics to track during training
    - ``optimizer``: Optimizer settings

**Training Parameters:**
    - ``epochs``: Number of training epochs
    - ``batch_size``: Training batch size
    - ``learning_rate``: Learning rate for optimization
    - ``validation_split``: Fraction of data for validation

Input/Output Configuration (io.ini)
-----------------------------------

Paths and file management settings:

**Output Paths:**
    - ``results_path``: Directory for storing results
    - ``checkpoints_path``: Directory for model checkpoints
    - ``logs_path``: Directory for training logs
    - ``tensorboard_path``: TensorBoard logging directory
    - ``csv_net_evolution_file``: CSV file for network evolution tracking

**Data Paths:**
    - ``dataset``: Primary dataset path
    - ``test_dataset``: Test dataset path
    - ``validation_dataset``: Validation dataset path

Reinforcement Learning Configuration (reinforcement_model.ini)
------------------------------------------------------------

Reinforcement learning specific parameters:

**RL Algorithm Settings:**
    - ``gamma``: Discount factor (default: 0.99)
    - ``entropy_beta``: Entropy regularization coefficient (default: 0.01)
    - ``learning_rate_rl``: Learning rate for RL components
    - ``replay_buffer_size``: Size of experience replay buffer

**Environment Settings:**
    - ``action_space_size``: Number of possible actions
    - ``state_space_size``: Dimensionality of state representation
    - ``reward_threshold``: Threshold for episode success
    - ``max_episode_length``: Maximum steps per episode

Data Flow Configuration (flow.ini)
----------------------------------

Data processing and flow management:

**Flow Types:**
    - ``defaultFlow``: Basic data flow configuration
    - ``NetCatFlow``: Network traffic data flow
    - ``VMAFFlow``: Video quality assessment flow
    - ``MVNOFlow``: Mobile network operator data flow

**Flow Parameters:**
    - ``NetCat_columns``: Column specification for NetCat data
    - ``NetCat_dataset``: Dataset path for NetCat flow
    - ``VMAF_columns``: Column specification for VMAF data
    - ``VMAF_dataset``: Dataset path for VMAF flow
    - ``vmaf_threshold``: Quality threshold for VMAF processing
    - ``window``: Window size for data processing

**Extended Flows:**
    - ``MVNOFlow_inference``: Inference-specific MVNO flow
    - ``MVNOFlow_smartMemory``: Memory-optimized MVNO flow
    - ``MVNOFlow_extended``: Extended MVNO flow with additional features

**Flow Control:**
    - ``inference_dst_name``: Destination name for inference results
    - ``extra_dataset_base_name``: Base name for additional datasets
    - ``use_random_samples``: Flag for random sampling
    - ``gray_post_train_portion``: Portion of data for post-training
    - ``gray_in_train_portion``: Portion of data within training

Callbacks Configuration (callbacks.ini)
---------------------------------------

Training callbacks and monitoring:

**Logging Callbacks:**
    - ``csvLogger``: CSV logging for training metrics
    - ``tensorBoard``: TensorBoard visualization
    - ``saveEmbeddings``: Save learned embeddings
    - ``saveObject``: Save model objects

**Training Control:**
    - ``earlyStopping``: Early stopping configuration
        - ``patience``: 20 epochs
        - ``min_delta``: 0.001 minimum improvement
    - ``modelCheckpoint``: Model checkpoint saving

**Testing and Validation:**
    - ``testParam``: Testing parameter callback
        - ``testParamEpochThreshold``: 10 epochs
        - ``testParamEndTrainFlag``: False
        - ``testParamDelta``: 10.0
    - ``activate_metric``: Metric activation
    - ``set_metric_attr``: Metric attribute setting

**Reinforcement Learning Callbacks:**
    - ``reinforcement``: Main RL callback
        - ``reinforceCBThresholdUnit``: 'ep' (epochs)
        - ``reinforceCBThreshold``: 10
        - ``reinforceCBInitialTrainingEpochs``: 1
    - ``FastReplayMemGenerator``: Fast replay memory generation
    - ``FakeReinforceFixedAction``: Fixed action RL testing
        - ``fixedAction``: 0
    - ``RL_manager``: RL environment manager
        - ``RL_initial_skip``: 0
    - ``RL_partial_manager``: Partial RL environment manager

**Accuracy Callbacks:**
    - ``ott_accuracy``: Over-the-top accuracy measurement
        - ``ottAcc_fxdM_flag``: True
        - ``ottAcc_fxdM``: 0.5
    - ``mno_accuracy``: Mobile network operator accuracy
        - ``mnoAcc_fxdM_flag``: True
        - ``mnoAcc_fxdM``: 0.5

**Advanced Callbacks:**
    - ``controlledMargin``: Margin control callback
    - ``controlledCrossEntropy``: Cross-entropy control
        - ``categorical_threshold``: 1
        - ``categorical_value``: -1.0
        - ``categorical_triangular_flag``: True
        - ``categorical_dario_norm``: False
    - ``ACCCE``: Advanced Controlled Cross Entropy

Additional Configuration Files
-----------------------------

The SNN2 framework includes additional configuration files for specific components:

- ``actions.ini``: Action space definitions for reinforcement learning
- ``optimizers.ini``: Optimizer configurations (Adam, SGD, RMSprop, etc.)
- ``losses.ini``: Loss function definitions
- ``metrics.ini``: Custom metrics and evaluation functions
- ``datasets.ini``: Dataset loading and preprocessing configurations
- ``preprocessors.ini``: Data preprocessing pipeline configurations
- ``postprocessors.ini``: Post-processing and result formatting
- ``rewards.ini``: Reward function definitions for RL
- ``cure_algorithm.ini``: CURE clustering algorithm parameters
- ``models.ini``: Additional model architecture definitions

Configuration Usage
-------------------

To use these configurations in your experiments:

1. **Default Values**: All parameters have sensible defaults defined in the INI files
2. **Override Values**: You can override any parameter by specifying it in your experiment configuration
3. **Parameter Interpolation**: Use ``{parameter_name}`` to reference other configuration values
4. **Environment Variables**: Some parameters can be set via environment variables

Example configuration override::

    [model]
    epochs = 100
    batch_size = 64
    learning_rate = 0.001

    [reinforcement_model]
    gamma = 0.95
    entropy_beta = 0.02

For more details on specific parameters and their effects, refer to the individual configuration files in the ``SNN2/default_ini/`` directory.