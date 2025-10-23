First Example
=============

This section provides a first example of how to run experiments with the SNN Framework using a custom configuration and datasets.

This example demonstrates how to run a Siamese Neural Network experiment for video quality
prediction using the NetKPI-QoS dataset.
The experiment focuses on predicting VMAF (Video Multi-method Assessment Fusion) scores based on
network performance indicators.

Dataset Overview
~~~~~~~~~~~~~~~~

The NetKPI-QoS dataset contains network performance metrics and corresponding video quality measurements:

* **Location**: `Data/NetKPI-QoS/scarlet.csv`
* **Features**:

  - `packet_drop_rate`: Packet loss percentage
  - `byte_drop_rate`: Byte loss percentage
  - `avg_timeDelta`: Average inter-packet time
  - `std_timeDelta`: Standard deviation of inter-packet time
  - `skw_timeDelta`: Skewness of inter-packet time distribution
  - `kur_timeDelta`: Kurtosis of inter-packet time distribution
  - `vmaf`: Target VMAF video quality score (0-100)

* **Problem Type**: Video quality regression with network-based features

Configuration Details
~~~~~~~~~~~~~~~~~~~~~~

The NetKPI-QoS configuration is located in `conf/NetKPI-QoS/` and includes:

**Model Configuration** (`model.ini`):

* **Architecture**: Siamese Neural Network with LSTM layers
* **Input Features**: 3 key network metrics (`packet_drop_rate`, `skw_timeDelta`, `kur_timeDelta`)
* **Window Size**: 9 temporal steps
* **Network Structure**:

  - LSTM layer (128 nodes)
  - Flatten layer
  - Dense layer (128 nodes, ReLU activation)
  - Dense layer (64 nodes, ReLU activation)
  - Output layer (4 nodes)

* **Optimizer**: Adam with learning rate 0.0001

**Preprocessing Configuration** (`pp.ini`):

* **Data Pipeline**:

  1. Load raw data
  2. Drop outliers
  3. Remove NaN values
  4. Good/Bad/Gray quality separation
  5. Duration-based windowing

* **Data Splits**:

  - Training: 99% good, 1% bad quality samples
  - Validation: 35% good, 30% gray, 35% bad
  - Test: 35% good, 30% gray, 35% bad

* **Window Processing**: 9-step temporal windows with striding overlap

**Training Configuration** (`experiment.ini`):

* **Epochs**: 10
* **Batch Size**: 10
* **Callbacks**: CSV logging, model checkpointing
* **Verbosity**: Standard training output

**IO configuration** (`io.ini`):

* **Main folder**: Substitute with the local full path
* **Results Directory**: `results-test/NetKPI-QoS/`
* **Dataset path and file**: Can be configured using the respective parameters
* **Logging file**: Name of the lof file, it can be modified with a specific action to produce one log file at each iteration


Running the Experiment
~~~~~~~~~~~~~~~~~~~~~~~

**New training cycle**:

.. code-block:: bash

   # Navigate to SNN directory
   cd SNN

   # Run NetKPI-QoS experiment
   snn -c conf/NetKPI-QoS

**With Verbose logging**:

.. code-block:: bash

   # Enable detailed logging
   snn -c conf/NetKPI-QoS -vvv

Expected Output
~~~~~~~~~~~~~~~

The outcome of the experiment should be contained in the `results_path` configured
in `io.ini`.
The folder structure for the outcome is the following:

.. code-block:: text

   results-test/
   └── NetKPI-QoS/
       ├── ckpt/
       │   └── Directory used to store model checkpoints
       ├── csv/
       │   └── Directory used to store CSV files generated during the experiment
       ├── log/
       │   └── Logs folder
       ├── pkl/
       │   └── Directory used to store pickle files generated during the experiment
       ├── plot/
       │   └── Directory used to store plots generated during the experiment, this generally is deprecated, is preferable to use other tools to generate plots from the CSV files
       ├── CommonReplayMem/
       │   └── Directory used to store common replay memory files for Reinforcement Learning experiments
       └── tensorboard/
           └── Directory used by tensorboard to store event files, this feature is still under development

Customization Options
~~~~~~~~~~~~~~~~~~~~~~

**Modify Model Architecture**:

Edit `conf/NetKPI-QoS/model.ini` and `conf/NetKPI-QoS/layers.ini` to change:

.. code-block:: ini

   # Increase LSTM complexity
   [n_nodes]
   value=256

   # Add more layers
   [layers]
   value=['lstm', 'flatten', 'dense', 'dense2', 'dense3', 'dense4']

   [dense4]
   type=layer
   value=dense
   args='8'
   kwargs='activation':'${activation:value}','input_shape':'${shape:value}'

**Adjust Data Processing**:

Modify `conf/NetKPI-QoS/pp.ini` for different data splits:

.. code-block:: ini

   # More balanced training data
   [training_good_percentage]
   value=0.7

   [training_bad_percentage]
   value=0.3

**Experiment Parameters**:

Update `conf/NetKPI-QoS/experiment.ini` for longer training:

.. code-block:: ini

   # Extended training
   [epochs]
   value=50

   # Larger batches
   [batch_size]
   value=32