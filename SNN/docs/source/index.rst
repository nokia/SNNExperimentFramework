SNN FrameWork Documentation
=============================

Welcome to the **SNN (Siamese Neural Network) FrameWork** - a comprehensive,
highly customizable framework for experimenting with Siamese Neural Networks and
Reinforcement Learning approaches to network data analysis.

.. image:: https://img.shields.io/badge/Python-3.7+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/TensorFlow-2.x-orange.svg
   :target: https://tensorflow.org/
   :alt: TensorFlow Version

.. image:: https://img.shields.io/badge/License-BSD--3--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License: BSD-3-Clause


Overview
--------

The SNN FrameWork is designed for researchers and practitioners working with
network traffic analysis, video quality prediction, and machine learning on
time-series data. It provides a modular architecture that supports both
traditional supervised learning and reinforcement learning approaches.

**Key Features:**

* **Siamese Neural Networks**: Specialized architecture for similarity learning and metric learning tasks
* **Reinforcement Learning Integration**: Actor-Critic models for adaptive decision making
* **Video Quality Assessment**: VMAF-based quality prediction and optimization
* **Modular Design**: Easily extensible with custom models, callbacks, and data processing pipelines
* **Configuration-Driven**: INI-based configuration system for reproducible experiments
* **Comprehensive Logging**: Detailed experiment tracking and result analysis


Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone <repository-url>
   cd SNN2

   # Install dependencies
   pip install -r requirements.txt

   # Install the package
   pip install -e .

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Run with default configuration
   snn -c config/basic

   # Enable reinforcement learning
   snn -c config/basic --reinforcement

   # Run inference only
   snn -c config/basic --inference "model_label"

   # Enable verbose logging
   snn -c config/basic -vvv


Architecture Overview
---------------------

The framework is organized into several key components:

Core Components
~~~~~~~~~~~~~~~

* **Data Processing** (`src/core/data/`): Handles data preprocessing, windowing, and separation
* **Models** (`src/model/`): Siamese networks, custom architectures, and reinforcement learning models
* **Experiments** (`src/core/experiment.py`): Orchestrates training, evaluation, and inference
* **Actions** (`src/actions/`): Data transformation and separation strategies
* **I/O System** (`src/io/`): Configuration management, logging, and file handling

Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~

* **Actor-Critic Models**: Policy gradient methods for adaptive learning
* **Reward Functions**: Customizable reward systems for different objectives
* **Action Policies**: Various exploration and exploitation strategies
* **Environment Management**: Configurable RL environments for different scenarios


Configuration System
--------------------

The framework uses a hierarchical INI-based configuration system with the following structure:

* **Default Configuration**: Base settings in `default_ini/`
* **Experiment-Specific**: Override configurations in `Conf/` directories
* **Parameter Interpolation**: Dynamic parameter substitution and inheritance

Example configuration sections:

* `model.ini`: Neural network architecture settings
* `reinforcement.ini`: RL-specific parameters (γ, β, action spaces)
* `flow.ini`: Data flow and preprocessing parameters
* `callbacks.ini`: Training callbacks and monitoring


Use Cases
----------

Video Quality Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework excels at video quality prediction and optimization tasks:

* **VMAF Prediction**: Learn to predict video quality metrics
* **Adaptive Streaming**: RL-based bitrate adaptation
* **Quality-Aware Encoding**: Optimize encoding parameters for perceptual quality

Network Traffic Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

Apply Siamese networks to network data:

* **Traffic Classification**: Identify similar network flows
* **Anomaly Detection**: Detect unusual patterns in network behavior
* **Performance Prediction**: Forecast network performance metrics

Research Applications
~~~~~~~~~~~~~~~~~~~~~

* **Metric Learning**: Learn meaningful distance functions for complex data
* **Few-Shot Learning**: Leverage Siamese architectures for limited data scenarios
* **Multi-Objective Optimization**: Balance multiple competing objectives with RL


Advanced Features
-----------------

Custom Model Development
~~~~~~~~~~~~~~~~~~~~~~~~

Extend the framework with custom models:

.. code-block:: python

   from SNN2.src.model.custom.customWrapper import CustomWrapper

   class MyCustomModel(CustomWrapper):
       def build_model(self):
           # Implement your custom architecture
           pass

Reinforcement Learning Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom reward functions and action policies:

.. code-block:: python

   from SNN2.src.model.reward.rewardWrapper import RewardWrapper

   class CustomReward(RewardWrapper):
       def compute_reward(self, state, action, next_state):
           # Implement custom reward logic
           pass


.. toctree::
   :maxdepth: 4
   :caption: API Documentation:
   :hidden:

   modules


.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   :hidden:

   installation
   arguments
   configuration
   contributing


Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Support and Contributing
========================

* **Documentation**: Complete API reference and examples
* **Issues**: Report bugs and request features on the issue tracker
* **Contributing**: See `CONTRIBUTING.md` for development guidelines
* **License**: BSD-3-Clause license

The SNN FrameWork is actively developed and maintained. We welcome contributions from the community!
