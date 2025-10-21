Command Line Arguments
======================

The SNN2 framework provides a comprehensive command-line interface that allows you to configure and control various aspects of the neural network training and inference process. This page documents all available command-line arguments and their usage.

Usage
-----

.. code-block:: bash

   python main.py [options]

Basic Syntax
~~~~~~~~~~~~

.. code-block:: bash

   # Basic usage with default configuration
   python main.py -c config/basic

   # Full example with multiple options
   python main.py -c config/custom --reinforcement --study -vvv --debug


Configuration and Basic Options
--------------------------------

-c, --confFolder
~~~~~~~~~~~~~~~~

**Type:** string
**Default:** ``config/basic``

Defines the input configuration folder where INI files are located. The framework uses a hierarchical configuration system where custom configurations override default settings.

.. code-block:: bash

   # Use custom configuration
   python main.py -c config/my_experiment

   # Use relative path
   python main.py -c ../experiments/video_quality

-v, --verbose
~~~~~~~~~~~~~

**Type:** count (can be repeated)
**Default:** 1

Controls the verbosity level of logging output. Each additional ``-v`` increases the detail level:
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

* ``-v``: Critical logging (CRITICAL level)
* ``-vv``: Error logging (ERROR level)
* ``-vvv``: Warning logging (WARNING level)
* ``-vvvv``: Info logging (INFO level) - default level
* ``-vvvvv``: Debug logging (DEBUG level)

.. code-block:: bash

   # Standard verbosity
   python main.py -c config/basic -v

   # Maximum verbosity for debugging
   python main.py -c config/basic -vvv

-r, --redirect
~~~~~~~~~~~~~~

**Type:** string
**Default:** None

Redirects stdout to a specified file. Useful for capturing output during long training runs or batch processing.

.. code-block:: bash

   # Redirect output to log file
   python main.py -c config/basic -r experiment_output.log

   # Redirect to timestamped file
   python main.py -c config/basic -r "training_$(date +%Y%m%d_%H%M%S).log"


Debug and Development Options
-----------------------------

-D, --debug
~~~~~~~~~~~

**Type:** flag
**Default:** False

Disables model saving functionality. Useful during development and testing when you don't want to persist model checkpoints or final models.

.. code-block:: bash

   # Run without saving models
   python main.py -c config/basic --debug

   # Combine with verbose output for development
   python main.py -c config/basic --debug -vvv

-H, --hash
~~~~~~~~~~

**Type:** string
**Default:** None (auto-generated)

Defines a fixed hash for reproducible experiments. By default, the framework generates a unique hash for each run to identify experiments and outputs.

.. code-block:: bash

   # Use fixed hash for reproducibility
   python main.py -c config/basic -H "exp_001"

   # Restart interrupted experiment with same hash
   python main.py -c config/basic -H "interrupted_run_abc123"


Execution Mode Options
----------------------

--reinforcement
~~~~~~~~~~~~~~~

**Type:** flag
**Default:** False

Activates reinforcement learning during training. This enables the Actor-Critic model and associated RL components like reward functions, action policies, and environment interactions.

.. code-block:: bash

   # Enable reinforcement learning
   python main.py -c config/rl_experiment --reinforcement

   # Combine RL with study mode
   python main.py -c config/rl_experiment --reinforcement --study

**Configuration Requirements:**
When using ``--reinforcement``, ensure your configuration includes:

* Reward function parameters in ``reinforcement.ini``
* Action policy settings
* RL performance evaluation metrics
* Observation preprocessing parameters

--study
~~~~~~~

**Type:** flag
**Default:** False

Activates dataset study and analysis mode. This generates various plots, statistics, and visualizations about your dataset including:

* Data distribution analysis
* Feature correlation studies
* UMAP embeddings
* Similarity analysis between positive/negative samples

.. code-block:: bash

   # Run data analysis only
   python main.py -c config/basic --study

   # Combine study with training
   python main.py -c config/basic --study --reinforcement

**Output:**
Study mode generates outputs in the directory specified by ``studyOutput`` parameter in your configuration, typically including:

* Statistical distribution plots (PDF/ECDF)
* UMAP visualizations
* Correlation matrices
* Distance analysis plots

--inference
~~~~~~~~~~~

**Type:** string
**Default:** None

Activates inference-only mode using a pre-trained model. No training is performed; the system loads an existing model and performs predictions on the provided data.

.. code-block:: bash

   # Run inference with specific model label
   python main.py -c config/basic --inference "trained_model_v1"

   # Use hash-based model identification
   python main.py -c config/basic --inference "model_abc123"

**Inference Process:**
1. **Embedding Inference**: Generates embeddings for windowed data
2. **Triplet Prediction**: Performs predictions on triplet destination data
3. **Early Exit**: Returns immediately after inference (return code 0)

**Requirements:**
* Pre-trained model files must be available in the expected location
* Input data must be preprocessed and available
* Configuration must match the original training setup

--extension
~~~~~~~~~~~

**Type:** string
**Default:** None

Activates model extension functionality. This allows loading and extending pre-trained models with additional components or fine-tuning capabilities.

.. code-block:: bash

   # Load model extension
   python main.py -c config/basic --extension "quality_predictor"

   # Combine extension with RL
   python main.py -c config/basic --extension "adaptive_encoder" --reinforcement


Common Usage Patterns
---------------------

Development and Testing
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Quick development run with debugging
   python main.py -c config/test --debug -vv

   # Study data without training
   python main.py -c config/basic --study --debug

Production Training
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Standard training with logging
   python main.py -c config/production -r training.log

   # Reinforcement learning experiment
   python main.py -c config/rl_prod --reinforcement -r rl_experiment.log

Reproducible Experiments
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Fixed hash for reproducibility
   python main.py -c config/paper_exp -H "paper_experiment_1" -r results.log

   # Resume interrupted training
   python main.py -c config/resume -H "interrupted_hash_123"

Data Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Step 1: Analyze dataset
   python main.py -c config/analysis --study --debug

   # Step 2: Train based on insights
   python main.py -c config/optimized --reinforcement

   # Step 3: Run inference on new data
   python main.py -c config/inference --inference "final_model"


Error Handling and Troubleshooting
----------------------------------

Common Issues
~~~~~~~~~~~~~

**Configuration Not Found**
   Ensure the path specified with ``-c`` exists and contains valid INI files.

**Memory Issues**
   Use ``--debug`` flag to disable model saving and reduce memory usage during development.

**Interrupted Training**
   Use the same hash (``-H``) to potentially resume from checkpoints.

**Missing Dependencies**
   Check ``requirements.txt`` and ensure all packages are installed.

Exit Codes
~~~~~~~~~~

* **0**: Successful completion (or inference mode)
* **Non-zero**: Error occurred (check logs for details)