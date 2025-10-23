Installation
============

This guide covers the installation of the SNN2 framework on various platforms.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~~

* **Python**: 3.7 or higher
* **Operating System**: Linux, macOS (Not tested), or Windows (Not tensted)
* **Memory**: Depends on the datasets used, suggested at least 16GB RAM
* **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

Required Dependencies
~~~~~~~~~~~~~~~~~~~~~

The framework relies on several key libraries:

* **TensorFlow**: 2.x (for neural network operations)
* **NumPy**: For numerical computations
* **Pandas**: For data manipulation
* **Matplotlib/Seaborn**: For visualization
* **scikit-learn**: For machine learning utilities
* **UMAP**: For dimensionality reduction
* **FastDTW**: For dynamic time warping

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone the Repository**

   .. code-block:: bash

      git clone <repository-url>
      cd SNN2

2. **Create Virtual Environment** (Recommended)

   .. code-block:: bash

      # Install the required applications (linux system)
      make install

      # Create the virtual environment automatically manages the installation of the latest requirements
      make virtualenv

4. **Install the Package**

   .. code-block:: bash

      # Development installation (editable)
      pip install -e .

      # Or regular installation
      pip install .


GPU Support
-----------

CUDA Setup
~~~~~~~~~~~

For GPU acceleration, ensure you have:

1. **NVIDIA Drivers**: Latest drivers for your GPU
2. **CUDA Toolkit**: Compatible version with TensorFlow
3. **cuDNN**: NVIDIA Deep Neural Network library

.. code-block:: bash

   # Verify GPU detection
   python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"

TensorFlow GPU
~~~~~~~~~~~~~~

Install TensorFlow with GPU support:

.. code-block:: bash

   pip install tensorflow-gpu

Or use the unified TensorFlow package (TF 2.1+):

.. code-block:: bash

   pip install tensorflow

Verification
------------

Test Installation
~~~~~~~~~~~~~~~~~

1. **Basic Import Test**

   .. code-block:: bash

      python -c "import SNN2; print('SNN2 imported successfully')"

2. **Run Help Command**

   .. code-block:: bash

      python SNN2/main.py --help


Development Setup
-----------------

Up to now the development setup is identically to the main setup.
The following sections will be updated once there will be a differentiation between the two setups.



Next Steps
----------

After successful installation:

2. **Review Command Line Arguments**: :doc:`arguments`
3. **Explore Configuration Options**: :doc:`configuration`