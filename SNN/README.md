# SNN Framework

The Siamese Neural Network (SNN) Framework is a comprehensive machine learning toolkit designed for neural network experimentation and training. This framework provides a highly customizable and extensible environment for implementing siamese neural networks ([Keras Siamese Networks documentation](https://keras.io/examples/vision/siamese_network/))
with support for reinforcement learning, data preprocessing, and automated experimentation workflows.

The toolkit documentation is available at: [SNN Framework docs](https://tiamilani.github.io/SNNExperimentFramework/)

## Key Features

- **Siamese Neural Network Architecture**: Specialized framework for training and experimenting with siamese neural networks
- **Reinforcement Learning Integration**: Built-in support for reinforcement learning algorithms with customizable reward functions and policies
- **Modular Design**: Highly configurable system with pluggable components for models, callbacks, loss functions, and metrics
- **Data Preprocessing Pipeline**: Data handling and preprocessing capabilities
- **Automated Experimentation**: Support for batch experiments with configurable parameters
- **Extensible Architecture**: Easy integration of custom models, layers, and evaluation metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

It's highly suggested to use a virtualenvironment, the makefile contained in the repositroy automatically installs and generates a local environment.
If you prefer to not use a local environment install the requirements contained in `requirements.txt` through a pip command.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tiamilani/SNNExperimentFramework.git
   cd SNNExperimentFramework/SNN
   ```

2. **Install virtual environment and requirements**:
   ```bash
   make virtualenv
   ```

3. **Install the SNN package**:
   ```bash
   python -m pip install -e .
   ```

## Quick Start

### Basic Usage

Run the SNN framework with an experiment configuration:

```bash
snn -c config/experiment
```

### Command Line Options

- `-c, --confFolder`: Specify configuration folder (default: `config/basic`)
- `-v, --verbose`: Increase verbosity level
- `-r, --redirect`: Redirect stdout to file
- `--reinforcement`: Enable reinforcement learning mode
- `--study`: Execute dataset analysis and visualization
- `--inference LABEL`: Run inference mode with specified label
- `--debug`: Enable debug mode (disables model saving)

### Configuration

The framework uses INI-based configuration files located in the configuration folder. Key configuration categories include:

- **Model Configuration**: Neural network architecture and parameters
- **Data Processing**: Input data handling and preprocessing settings
- **Training Parameters**: Learning rates, batch sizes, and optimization settings
- **Reinforcement Learning**: Reward functions and policy configurations
- **Environment**: GPU/CPU settings and resource allocation

All the defulat parameters for the configuration are contained in `SNN2/default_ini` all `.ini` files in this directory are loaded by the tool when executed, if you prefer to change some of the paramters, e.g. use a different path for the logs or a different number of training iterations, you can load your own configuration files through the `-c` option.
Custom parameters always takes priority over default configured values.

### Example Execution Modes

1. **Standard Training**:
   ```bash
   snn -c config/my_experiment -v
   ```

2. **Reinforcement Learning Mode**:
   ```bash
   snn -c config/rl_experiment --reinforcement
   ```

3. **Data Analysis**:
   ```bash
   snn -c config/basic --study
   ```

4. **Inference Mode**:
   ```bash
   snn -c config/trained_model --inference my_label
   ```

## Documentation

Generate comprehensive documentation using:

```bash
make docs
```

The generated documentation will be available in `docs/public/index.html`.

## Development

### Testing

Run the test suite:

```bash
pytest tests/
```

### Code Quality

Check code quality with pylint:

```bash
make pylint
```

## License

© 2024 Nokia
Licensed under the BSD 3 Clause license
SPDX-License-Identifier: BSD-3-Clause

## Author

**Mattia Milani**
Email: mattia.milani@nokia.com

## Version

Current version: 3.0.1
