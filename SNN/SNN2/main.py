# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Main module
===========

Main module of the project, use it to load all the different parts and
generate the required datasets

"""


import argparse
import os
import pkg_resources
import tensorflow as tf

from SNN2.src.environment.conf import conf as env_conf
from SNN2.src.util.strings import s
from SNN2.src.io.utils import redirect
from SNN2.src.io.directories import DirectoryHandler as DH
from SNN2.src.io.configuration import ConfHandler as CH
from SNN2.src.io.IOHandler import IOHandler as IOH
from SNN2.src.io.logger import LogHandler as LH
from SNN2.src.params.paramHandler import ParamHandler as PH
from SNN2.src.io.pickleHandler import PickleHandler as PkH
from SNN2.src.core.data.PreProcessing import PreProcessing as PP
from SNN2.src.core.experiment import ExperimentEnv as Exp
from SNN2.src.model.managers.selector import selector as MH
from SNN2.src.model.reinforcement.reinfoceModelHandler import ReinforceModelHandler as RLMH
from SNN2.src.plot.study import Study

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Configure command line argument parser for the main application
parser = argparse.ArgumentParser(usage="usage: main.py [options]",
                      description="Execute the learning on the network datasets",
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Configuration and basic options
parser.add_argument("-c", "--confFolder", dest="confFolder", default="config/basic",
                    action="store",
                    help="define the input configuration folder where to find ini files")
parser.add_argument("-v", "--verbose", dest="verbosity", default=1,
                    action="count", help=s.verbose_help)
parser.add_argument("-r", "--redirect", dest="redirect", default=None,
                    action="store", help="Redirect the stdout")

# Debug and development options
parser.add_argument("-D", "--debug", dest="debug", default=False,
                    action="store_true", help="Disable the model save functionality")
parser.add_argument("-H", "--hash", dest="fix_hash", default=None,
                    action="store", help="Define a fixed hash")

# Execution mode options
parser.add_argument("--reinforcement", dest="reinforcement", default=False,
                    action="store_true",
                    help="Activate the flag to use Reinforcement learning during the training")
parser.add_argument("--study", dest="study", default=False,
                    action="store_true",
                    help="Deactivate the flag to execute the study of the datasets")
parser.add_argument("--inference", dest="inference", default=None,
                    action="store",
                    help="Activate the inference execution, if None no inference is executed,"\
                         "otherwise the value is the label to use")
parser.add_argument("--extension", dest="extension", default=None,
                    action="store", help="Activate the extension execution")

def main():
    """
    Main execution function for the SNN2 neural network training system.

    This function orchestrates the entire machine learning pipeline including:
    - Configuration parsing and setup
    - Data preprocessing
    - Model initialization (regular and reinforcement learning)
    - Training/fitting phase
    - Evaluation and testing

    The function supports multiple execution modes:
    - Standard training mode
    - Reinforcement learning mode (--reinforcement)
    - Data study mode (--study)
    - Inference mode (--inference)

    Returns:
        int: 0 if inference mode is executed, otherwise None
    """
    # Parse command line arguments
    options = parser.parse_args()

    # Setup output redirection if specified
    output = None if options.redirect is None else redirect(options.redirect)

    # Initialize configuration system
    # Load custom configuration files from specified folder
    main_dir: DH = DH(options.confFolder)

    # Verify and load default configuration files
    # This ensures all required parameters have default values
    assert pkg_resources.resource_exists(__name__, s.default_conf)
    default_conf = pkg_resources.resource_filename(__name__, s.default_conf)
    conf: CH = CH(DH(default_conf))
    # Override defaults with user-specified configuration
    conf.update(main_dir)

    print("---- Parameter parsing ----")
    # Initialize I/O handler from configuration
    io: IOH = IOH.from_cfg(conf)
    # Setup logging system with appropriate verbosity level
    logger: LH = LH(io[s.log_file], LH.findLL(options.verbosity))

    logger("SNN2.main", "Loading the parameters")
    # Load and parse all configuration parameters
    pm = PH.from_cfg(conf, io, logger=logger)
    # Override hash if specified via command line
    if options.fix_hash is not None:
        pm.force_hash = options.fix_hash
    # Initialize pickle handler for data serialization
    ph = PkH(io, pm[s.appendix_key], logger, hash=pm.hash,
             unix_time=pm[s.unix_time])

    print("---- Environment configuration ----")
    # Configure the execution environment (GPU settings, memory limits, etc.)
    env_conf(PH.filter(pm, s.param_Environment_type))

    print("---- Data PreProcessing ----")
    # Force data preprocessing to run on CPU to avoid GPU memory issues
    with tf.device("cpu:0"):
        # Initialize data preprocessing pipeline
        # This handles data loading, cleaning, normalization, and transformation
        pp = PP(PH.filter(pm, s.param_PreProcessing_type),
                PH.filter(pm, s.param_action_type),
                PH.filter(pm, s.param_flow_type),
                ph, logger=logger)

    print("---- Data Study ----")
    # Optional: Execute data analysis and visualization studies
    # This generates plots and statistics about the dataset
    if options.study:
        Study(PH.filter(pm, s.param_study_type), pp,
              PH.filter(pm, s.param_action_type),
              ph, logger=logger, hash=pm.hash)
        return 0

    print("---- Model definition ----")
    # Initialize the main neural network model
    # This sets up the architecture, layers, loss functions, and metrics
    model = MH(pm["ModelManager"],
               PH.filter(pm, s.param_model_type),
               PH.filter(pm, s.param_embedding_type),
               PH.filter(pm, s.param_layer_type),
               PH.filter(pm, s.param_metric_type),
               PH.filter(pm, s.param_loss_type),
               PH.filter(pm, s.param_lossParam_type),
               PH.filter(pm, s.param_numpyRng_type),
               ph, logger, pp=pp, hash=pm.hash, debug=options.debug)

    # Check if running in inference mode (no training, only prediction)
    if options.inference is not None:
        print("---- Inference Cluster ----")
        # Perform embedding inference on windowed data
        Exp.emb_inference(model, pp.data_prop['Windows']["TfDataset"],
                          options.inference, ph)
        print("---- Prediction triplet ----")
        # Generate predictions on triplet destination data
        Exp.predict(model, pp.data_prop['TripletDst']["TfDataset"])
        return 0  # Exit early, no training needed

    # Initialize reinforcement learning model if requested
    rl_model = None
    if options.reinforcement:
        print(f"---- RL Model Definition {pm['RLModelHandler']} ----")
        # Setup reinforcement learning model with reward functions and policies
        # This enables adaptive learning during training
        rl_model = RLMH(pm["RLModelHandler"], pp,
                     PH.filter(pm, s.param_reward_function_type),
                     PH.filter(pm, s.param_RLPerfEval_type),
                     PH.filter(pm, s.param_RLObsPP_type),
                     PH.filter(pm, s.param_RLActPolicy_type),
                     PH.filter(pm, s.param_reinforce_model_type),
                     PH.filter(pm, s.param_embedding_type),
                     PH.filter(pm, s.param_layer_type),
                     PH.filter(pm, s.param_metric_type),
                     PH.filter(pm, s.param_loss_type),
                     PH.filter(pm, s.param_lossParam_type),
                     PH.filter(pm, s.param_numpyRng_type),
                     ph, logger, hash=pm.hash, debug=options.debug)

    print("---- Experiment preparation ----")
    # Setup the experiment environment with all components
    # This orchestrates the training process, callbacks, and evaluation
    exp = Exp(pp, model,
              PH.filter(pm, s.param_experiment_type),
              PH.filter(pm, s.param_Environment_type),
              PH.filter(pm, s.param_callback_type),
              PH.filter(pm, s.param_fitmethod_type),
              PH.filter(pm, s.param_action_type),
              PH.filter(pm, s.param_grayEvolution_type),
              PH.filter(pm, s.param_numpyRng_type),
              PH.filter(pm, s.param_RLEnvExitFunction_type),
              ph, logger=logger,
              rl_model=rl_model,
              model_extension_name=options.extension)

    print("---- Fitting phase ----")
    # Execute the main training loop
    # This trains the model on the preprocessed data
    exp.fit()

    print("---- Tests Execution ----")
    # Evaluate the trained model and save results
    # This generates performance metrics and saves outputs
    exp.evaluate(save_output=True)

    # Clean up: close output redirection if it was used
    if options.redirect is not None and output is not None:
        output.close()

# Entry point: execute main function only if script is run directly
if __name__ == "__main__":
    main()
