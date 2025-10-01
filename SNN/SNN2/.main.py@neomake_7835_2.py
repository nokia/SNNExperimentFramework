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
# Copyright (C) 2020 Mattia Milani <mattia.milani@nokia.com>

"""
Main module
===========

Main module of the project, use it to load all the different parts and
generate the required datasets

"""


import argparse
import pkg_resources
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


parser = argparse.ArgumentParser(usage="usage: main.py [options]",
                      description="Execute the learning on the network datasets",
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--confFolder", dest="confFolder", default="config/basic",
                    action="store", help="define the input configuration folder where to find ini files")
parser.add_argument("-v", "--verbose", dest="verbosity", default=1,
                    action="count", help=s.verbose_help)
parser.add_argument("-r", "--redirect", dest="redirect", default=None,
                    action="store", help="Redirect the stdout")
parser.add_argument("-D", "--debug", dest="debug", default=False,
                    action="store_true", help="Disable the model save functionality")
parser.add_argument("--reinforcement", dest="reinforcement", default=False,
                    action="store_true", help="Activate the flag to use Reinforcement learning during the training")
parser.add_argument("--study", dest="study", default=False,
                    action="store_true", help="Deactivate the flag to execute the study of the datasets")
parser.add_argument("-H", "--hash", dest="fix_hash", default=None,
                    action="store", help="Define a fixed hash")

def main():
    # Parse the arguments
    options = parser.parse_args()

    # Redirect the output
    output = None if options.redirect is None else redirect(options.redirect)

    # Load the custom configuration files
    main_dir: DH = DH(options.confFolder)

    # check that the default configuration exists
    assert pkg_resources.resource_exists(__name__, s.default_conf)
    default_conf = pkg_resources.resource_filename(__name__, s.default_conf)
    conf: CH = CH(DH(default_conf))
    conf.update(main_dir)

    print("---- Parameter parsing ----")
    io: IOH = IOH.from_cfg(conf)
    logger: LH = LH(io[s.log_file], LH.findLL(options.verbosity))

    logger("SNN2.main", "Loading the parameters")
    pm = PH.from_cfg(conf, io, logger=logger)
    if options.fix_hash is not None:
        pm.force_hash = options.fix_hash
    ph = PkH(io, pm[s.appendix_key], logger, hash=pm.hash)

    print("---- Environment configuration ----")
    env_conf(PH.filter(pm, s.param_Environment_type))

    print("---- Data PreProcessing ----")
    with tf.device("cpu:0"):
        pp = PP(PH.filter(pm, s.param_PreProcessing_type),
                PH.filter(pm, s.param_action_type),
                PH.filter(pm, s.param_flow_type),
                ph, logger=logger)

    print("---- Data Study ----")
    if options.study:
        Study(PH.filter(pm, s.param_study_type), pp,
              PH.filter(pm, s.param_action_type),
              ph, logger=logger, hash=pm.hash)

    print("---- Model definition ----")
    model = MH(pm["ModelManager"],
               PH.filter(pm, s.param_model_type),
               PH.filter(pm, s.param_embedding_type),
               PH.filter(pm, s.param_layer_type),
               PH.filter(pm, s.param_metric_type),
               PH.filter(pm, s.param_loss_type),
               PH.filter(pm, s.param_lossParam_type),
               PH.filter(pm, s.param_numpyRng_type),
               ph, logger, pp=pp, hash=pm.hash, debug=options.debug)

    rl_model = None
    if options.reinforcement:
        print(f"---- RL Model Definition {pm['RLModelHandler']} ----")
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
              rl_model=rl_model)

    print("---- Fitting phase ----")
    exp.fit()

    print("---- Tests Execution ----")
    test_stats = exp.evaluate(save_output=True)

    if options.redirect is not None and output is not None:
        output.close()

if __name__ == "__main__":
    main()
