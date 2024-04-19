# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Main module
===========

Main module of the project, use it to load all the different parts and
execute the experiments

The main components required are:
- A configuration file that points out all the variable that will
be used in the environment
- Few variables can be overwritten by the command line in order to execute
the experiments with small changes
"""

import pkg_resources
import argparse
import yaml

# from pkg_resources import resource_filename
from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.io.directories import DirectoryHandler as DH
from Dkr5G.src.core.graphHandler import GraphHandler as GH
from Dkr5G.src.core.experiment import ExperimentHandler as EXP
from Dkr5G.src.core.environment import EnvironmentHandler as ENV
from Dkr5G.src.io.configuration import ConfHandler as CH

parser = argparse.ArgumentParser(usage="usage: Dkr5G.py [options]",
                      description="Generate and execute a 5G dockerized environment",
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--detect", dest="detectFolder", default=None,
                    action="store", help=s.detect_help)
parser.add_argument("-c", "--conf", dest="conf", default=s.default_conf,
                    action="store", help=s.conf_help)
parser.add_argument("-g", "--graph", dest="graph", default=None,
                    action="store", help=s.graph_help)
parser.add_argument("-e", "--events", dest="events", default=None,
                    action="store", help=s.events_help)
parser.add_argument("-v", "--verbose", dest="verbosity", default=1,
                    action="count", help=s.verbose_help)
parser.add_argument("-D", "--debug", dest="debug", default=False,
                    action="store_true", help=s.debug_help)


def main():
    # Parse the arguments
    options = parser.parse_args()

    # Check if a detect folder has been given
    if not options.detectFolder is None:
        main_dir = DH(options.detectFolder)
        custom_cfg, custom_graph, custom_events = FH.detect(main_dir)
    else:
        custom_cfg = FH(options.conf, create=False)
        custom_graph = FH(options.graph, create=False)
        custom_events = FH(options.events, create=False)

    assert pkg_resources.resource_exists(__name__, s.default_conf)
    default_conf = pkg_resources.resource_filename(__name__, s.default_conf)
    conf = CH(FH(default_conf, create=False))
    if not custom_cfg is None:
        conf.update(custom_cfg)

    # Configure the environment
    # Load the IO part in the IO object
    io = IOH.from_cfg(conf)

    # Load the logger
    logger = LH(io[s.log_file], LH.findLL(options.verbosity))

    # load environment variables
    logger("Dkr5G.main", "Loading the environment")
    default_env_handler = io.get_handler(s.default_env_key)
    assert isinstance(default_env_handler, (FH, DH))
    env_cfg = CH(default_env_handler)

    if io[s.default_env_key] != io[s.env_key]:
        custom_env = io.get_handler(s.env_key)
        assert isinstance(custom_env, (FH, DH))
        env_cfg.update(custom_env)

    env = ENV.from_cfg(env_cfg, io, logger)

    # load the graph
    logger("Dkr5G.main", "Loading the graph")
    graph = GH(custom_graph, logger)

    # Generate the environment
    logger("Dkr5G.main","Generate the docker-compose file")
    graph.generateDockerFile(env, io)

    # Load the experiment handler
    logger("Dkr5G.main","Generate the experiment handler")
    exp = EXP(graph, env, io, logger)

    # Schedule the jobs
    logger("Dkr5G.main","Schedule the jobs")
    with open(custom_events.path, 'r') as file:
        data = yaml.full_load(file)
        exp.scheduleJobs(data[s.events_key])
        exp.schedulePostJobs(data[s.events_postDocker_key])

    logger("Dkr5G.main","Load the docker environment")
    exp.start_env()

    # execute the jobs
    logger("Dkr5G.main","Execute the jobs scheduled")
    exp.executeAll(debug=options.debug)

    # Stop the environment
    logger("Dkr5G.main","Stop the docker environment")
    exp.stop_env(debug=options.debug)

    logger("Dkr5G.main", "Post jobs execution")
    exp.postExecution(debug=options.debug)

if __name__ == "__main__":
    main()

