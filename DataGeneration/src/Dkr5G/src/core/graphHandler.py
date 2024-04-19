# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Graph Handler module
====================

Module use to manage the graph file

"""

import re
import networkx as nx

from Dkr5G.src.util.strings import strings as s
from Dkr5G.src.io.logger import LogHandler as LH
from Dkr5G.src.io.IOHandler import IOHandler as IOH
from Dkr5G.src.core.environment import EnvironmentHandler as ENV
from Dkr5G.src.io.files import FileHandler as FH
from Dkr5G.src.core.templateHandler import serviceTemplate as STmpl
from Dkr5G.src.core.templateHandler import networkTemplate as NTmpl
from Dkr5G.src.core.templateHandler import DkrComposeTemplate as DCTmpl

class GraphHandler():
    """GraphHandler.
    Class used to handle the graph
    """

    def __init__(self, graphFile: FH, logger: LH):
        """__init__.

        Parameters
        ----------
        graphFile : FH
            graphFile File Handler
        logger : LH
            logger handler
        """
        self.logger = logger
        self.__graph = nx.read_graphml(graphFile.path)
        self.logger.write(self.__class__.__name__, f"Graph loaded {self.graph}", level=LH.DEBUG)

    def generateDockerFile(self, env: ENV, io: IOH) -> None:
        """generateDockerFile.
        Function that use the current graph configuration, plus
        the environment and the input output handler in order to generate
        the docker compose file that will be used to start all the services

        Parameters
        ----------
        env : ENV
            environment that is going to be used by the function
        io : IOH
            Input output handler used by the function

        Returns
        -------
        None

        """
        # use environment variables to fulfill the strings attributes of the
        # graphml file.
        services = []
        for node in self.graph.nodes(data=True):
            # Check variables in each object
            for obj in node[1].keys():
                node[1][obj] = self.evaluateObj(node[1][obj], env, io)

                # Check that the commands are not going to harm
                if not self.checkObj(node[1][obj]):
                    raise Exception(f"Harmful command found {node[1][obj]}")

            # Require the generation of all the services templates
            services.append(STmpl(node, io, self.logger))

        # Load the network from the environment
        net = str(NTmpl(env, io, self.logger))

        # Save all the generated templates inside the docker-file
        dc = DCTmpl(services, net, io, self.logger)

        # Check that the file is docker-compatible
        assert dc.docker_compatible()

        self.logger.write(self.__class__.__name__, "The graph is docker compatible")

        # Save the template to a file
        with open(io[s.docker_file], "w") as dc_file:
            dc_file.write(str(dc))

    def evaluateObj(self, obj: str, env: ENV, io: IOH) -> str:
        """evaluateObj.
        Function used in order to evaluate a particular object of the
        graph if it contains a reference to another point in the
        graphml file itself or the environment

        Parameters
        ----------
        obj : str
            obj that needs to be analyzed
        env : ENV
            environment to use
        io : IOH
            io handler to use

        Returns
        -------
        str
            correctly formatted

        """

        # Look for the regular expression into the string
        result = re.findall(r"\{[^\}]*\}", obj)
        result = [element.replace('{', '').replace('}', '') for element in result]

        # If there are no correspondences just return the object itself
        if len(result) == 0:
            return obj

        elements = {}
        # For each element in the object substitute it with the correct translation
        for res in result:
            # Check if it is a reference to another graph object
            if '[' in res and ']' in res:
                node = res.split('[')[0]
                key = res.split('[')[1].replace(']', '')
                if key is not None:
                    d_node_objects = dict(self.graph.nodes(data=key, default=None))
                    if node in d_node_objects.keys():
                        elements[node] = {key: d_node_objects[node]}
                    if len(re.findall(r"\{[^\}]*\}", elements[node][key])) > 0:
                        elements[node] = {key: self.evaluateObj(elements[node][key], env, io)}

            # Check if it is an environment reference
            elif env.check(res):
                elements[res] = env[res]
            # Check if it is a io reference
            elif io.check(res):
                elements[res] = io[res]
            # If none of the previous apply just set it to None
            else:
                elements[res] = None

        # Return the object correctly formatted
        return obj.format(**elements)

    # TODO replace with the real checkObj to avoid harmful commands
    def checkObj(self, obj: object) -> bool:
        return True

    @property
    def graph(self) -> nx.Graph:
        """graph.

        Parameters
        ----------

        Returns
        -------
        nx.Graph

        """
        return self.__graph

    def __str__(self) -> str:
        """__str__.

        Parameters
        ----------

        Returns
        -------
        str

        """
        return str(self.graph)

