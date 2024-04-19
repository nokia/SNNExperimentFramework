# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
Service Handler module
======================

Module use to manage templates file

"""

import ast

from .environment import EnvironmentHandler as ENV
from ..util.strings import strings as s
from ..io.logger import LogHandler as LH
from ..io.IOHandler import IOHandler as IOH
from typing import List, Dict


class template():
    """template.
    General template class with the default methods to handle a template
    """

    def __init__(self,  logger: LH, class_name: str = "Template", default: str = ""):
        """__init__.
        Template class initializer

        Parameters
        ----------
        logger : LH
            logger used by the classes
        class_name : str
            class_name to use in the log messages
        default : str
            default
        """
        self.template = default
        self.class_name = class_name
        # load the logger
        self.logger: LH = logger

    def write(self, msg: str, level: int) -> None:
        """write.
        Function used by the class to write into the logger

        Parameters
        ----------
        msg : str
            msg that needs to be printed
        level : int
            level of the message that needs to be printed

        Returns
        -------
        None

        """
        self.logger.write(f"{self.class_name}: {msg}", level)

    def __general_format_list(self, fstr: str, lst: List[str], entering: int = 6) -> str:
        """__general_format_list.
        Function used to generally format a list into a string
        sequence used for the docker yaml file

        Parameters
        ----------
        fstr : str
            Formatting string to use, common one would be '- {}'
        lst : List[str]
            List object with all the elements that needs to be formatted
        entering : int
            Number of spaces to put before the object

        Returns
        -------
        str object formatted correctly

        """
        resulting_str = ""
        for i, obj in enumerate(lst):
            if i == 0:
                resulting_str += fstr.format(obj)
            else:
                resulting_str += f"\n{entering*' '}" + fstr.format(obj)

        return resulting_str

    def attribute_formatting(self, attributes: Dict, fstr: str, key: str, entering: int = 6) -> None:
        """attribute_formatting.
        General attribute formatting string, pass the format style and
        the keyword to use, the keyword would be checked in the self.attribues
        object

        Parameters
        ----------
        attributes: Dict
            Attributes dictionaryt that needs to be formatted
        fstr : str
            Formatting string
        key : str
            Key object that needs to be used
        entering : int
            Number of spaces to use before the object in the string format
            for correctly format the yaml file

        Returns
        -------
        None

        """
        attributes[key] = [attributes[key]] \
                if type(attributes[key]) != list \
                else attributes[key]
        attributes[key] = self.__general_format_list(fstr, attributes[key], entering=entering)
        self.write(f"{key} formatting: {attributes[key]}", LH.DEBUG)

    def __str__(self) -> str:
        """__str__.
        Returns the string formatting

        Parameters
        ----------

        Returns
        -------
        str

        """
        return self.template


class serviceTemplate(template):
    """serviceTemplate.
    Class used to handle templates of a service
    """

    def __init__(self, node: dict, io: IOH, logger: LH):
        # Inheritance init
        super().__init__(logger, class_name="serviceTemplate")

        # Open the template for the services
        with open(io[s.service_template], "r") as service_template:
            self.serviceTemplate = service_template.read()

        # Load the node properties
        node = node[1]
        self.attributes: Dict = {
                s.service_name: node["name"],
                s.context_path: node["context"],
                s.image: node["image"],
                s.host_name: node["name"],
                s.volumes: ast.literal_eval(node["volumes"]),
                s.devices: node["devices"],
                s.commands: ast.literal_eval(node["command"]),
                s.net_name: node["network"],
                s.ipv4_address: node["ipv4"],
            }

        super().write(f"Service dictionary loaded: {self.attributes}", LH.DEBUG)

        super().attribute_formatting(self.attributes, "- {}", s.volumes)
        super().attribute_formatting(self.attributes, "- {}", s.devices)
        super().attribute_formatting(self.attributes, "- {}", s.commands)

        self.attributes[s.dependencies] = str(dependenciesTemplate(node, io, logger))

        # Generate the template with the properties
        self.template = self.serviceTemplate.format(**self.attributes)
        super().write(f"Teamplatify: \n{self.template}", LH.DEBUG)

class dependenciesTemplate(template):
    """dependenciesTemplate.
    Class used to handle templates of dependencies that will be used by
    a docker compose service
    """

    def __init__(self, node: dict, io: IOH, logger: LH):
        # Inheritance init
        super().__init__(logger, class_name="dependenciesTemplate")

        # Open the template for the services
        with open(io[s.dependencies_template], "r") as template:
            self.dependenciesTemplate = template.read()

        self.attributes: Dict = {
                s.dependencies: ast.literal_eval(node["depends"])
            }

        # load the logger
        self.logger: LH = logger
        super().write(f"Dependencies template loaded {self.attributes}", LH.INFO)

        if len(self.attributes[s.dependencies]) != 0:
            super().write(f"dependencies in {self.attributes[s.dependencies]}", LH.DEBUG)
            super().attribute_formatting(self.attributes, "- {}", s.dependencies)
            self.template = self.dependenciesTemplate.format(**self.attributes)

        # Generate the template with the properties
        super().write(f"Teamplatify: \n{self.template}", LH.DEBUG)

class networkTemplate(template):
    """networkTemplate.
    Class used to handle templates of a network that will be used by
    a docker compose environment
    """

    def __init__(self, environment: ENV, io: IOH, logger: LH):
        # Inheritance init
        super().__init__(logger, class_name="networkTemplate")

        # Open the template for the services
        with open(io[s.network_template], "r") as network_template:
            self.networkTemplate = network_template.read()

        self.attributes: Dict = {
                s.network: environment[s.network],
                s.net_name: environment[s.network_name]
            }

        # load the logger
        self.logger: LH = logger
        super().write(f"Network template loaded {self.attributes}", LH.INFO)

        # Generate the template with the properties
        self.template = self.networkTemplate.format(**self.attributes)
        super().write(f"Teamplatify: \n{self.template}", LH.DEBUG)

class DkrComposeTemplate(template):
    """DkrComposeTemplate.
    Class used to handle templates of a Docker compose yaml file
    """

    def __init__(self, services: List[serviceTemplate], network: networkTemplate,
                 io: IOH, logger: LH):
        # Inheritance init
        super().__init__(logger, class_name="DkrComposeTemplate")

        # Open the template for the services
        with open(io[s.compose_template], "r") as compose_template:
            self.composeTemplate = compose_template.read()

        self.attributes: Dict = {
                s.cmp_services: services,
                s.cmp_networks: network
            }

        # load the logger
        self.logger: LH = logger
        super().write(f"Dkr Compose template loaded", LH.INFO)

        super().attribute_formatting(self.attributes, "{}", s.cmp_services, entering=0)

        # Generate the template with the properties
        self.template = self.composeTemplate.format(**self.attributes)
        super().write(f"Teamplatify: \n{self.template}", LH.DEBUG)

    def docker_compatible(self) -> bool:
        return True

