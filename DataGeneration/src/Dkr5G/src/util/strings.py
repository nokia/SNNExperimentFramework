# © 2024 Nokia
# Licensed under the BSD 3 Clause license
# SPDX-License-Identifier: BSD-3-Clause

"""
strings module
==============

Module used to control in one place all the strings of the program

"""

class strings:

    # Parameters
    default_conf = "conf/default.cfg"
    conf_help = "Define the yaml input file with the environment configuration"
    verbose_help = "Define the verbose level of the program"
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    debug_help = "Use this option to enter in debug mode, the environment will be loaded but not created and all the schedule jobs will be loaded but not executed"
    graph_help = "Graphml file that defines the graph to use during the experiment"
    events_help = "Events yaml file with all the events that should be scheduled"
    detect_help = "Automatically detect the configuration, graph and events file just giving the main folder of the experiment, the files have to be placed in the correct folder with the correct name inside the main folder \'Conf/configuration.cfg\', \'Graph/graph.graphml\' and \'Events/events.yaml\', this command has priority over the conf, graph and events"

    # Data
    data_key = "data"
    io_name_key = "name"
    io_path_key = "path"
    io_type_key = "type"
    io_exists_key = "exists"
    folder_obj_types = ["folder", "directory"]
    file_obj_types = ["file"]

    # Conf keys
    log = "Log_file"

    # Environment
    default_env_key = "default_environment"
    env_key = "environment"
    env_name_key = "name"
    env_type_key = "type"
    env_value_key = "value"
    env_types = ["string", "int", "float"]

    env_id = "id"
    network = "network"
    network_name = "network_name"
    server_video = "server_video"
    extension = "extension"
    server_result = "server_result"

    # Mandatory files
    log_file = "log_file"
    graph_file = "graph_file"
    service_template = "service_template"
    dependencies_template = "dependencies_template"
    network_template = "network_template"
    compose_template = "compose_template"

    # Template attributes
    service_name = "service_name"
    context_path = "context_path"
    image = "image"
    host_name = "host_name"
    volumes = "volumes"
    devices = "devices"
    commands = "commands"
    net_name = "net_name"
    ipv4_address = "ipv4_address"
    dependencies = "dependencies"
    cmp_services = "services"
    cmp_networks = "networks"

    # Docker files
    docker_folder = "docker_folder"
    docker_file = "docker_file"

    # Commands
    docker_up = "docker-compose -f {file} up -d"
    docker_down=  "docker-compose -f {file} down"
    docker_command = "docker exec -d {} sh -c \"{}\""
    timeout_command = "timeout -k {} {} bash -c  \'{}\'"

    # Events
    events_key = "events"
    events_postDocker_key = "postEvents"
    eve_node = "node"
    eve_command = "command"
    eve_st = "start_time"
    eve_kt = "kill_time"

    # DetectFolder
    conf_default_folder = "Conf/"
    graph_default_folder = "Graph/"
    events_default_folder = "Events/"
    conf_default_file_name = "configuration.cfg"
    graph_default_file_name = "graph.graphml"
    events_default_file_name = "events.yaml"
