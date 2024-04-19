# Dkr5G-Exp

This repository includes the code and the resources to execute the experments
on the top of the `Dkr5G` available in the [Dkr5G repository](https://gitlabe2.ext.net.nokia.com/milani/dkr5g)

## Repository goal

The main goal of this repository is to provide all the resources to execute
experiments and/or reproduce experiments.
The code will be executed using `Dkr5G` without modifing it.
To require new features please refer to the docker implementation repository

## Experiment structure

The main execution flow of the experiments is the following one:

1) Configuration files
2) Dkr5G execution
2.1) Dkr5G - Pre-events scripts
2.2) Dkr5G - Events execution
2.3) Dkr5G - Post-events scripts
3) Post experiments data processing

The first point requires to correctly configure an experiment through the configuration files.
---To be inserted here a reference to the configuration file wiki---

The second step is actually to execute the experiments through the following command:

`dkr5g -d <experiment-directory/path>`

The software will take care of generating the docker compose environment based on the graphml provided and
execute all the events configured.
All the folders and log files depends on how the files have been configured.

The last step is usually composed of two major points:
- Aggregation of multiple experiments
- Datasets cleaning and plot generation

It's possible to find in the `bash` folder some helpers for this purposes.

## Experiment tutorials

Please refer to the following wiki sections to be guided throguh some initial
tutorials

## ATELIER reproducibility

Please follow this sections to reproduce the experiments and plots presented
into the CoNEXT paper
