#!/bin/bash

LC_NUMERIC=C

root="Conf/Dario-tests"

RL_input="${root}/RL-Epoch"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement

RL_input="${root}/RL-50-batch"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement

RL_input="${root}/RL-50-Epoch"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement

RL_input="${root}/RL-Epoch-NoMarginInf"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement

RL_input="${root}/RL-50-batch-NoMarginInf"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement

RL_input="${root}/RL-50-Epoch-NoMarginInf"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement

RL_input="Conf/ZoneRL/SL-OTTRL-Zone"
echo ${RL_input}
snn -c ${RL_input} --study --reinforcement


