#!/bin/bash

LC_NUMERIC=C

root="Conf/likeApple"

#RL_input="${root}/RL-finalConf-multiExp/RL-likeApple-finalConf-10Ep05-"
#parallel -j 1 --bar "snn -c ${RL_input}{} -vvv --study --reinforcement " ::: $(seq 0 4);

RL_input="${root}/RL-finalConf-multiExp-Start100/RL-likeApple-finalConf-10Ep05-"
parallel -j 1 --bar "snn -c ${RL_input}{} -vvv --study --reinforcement " ::: $(seq 3 3);
parallel -j 1 --bar "snn -c ${RL_input}{} -vvv --study --reinforcement " ::: $(seq 5 9);
