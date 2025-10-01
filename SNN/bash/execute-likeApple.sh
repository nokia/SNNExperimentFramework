#!/bin/bash

LC_NUMERIC=C

main_folder="Conf/likeApple/"

echo "SL+RL-Zone"
RL_input="${main_folder}/multiExp/RL-10-5btc-3actions-"
parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 100);

