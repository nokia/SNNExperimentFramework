#!/bin/bash

LC_NUMERIC=C

echo "SelfLearning"
sl_input="Conf/Final_comparison/SL-multiExp-reset/SL-oneShot-"
parallel -j 1 --bar "snn -c ${sl_input}{} --reinforcement -vvvvvv" ::: $(seq 0 2);

echo "Trivial"
Trivial_input="Conf/Final_comparison/Trivial-multiExp-reset/Trivial-oneShot-"
parallel -j 1 --bar "snn -c ${Trivial_input}{} --reinforcement -vvvvvv" ::: $(seq 0 2);

echo "SL+RL"
RL_input="Conf/Final_comparison/SL_plus_OTTRL-multiExp-Entropy-TrnFrz-reset/SL_plus_OTTRL-oneShot-"
parallel -j 1 --bar "snn -c ${RL_input}{} --reinforcement -vvvvvv" ::: $(seq 0 100);
