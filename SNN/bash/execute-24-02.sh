#!/bin/bash

LC_NUMERIC=C

#echo "Trivial"
#t_input="Conf/final-afterBug/trivial-multiExp/Trivial-oneShot-"
#parallel -j 1 --bar "snn -c ${t_input}{} --study -vvvvvv" ::: $(seq 0 3);

#echo "SelfLearning"
#sl_input="Conf/final-afterBug/SL-multiExp/SL-oneShot-"
#parallel -j 1 --bar "snn -c ${sl_input}{} --study -vvvvvv" ::: $(seq 0 3);

echo "SL+RL"
RL_input="Conf/final-afterBug/SL_plus_OTTRL-multiExp/SL_plus_OTTRL-oneShot-"
parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement -vvvvvv" ::: $(seq 0 100);
