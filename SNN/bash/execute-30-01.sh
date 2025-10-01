#!/bin/bash

LC_NUMERIC=C

echo "SelfLearning"
sl_input="Conf/TestAccuracy/SL-multiRNG-OTTAcc-MNOcm/SelfCurriculumLearning-"
sl_output="results-SNN2/TestAccuracy/SelfCurriculumLearning/SelfLearning-"
parallel -j 1 --bar "./bash/reRun_same_hash.sh -i ${sl_input}{} -r ${sl_output}{} 2>&1 >/dev/null" ::: $(seq 0 100);

echo "Trivial"
Trivial_input="Conf/TestAccuracy/Trivial-multiRNg-OTTacc-MNOcm/Default-"
Trivial_output="results-SNN2/TestAccuracy/Default/default-"
parallel -j 1 --bar "./bash/reRun_same_hash.sh -i ${Trivial_input}{} -r ${Trivial_output}{} 2>&1 >/dev/null" ::: $(seq 0 100);

echo "SL+RL re-run"
RL_input="Conf/Final_comparison/SL_plus_OTTRL-multiExp-Entropy-TrnFrz/SL_plus_OTTRL-oneShot-"
RL_output="results-final/SL_plus_OTTRL-multiExp-Entropy-TrnFrz/exp-"
parallel -j 1 --bar "./bash/reRun_same_hash_reinforcement.sh -i ${RL_input}{} -r ${RL_output}{}" ::: $(seq 0 65);

echo "SL+RL"
RL_input="Conf/Final_comparison/SL_plus_OTTRL-multiExp-Entropy-TrnFrz/SL_plus_OTTRL-oneShot-"
parallel -j 1 --bar "snn -c ${RL_input}{} --reinforcement -vvvvvv" ::: $(seq 65 100);
