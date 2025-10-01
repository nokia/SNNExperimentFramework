#!/bin/bash

LC_NUMERIC=C

main_folder="Conf/bunny-slicing"

#echo "Trivial"
#t_input="${main_folder}/trivial-multiExp/trivial-"
#parallel -j 1 --bar "snn -c ${t_input}{} --study " ::: $(seq 0 100);

#echo "SelfLearning"
#sl_input="${main_folder}/SL-multiExp/SL-"
#parallel -j 1 --bar "snn -c ${sl_input}{} --study " ::: $(seq 0 100);

#echo "SL+RL-zone"
#RL_input="${main_folder}/SL-OTTRL-zone-multiExp/SL-OTTRL-zone-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 15);

#echo "SL+RL-External-RandomReset"
#RL_input="Conf/External-RL/SL-OTTRL-RandomReset-multiExp/SL-OTTRL-RandomReset-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 30);

#echo "SL+RL-External-NoStop"
#RL_input="Conf/External-RL/SL-OTTRL-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 10);

#echo "SL+RL-External-RandomReset-Long"
#RL_input="Conf/External-RL/SL-OTTRL-Rand-Long-multiExp/SL-OTTRL-Rand-Long-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 30);

#echo "SL+RL-External-NoStop-Long"
#RL_input="Conf/External-RL/SL-OTTRL-Long-multiExp/SL-OTTRL-Long-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 30);

#echo "SL+RL-External-Zone"
#RL_input="Conf/External-RL/SL-OTTRL-Zone-multiExp/SL-OTTRL-Zone-"
#parallel -j 1 --bar "snn -c ${RL_input}{} -vvv --study --reinforcement " ::: $(seq 0 100);

#echo "SL+RL-RandomReset"
#RL_input="${main_folder}/SL-OTTRL-RandomReset-multiExp/SL-OTTRL-RandomReset-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 55 100);

#echo "SL+RL-NoReset"
#RL_input="${main_folder}/SL-OTTRL-NoReset-multiExp/SL-OTTRL-NoReset-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 30);

#echo "SL+RL"
#RL_input="${main_folder}/SL-OTTRL-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 14 100);

#echo "SL+RL-External-NoLimit"
#RL_input="Conf/External-RL/SL-OTTRL-noStop-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 42 100);

#echo "SL+RL-External-NoLimit"
#RL_input="Conf/External-RL/SL-OTTRL-noStop0-55-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 52 100);

#echo "SL+RL-External-NoLimit"
#RL_input="Conf/External-RL/SL-OTTRL-noStop0.6-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 100);

#echo "SL+RL-External-NoLimit"
#RL_input="Conf/External-RL/SL-OTTRL-noStop0.65-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 100);

#echo "SL+RL-AfterTrivial"
#RL_input="Conf/External-RL/SL-OTTRL-AfterTrivial-multiExp/SL-OTTRL-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 0 100);

#echo "SL+RL-Zone-tests"
#RL_input="Conf/ZoneRL/RL-Zone-tests/SL-OTTRL-Zone-"
#parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 1 10);

#exit 0

echo "SL+RL-Zone"
RL_input="Conf/ZoneRL/RL-Zone-multiExp/SL-OTTRL-Zone-3-"
parallel -j 1 --bar "snn -c ${RL_input}{} --study --reinforcement " ::: $(seq 5 100);

