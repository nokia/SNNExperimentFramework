#!/bin/bash

LC_NUMERIC=C

root="results-SNN2/RL2/Exploration"
appendix="test"
output="results-SNN2/RL2/Exploration"
n=9

while getopts "f:a:o:i:" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		a) appendix=$OPTARG;;
		o) output=$OPTARG;;
		i) n=$OPTARG;;
    esac
done

for exp in ${root}/exp-*; do
		sourcedir="${exp%"${exp##*[!/]}"}" 	  # extglob-free multi-trailing-/ trim
		sourcedir="${sourcedir##*/}"                  # remove everything before the last /
		sourcedir=${sourcedir:-/}                     # correct for dirname=/ case
		plt_dir="${output}/${sourcedir}/plot"
		mkdir -p ${plt_dir}
		./bash/plot_emb.sh -f ${exp} -o ${plt_dir} -a ${appendix}-${sourcedir} -i ${n}
		# py/embeddingTSNE.py -fy/embeddingTSNE.py -f results-noDifficult/Trivial-only/pkl/embedding_goods_iteration_0_test3170b1afd2e36ba2.pkl results-noDifficult/Trivial-only/pkl/embedding_bads_iteration_0_test3170b1afd2e36ba2.pkl results-noDifficult/Trivial-only/pkl/embedding_test_iteration_0_test3170b1afd2e36ba2.pkl -l Good Bad Test-Difficult -o results-noDifficult/Trivial-only/plot/ -x good-bad-Test_difficult --origin results-noDifficult/Trivial-only/pkl/object_goods_exp_l_iteration_0_test3170b1afd2e36ba2.pkl results-noDifficult/Trivial-only/pkl/object_bads_exp_l_iteration_0_test3170b1afd2e36ba2.pkl results-noDifficult/Trivial-diffTraining/pkl/object_test_exp_l_iteration_0_tested0c7d56b5c11950.pkl --subselect All All 1; python3 py/embeddingTSNE.py -f results-noDifficult/Trivial-diffTraining/pkl/embedding_goods_iteration_0_tested0c7d56b5c11950.pkl results-noDifficult/Trivial-diffTraining/pkl/embedding_bads_iteration_0_tested0c7d56b5c11950.pkl results-noDifficult/Trivial-diffTraining/pkl/embedding_test_iteration_0_tested0c7d56b5c11950.pkl -l Good Bad Test-Difficult -o results-noDifficult/Trivial-diffTraining/plot/ -x good-bad-Test_difficult-DiffInTraining  --origin results-noDifficult/Trivial-diffTraining/pkl/object_goods_exp_l_iteration_0_tested0c7d56b5c11950.pkl results-noDifficult/Trivial-diffTraining/pkl/object_bads_exp_l_iteration_0_tested0c7d56b5c11950.pkl results-noDifficult/Trivial-diffTraining/pkl/object_test_exp_l_iteration_0_tested0c7d56b5c11950.pkl --subselect All All 1
done

