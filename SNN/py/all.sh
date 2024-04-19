#!/bin/bash

LC_NUMERIC=C

root="results-SNN2/GrayDeeper-study"
iterations=9
appendix="test"

while getopts "f:i:a:" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		i) iterations=$OPTARG;;
		a) appendix=$OPTARG;;
    esac
done

folder="${root}/csv"
pkl="${root}/pkl"
output="${root}/plot/"
train1="${folder}/net_evolution_50ep_${appendix}_train1.csv"
gray_stat="${folder}/grays_stats_${appendix}.csv"
gray_pred="${folder}/grays_predictions_${appendix}.csv"
gray_train="${folder}/net_evolution_50ep_${appendix}.csv"

python3 py/plot.py -f ${train1} -o ${output} -g ${gray_stat} -m ${gray_train} -u
python3 py/grayPredEvaluation.py -f ${gray_pred} -o ${output}

exit 0
for i in $(seq ${iterations});
do
		good_file="${pkl}/embedding_goods_iteration_${i}*.pkl"
		bad_file="${pkl}/embedding_bads_iteration_${i}*.pkl"
		gray_file="${pkl}/embedding_grays_iteration_${i}*.pkl"
		python3 py/embeddingTSNE.py -f ${good_file} ${bad_file} ${gray_file} -l goods bads Undefined -o ${output} -x Iteration${i}
done

