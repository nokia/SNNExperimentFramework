#!/bin/bash

LC_NUMERIC=C

root="results-SNN2/RL2/Exploration"
appendix="test"
output="results-SNN2/RL2/Exploration"
TrnFrz=false
TrnFrz_dim=10
remove=false
rl=true

while getopts "f:a:o:td:rR" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		a) appendix=$OPTARG;;
		o) output=$OPTARG;;
		t) TrnFrz=true;;
		d) TrnFrz_dim=$OPTARG;;
		r) remove=true;;
		R) rl=false;;
    esac
done

if [ $remove ] && find  ${root} -name 'merged_*' | grep -q '.'; then
		for file in ${root}/merged_*
		do
				dst="/tmp/${file//\//_}_$(date '+%d-%m-%Y_%H-%M-%S')"
				mv ${file} ${dst}
		done
fi

echo "HERE 1!!!!!!!!!!!!!!!!!!!!!"
./bash/merge_ott_stats.sh -f ${root} -o ${output} -a ${appendix} -c
echo "HERE 2!!!!!!!!!!!!!!!!!!!!!"
./bash/merge_gray_stats.sh -f ${root} -o ${output} -a ${appendix} -c
echo "HERE 3!!!!!!!!!!!!!!!!!!!!!"
if $rl; then
		./bash/merge_reinforce_stats.sh -f ${root} -o ${output} -a ${appendix} -c
fi
echo "HERE 4!!!!!!!!!!!!!!!!!!!!!"

sed -i -e 's/distance_accuracy/MNO\_Accuracy/' ${root}/merged_evaluation_stats_${appendix}.csv
sed -i -e 's/OTT_accuracy/OTT\_Accuracy/' ${root}/merged_ott_stats_${appendix}.csv

awk 'FNR==1 && NR!=1{next;}{print}' ${root}/merged_evaluation_stats_${appendix}.csv ${root}/merged_mno_cm_${appendix}.csv ${root}/merged_ott_stats_${appendix}.csv >> ${root}/merged_evaluation_full.csv

if $TrnFrz; then
		#./bash/apply_trnfrz.sh -f ${root}/merged_grays_stats_${appendix}.csv -n ${TrnFrz_dim}
		python3 py/RL_apply_trnfrz.py -f ${root}/merged_grays_stats_${appendix}.csv -n ${TrnFrz_dim} -c Evaluation
		if $rl; then
				python3 py/RL_apply_trnfrz.py -f ${root}/merged_reinforcement_step_${appendix}.csv -n ${TrnFrz_dim}
				python3 py/evolution_apply_cycles.py -f ${root}/merged_evolution_stats_${appendix}.csv -n 10 -r 2
		else
				python3 py/evolution_apply_cycles.py -f ${root}/merged_evolution_stats_${appendix}.csv -n 0 -r 2
		fi
fi
