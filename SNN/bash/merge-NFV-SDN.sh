#!/bin/bash

LC_NUMERIC=C

root="results-NFV-SDN/multi-exp/baseline"
output="results-NFV-SDN/multi-exp/baseline"
remove=false
appendix="test"
epochs=500

while getopts "f:o:a:e:r" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		o) output=$OPTARG;;
		a) appendix=$OPTARG;;
		e) epochs=$OPTARG;;
		r) remove=true;;
    esac
done

if $remove; then
		for file in ${root}/merged_*
		do
				dst="/tmp/${file//\//_}_$(date '+%d-%m-%Y_%H-%M-%S')"
				mv ${file} ${dst}
		done
else
		if compgen -G "${root}/merged_*" > /dev/null; then
				echo "pattern exists!"
		fi
fi

prefix="merged_net_evaluation_"
evaluation_file="net_evaluation_stats_${epochs}ep_"
./bash/merge_generic-evaluation.sh -f ${root} -o ${output} -a ${appendix} -p ${prefix} -e ${evaluation_file} -c
prefix="merged_net_evolution_"
evaluation_file="net_evolution_${epochs}ep_"
./bash/merge_generic-evaluation.sh -f ${root} -o ${output} -a ${appendix} -p ${prefix} -e ${evaluation_file} -c

