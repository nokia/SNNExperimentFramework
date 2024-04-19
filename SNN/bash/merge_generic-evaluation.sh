#!/bin/bash

LC_NUMERIC=C

root="results-SNN2/RL2/Exploration"
prefix="merge_generic-evaluation"
appendix="test"
output="results-SNN2/RL2/Exploration"
clean=false
evaluation_file="net_evaluation_stats_"

while getopts "f:p:a:o:e:c" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		p) prefix=$OPTARG;;
		a) appendix=$OPTARG;;
		o) output=$OPTARG;;
		e) evaluation_file=$OPTARG;;
		c) clean=true;
    esac
done

merge () {
		local output_file=$1
		local root=$2
		local sub_folder=$3
		local file_name=$4
		local post_name=$5
		local i=0

		if [[ -f ${output_file} ]]; then
				echo "${output_file} already exists"
				exit 1
		fi

		for folder in ${root}/*
		do
				if [[ -d ${folder} ]]; then
						file="${folder}/${sub_folder}/${file_name}${appendix}.csv${post_name}"

						gawk --include inplace -v INPLACE_SUFFIX=.bak -v i="${i}" '{print (NR==1?"Experiment":i)","$0}' ${file}
						if [[ -f ${output_file} ]]; then
								tail -n+2 ${file} >> ${output_file}
						else
								touch ${output_file}
								cat ${file} >> ${output_file}
						fi

						i=$((${i}+1))
				fi
		done
}

output_file="${output}/${prefix}${appendix}.csv"
for folder in ${root}/*
do
		if [[ -d ${folder} ]]; then
				file="${folder}/csv/${evaluation_file}${appendix}.csv"
				cp ${file} "${file}_awk.csv"
		fi
done

merge ${output_file} ${root} "csv" ${evaluation_file} "_awk.csv"

if $clean; then
		for folder in ${root}/*
		do
				if [[ -d ${folder} ]]; then
						file="${folder}/csv/${evaluation_file}${appendix}.csv_awk.csv"
						mv ${file} /tmp/
				fi
		done
fi
