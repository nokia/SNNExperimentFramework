#!/bin/bash

LC_NUMERIC=C

root="results-SNN2/MultiGrayDefault"
appendix="test"
output="results-SNN2/MultiGrayDefault"
clean=false

while getopts "f:a:o:c" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		a) appendix=$OPTARG;;
		o) output=$OPTARG;;
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

output_file="${output}/merged_grays_stats_${appendix}.csv"
for folder in ${root}/*
do
		if [[ -d ${folder} ]]; then
				file="${folder}/csv/grays_stats_${appendix}.csv"
				#cut -d',' -f3- ${file} > ${file}_tmp && mv ${file}_tmp ${file}
				gawk '{print (NR==1?"Evaluation":n++)","$0}' ${file} > "${file}_awk.csv"
				cp ${folder}/csv/net_evaluation_stats_* ${folder}/csv/net_evaluation_stats_${appendix}.csv_awk.csv
				cp ${folder}/csv/net_evolution_* ${folder}/csv/net_evolution_stats_${appendix}.csv_awk.csv

				#cp ${folder}/csv/mno_cm_* ${folder}/csv/mno_cm_${appendix}.full.csv_awk.csv

				if [ $(awk -F ',' "{print NF; exit}" ${folder}/csv/mno_cm_*) -eq 3 ]; then
						last=$(tail -n 1 ${folder}/csv/mno_cm_* | awk -F ',' '{print $1}')
						awk -F ',' -v id=${last} 'NR==1 {print $2","$3}; { if($1 == id && $2 != "Accuracy") {print $2","$3} }' ${folder}/csv/mno_cm_* > ${folder}/csv/mno_cm_${appendix}.csv_awk.csv
				else
						cp ${folder}/csv/mno_cm_* ${folder}/csv/mno_cm_${appendix}.csv_awk.csv
				fi

		fi
done

merge ${output_file} ${root} "csv" "grays_stats_" "_awk.csv"


output_file="${output}/merged_evolution_stats_${appendix}.csv"
merge ${output_file} ${root} "csv" "net_evolution_*" "_awk.csv"
output_file="${output}/merged_evaluation_stats_${appendix}.csv"
merge ${output_file} ${root} "csv" "net_evaluation_stats_" "_awk.csv"
output_file="${output}/merged_mno_cm_${appendix}.csv"
merge ${output_file} ${root} "csv" "mno_cm_" "_awk.csv"

if $clean; then
		for folder in ${root}/*
		do
				if [[ -d ${folder} ]]; then
						file="${folder}/csv/grays_stats_${appendix}.csv_awk.csv"
						mv ${file} /tmp/
						file="${folder}/csv/net_evaluation_stats_${appendix}.csv_awk.csv"
						mv ${file} /tmp/
						file="${folder}/csv/net_evolution_*${appendix}.csv_awk.csv"
						mv ${file} /tmp/
						file="${folder}/csv/mno_cm_${appendix}.csv_awk.csv"
						mv ${file} /tmp/
				fi
		done
fi
