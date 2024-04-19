#!/bin/bash

LC_NUMERIC=C

root="/dev/null"
output="/dev/null"
video_name="VideoName"
remove=false

while getopts "f:a:o:v:tr" OPTNAME
do
    case $OPTNAME in
		f) root=$OPTARG;;
		o) output=$OPTARG;;
		v) video_name=$OPTARG;;
		r) remove=true;;
    esac
done

# Check if the output path already exists
if [[ -d ${output} ]]
then
		if [[ "$(ls -A ${output})" ]] && ! ${remove}; then
				echo "${output} Not Empty"
				exit 1
		elif [[ "$(ls -A ${output})" ]] && ${remove}; then
				for file in ${output}/*; do
						dst="/tmp/${file//\//_}_$(date '+%d-%m-%Y_%H-%M-%S')"
						mv ${file} ${dst}
				done
		fi

else
		echo "${output} does not exists; Creating ${output}"
		mkdir ${output}
fi

echo "----- Start plotting -----"

#deactivate
#source env/bin/activate
python3 py/grayComparison.py -f ${root}/difficult.csv -u --multiExperiment -o ${output} -e ${root}/evaluation.csv -p --evolution ${root}/evolution.csv -t ${video_name}
#deactivate
#python3 py/MultiExp_reinforceStudy.py -f ${root}/reinforcement.csv -o ${output} -p
