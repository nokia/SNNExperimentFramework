#!/bin/bash

LC_NUMERIC=C

appendix="test"
output="/dev/null"
remove=false
rl=true
vmaf=80

while getopts "a:o:v:rR" OPTNAME
do
    case $OPTNAME in
		a) appendix=$OPTARG;;
		o) output=$OPTARG;;
		v) vmaf=$OPTARG;;
		r) remove=true;;
		R) rl=false;;
    esac
done

shift $((OPTIND - 1))

# Check if the output path already exists
if [[ -d ${output} ]]; then
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

echo "--------- Labels ----------"
labels=("Baseline" "SL" "ATELIER")
#for dir in "${@}"; do
#        printf "Enter label for ${dir}: "
#        read var
#        labels+=($var)
#done

merge () {
		local input_file=$1
		local output_file=$2
		local vmaf=$3
		shift $((OPTIND - $4))
		local folders=$@

		local awk_suffix="_awk.csv"
		local eval_files=()
		local awk_files=()
		local local_labels=("${labels[@]}")

		for d in $folders; do
				d_eval_file="${d}/${input_file}"
				if [ -f "${d_eval_file}" ]; then
						eval_files+=($d_eval_file)
				else
						echo "${d_eval_file} doesn't exists"
						local_labels=(${local_labels[@]:1})
				fi
		done

		for i in "${!eval_files[@]}"; do
				file=${eval_files[i]}
				awk_file="${file}${awk_suffix}"
				awk_files+=($awk_file)
				label=${local_labels[i]}
				cp ${file} ${awk_file}
				gawk --include inplace -v INPLACE_SUFFIX=.bak -v i="${label}" -v vmaf="${vmaf}" '{print (NR==1?"VMAF,Origin":vmaf","i)","$0}' ${awk_file}
		done

		awk 'FNR==1 && NR!=1{next;}{print}' "${awk_files[@]}" >> ${output}/${output_file}

		for f in "${awk_files[@]}"; do
				rm ${f}
		done
}

echo "--------- Evaluation full ----------"

eval_file="merged_evolution_stats_${appendix}.csv"
eval_output_file="evolution.csv"
merge $eval_file $eval_output_file $vmaf 2 $@

eval_file="merged_evaluation_full.csv"
eval_output_file="evaluation.csv"
merge $eval_file $eval_output_file $vmaf 2 $@

gray_file="merged_grays_stats_${appendix}.csv"
gray_output_file="difficult.csv"
merge $gray_file $gray_output_file $vmaf 2 $@

#rl_file="merged_reinforcement_step_${appendix}.csv"
#rl_output_file="reinforcement.csv"
#merge $rl_file $rl_output_file 1 $@
