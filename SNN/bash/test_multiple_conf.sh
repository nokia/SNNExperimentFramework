#!/bin/bash

LC_NUMERIC=C
names=("fixStart" "fix" "disabled" "sigmoid")
bad=$(seq 0.01 0.01 0.2)
rng=$(seq 0 1 10)

execute_exp() {
		NUM=1.
		BAD=${2}
		good=$(awk '{print $1-$2}' <<<"${NUM} ${BAD}")
		base_folder="Conf/H264/GoodVSBad"
		new_appendix="Kick-Bad"
		output_file="goodVSbad.csv"
		new_file_name=${1}${new_appendix}${2}-${3}.yaml
		new_file_path=${base_folder}/${1}/${new_file_name}
		snn -c ${new_file_path} -t --silent --csv ${output_file}
}

export -f execute_exp

parallel --jobs 1 --bar execute_exp ::: ${names[*]} ::: ${bad[*]} ::: ${rng[*]}
