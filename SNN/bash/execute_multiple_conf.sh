#!/bin/bash

#echo "Remember to run only sigmoid from a certain point"
#exit 1

LC_NUMERIC=C
#names=("fixStart" "fix" "disabled" "sigmoid")
#names=("sigmoid")
#bad=$(seq 0.01 0.01 0.2)
#rng=$(seq 0 1 10)
names=("KickCV")
kick=$(seq 17.0 0.5 20.0)
cv=$(seq 0.5 0.5 20.0)

execute_exp() {
		#NUM=1.
		#BAD=${2}
		#good=$(awk '{print $1-$2}' <<<"${NUM} ${BAD}")
		base_folder="Conf/H264/ParTest"
		new_appendix="-parTest"
		new_file_name=${1}${new_appendix}${2}-${3}.yaml
		new_file_path=${base_folder}/${1}/${new_file_name}
		snn -c ${new_file_path} > /dev/null 2> /dev/null
}

export -f execute_exp

parallel --jobs 1 --bar execute_exp ::: ${names[*]} ::: ${kick[*]} ::: ${cv[*]}
