#!/bin/bash

LC_NUMERIC=C

input_folder="/dev/null"
result_folder="/tmp/"

while getopts "i:r:" OPTNAME
do
    case $OPTNAME in
		i) input_folder=$OPTARG;;
		r) result_folder=$OPTARG;;
    esac
done

obtain_hash () {
		fname="${1}/ckpt/net_weight*"
		fname=$(echo ${fname})
		fname="${fname##*/}"                  # remove everything before the last /
		fname="${fname##*-}"                  # remove everything before the last /
		h="${fname%\.*}"                  # remove everything before the last /
		echo ${h}
}

export -f obtain_hash

hash=$(obtain_hash ${result_folder})
snn -c ${input_folder} -vvvvvv -H ${hash} -r /dev/null --reinforcement
