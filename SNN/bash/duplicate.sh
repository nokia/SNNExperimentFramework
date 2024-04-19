#!/bin/bash

LC_NUMERIC=C

input_folder="/dev/null"
output_folder="/tmp/"
rng_value=2
before="/SL-oneShot"
after="/SL-multiExp"

n=50
while getopts "i:o:n:r:b:a:" OPTNAME
do
    case $OPTNAME in
		i) input_folder=$OPTARG;;
		o) output_folder=$OPTARG;;
		n) n=$OPTARG;;
		r) rng_value=$OPTARG;;
		b) before=$OPTARG;;
		a) after=$OPTARG;;
    esac
done
rng_seq=$(seq 0 ${n})

copy () {
		dirname=${1}
		rng_v=${4}
		find_seq=${5}
		sub_seq=${6}
		sourcedir="${dirname%"${dirname##*[!/]}"}" # extglob-free multi-trailing-/ trim
		sourcedir="${sourcedir##*/}"                  # remove everything before the last /
		sourcedir=${sourcedir:-/}                     # correct for dirname=/ case
		cp -r -u ${dirname} ${2}/${sourcedir}-${3}
		sed -i "s|value=${rng_v}|value=${3}|g" ${2}/${sourcedir}-${3}/environment.ini
		#sed -i "s|${find_seq}|${sub_seq}/exp-${3}|g" ${2}/${sourcedir}-${3}/io.ini
		sed -i "s|${find_seq}|${sub_seq}${3}|g" ${2}/${sourcedir}-${3}/io.ini
}


export -f copy

parallel -j 0 --bar "copy ${input_folder} ${output_folder} {} ${rng_value} ${before} ${after}" ::: ${rng_seq}
