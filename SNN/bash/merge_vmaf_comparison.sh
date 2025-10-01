#!/bin/bash

root="results-CoNEXT"
prefix="comparison_withRL"
output="None"
d="Bottle"
date=$(date '+%d-%m-%Y')

while getopts "i:o:d:p:D:" OPTNAME
do
    case $OPTNAME in
		i) root=$OPTARG;;
		o) output=$OPTARG;;
		d) d=$OPTARG;;
		p) prefix=$OPTARG;;
		D) date=$OPTARG;;
    esac
done

output="${root}/${prefix}-${d}-${date}"
mkdir -p ${output}
cp ${root}/${prefix}-${d}-VMAF80-*/* ${output}/
for v in 90 99; do
		for f in ${root}/${prefix}-${d}-VMAF${v}-*/*; do
				if [ -f ${f} ]; then
						source="${f%"${f##*[!/]}"}" 			# extglob-free multi-trailing-/ trim
						source="${source##*/}"                  # remove everything before the last /
						source=${source:-/}                     # correct for dirname=/ case
						tail -n+2 ${f} >> ${output}/${source}
				fi
		done
done
