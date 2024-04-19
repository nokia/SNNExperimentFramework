#!/bin/bash

root="results-CoNEXT"
#video=("Bottle" "Bunny" "Football" "Scarlet")
video=("Bunny" "Scarlet")
approaches=("Trivial" "SL" "SL-RL")
vmaf=(80 90 99)

while getopts "i:" OPTNAME
do
    case $OPTNAME in
		i) root=$OPTARG;;
    esac
done

output=${root}/UMAP-plots
mkdir -p ${output}

for a in ${approaches[@]}; do
		for d in ${video[@]}; do
		        for v in ${vmaf[@]}; do
						dir=${root}/${a}/${d}-VMAF${v}-multiExp
						for e in ${dir}/exp-*; do
								sourcedir="${e%"${e##*[!/]}"}" 	  # extglob-free multi-trailing-/ trim
								sourcedir="${sourcedir##*/}"                  # remove everything before the last /
								sourcedir=${sourcedir:-/}                     # correct for dirname=/ case
								i=$(($(ls -la ${e}/pkl/ | grep "iteration" | wc -l)/30-1))
								#echo ${d}-${a}-VMAF${v}-${sourcedir}: ${i}
								./bash/plot_emb.sh -f ${e} -o ${output} -a ${d}-${a}-VMAF${v}-${sourcedir} -i ${i}
						done
		        done
		done
done

