#!/bin/bash

root="results-CoNEXT-50Runs"
#video=("Bunny" "Bottle" "ScarletV3")
video=("ScarletV3")
#vmaf=(80 90 99)
vmaf=(80 99)
#date=$(date '+%d-%m-%Y')
date="19-11-2023"
prefix="comparison"

while getopts "i:" OPTNAME
do
    case $OPTNAME in
		i) root=$OPTARG;;
    esac
done

#for d in ${video[@]}; do
#        for v in ${vmaf[@]}; do
#                ./bash/merge_and_plot.sh -i ${root} -a ${d} -v ${v} -R
#        done
#done
#exit

#./bash/merge_vmaf_comparison.sh -i ${root} -d "Bottle" -p ${prefix}
#./bash/merge_vmaf_comparison.sh -i ${root} -d "Bunny" -p ${prefix}
#./bash/merge_vmaf_comparison.sh -i ${root} -d "ScarletV3" -p ${prefix}

#exit
#mv ${root}/${prefix}-ScarletV3-${date} ${root}/${prefix}-Scarlet-${date}
#exit

for d in Scarlet; do
		./bash/plot_merged.sh -f ${root}/${prefix}-${d}-${date} -o ${root}/${prefix}-${d}-${date}/plot -v ${d} -r
done
