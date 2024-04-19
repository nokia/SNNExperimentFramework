#!/bin/bash

root="results-CoNEXT-50Runs"
#video=("Bunny" "Bottle" "Scarlet")
video=("Bunny")
#vmaf=(80 90 99)
vmaf=(80 90)
date=$(date '+%d-%m-%Y')
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

./bash/merge_vmaf_comparison.sh -i ${root} -d "Bottle" -p ${prefix}
./bash/merge_vmaf_comparison.sh -i ${root} -d "Bunny" -p ${prefix}
#./bash/merge_vmaf_comparison.sh -i ${root} -d "Scarlet" -p ${prefix}

#mv ${root}/${prefix}-Scarlet-${date} ${root}/${prefix}-Scarlet-${date}

for d in Bunny Bottle; do
		./bash/plot_merged.sh -f ${root}/${prefix}-${d}-${date} -o ${root}/${prefix}-${d}-${date}/plot -v ${d} -r
done

for d in Bunny Bottle; do
		cp -u ${root}/${prefix}-${d}-${date}/plot/* ../../paperi/CoNext/figures/plots/${d}/
done
