#!/bin/bash

root="results-CoNEXT"
video="all"
vmaf=99
only_plot=false
no_plot=false
output="None"
rl=false

while getopts "i:a:v:o:pPR" OPTNAME
do
    case $OPTNAME in
		i) root=$OPTARG;;
		a) video=$OPTARG;;
		v) vmaf=$OPTARG;;
		o) output=$OPTARG;;
		p) only_plot=true;;
		R) rl=true;;
		P) no_plot=true;;
    esac
done

if ! ${only_plot}; then
		trivial="${root}/Trivial/${video}-VMAF${vmaf}-multiExp"
		sl="${root}/SL/${video}-VMAF${vmaf}-multiExp"
		if ${rl}; then
				sl_rl="${root}/SL-RL/${video}-VMAF${vmaf}-multiExp"
		fi
		#sl_ottrl_keep="${root}/SL-OTTRL-NoReset-multiExp-correct"
		#sl_ottrl_random="${root}/SL-OTTRL-RandomRest-multiExp"

		./bash/merge.sh -f ${trivial} -o ${trivial} -r -t -d 0 -R
		./bash/merge.sh -f ${sl} -o ${sl} -r -t -d 0 -R
		if ${rl}; then
				./bash/merge.sh -f ${sl_rl} -o ${sl_rl} -r -t -d 10
		fi
		#./bash/merge.sh -f ${sl_ottrl_keep} -o ${sl_ottrl_keep} -r -t -d 10
		#./bash/merge.sh -f ${sl_ottrl_random} -o ${sl_ottrl_random} -r -t -d 10
fi

if [ ${output} == "None" ]; then
		output_folder="${root}/comparison-${video}-VMAF${vmaf}-"$(date '+%d-%m-%Y')
else
		output_folder=${output}
fi

echo ${output_folder}
mkdir -p ${output_folder}/plot

if ! ${only_plot}; then
		if ${rl}; then
				./bash/concatenate_merged.sh -o ${output_folder} -v ${vmaf} -r ${trivial} ${sl} ${sl_rl}
		else
				./bash/concatenate_merged.sh -o ${output_folder} -v ${vmaf} -r ${trivial} ${sl}
		fi
fi

#mkdir -p ${output_folder}/plot
#if ! ${no_plot}; then
#        ./bash/plot_merged.sh -f ${output_folder} -o ${output_folder}/plot -r -v ${video}
#fi
