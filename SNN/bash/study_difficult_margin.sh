#!/bin/sh

folder="results-noDifficult/Trivial-lessFeatures-2"
iteration="0"
appendix="DifficultMargin-study"

while getopts "f:a:i:" OPTNAME
do
    case $OPTNAME in
		f) folder=$OPTARG;;
		a) appendix=$OPTARG;;
		i) iteration=$OPTARG;;
    esac
done

python3 py/performanceOnMarginVariation.py -f ${folder}/pkl/embedding_goods_iteration_${iteration}_*.pkl 				\
										      ${folder}/pkl/embedding_difficult_a_iteration_${iteration}_*.pkl 			\
										      ${folder}/pkl/embedding_bads_iteration_${iteration}_*.pkl 				\
						    			   -l Difficult-emb 												   			\
										   -o ${folder}/plot/                                            	   			\
										   -x ${appendix}                                                	   			\
										   --origin ${folder}/pkl/object_difficult_origin_iteration_${iteration}_*.pkl 	\
										   --expL ${folder}/pkl/object_difficult_triplets_exp_l_iteration_${iteration}_*.pkl -O

