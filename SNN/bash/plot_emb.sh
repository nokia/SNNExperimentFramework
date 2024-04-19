#!/bin/sh

folder="results-noDifficult/Trivial-lessFeatures-2"
iteration="0"
appendix="good-bad-Test_difficult-lessFeatures-2"
output="/dev/null"

while getopts "f:a:i:o:" OPTNAME
do
    case $OPTNAME in
		f) folder=$OPTARG;;
		a) appendix=$OPTARG;;
		i) iteration=$OPTARG;;
		o) output=$OPTARG;;
    esac
done

python3 py/embeddingTSNE.py -f ${folder}/pkl/embedding_goods_iteration_${iteration}_*.pkl \
						       ${folder}/pkl/embedding_bads_iteration_${iteration}_*.pkl  \
							   ${folder}/pkl/embedding_difficult_iteration_${iteration}_*.pkl  \
						    -l Good Bad Difficult 										  \
							-o ${output}/                                            	  \
							-x ${appendix}                                                \
							--origin ${folder}/pkl/object_goods_exp_l_iteration_${iteration}_*.pkl               \
								     ${folder}/pkl/object_bads_exp_l_iteration_${iteration}_*.pkl                \
									 ${folder}/pkl/object_difficult_exp_l_iteration_${iteration}_*.pkl                \
						    --subselect All All Label

