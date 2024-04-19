#!/bin/bash

LC_NUMERIC=C

input_folder=/dev/null
output_folder="/tmp/"
appendix=""
dataset="training"

#for dir in "$@"
#do
#        input_folder+=("${dir}")
#done

while getopts "i:o:x:d:" OPTNAME
do
    case $OPTNAME in
		i) input_folder=$OPTARG;;
		o) output_folder=$OPTARG;;
		x) appendix=$OPTARG;;
		d) dataset=$OPTARG;;
    esac
done

pkl_folder="${input_folder}/pkl"
csv_folder="${input_folder}/csv"
emb="${pkl_folder}/embedding"
obj="${pkl_folder}/object"
training_triplet_emb="${emb}_${dataset}_triplets"
training_triplet_obj="${obj}_${dataset}_triplets"
iteration=0
positive_file=${training_triplet_emb}_positive_iteration_${iteration}_*.pkl
anchor_file=${training_triplet_emb}_anchor_iteration_${iteration}_*.pkl
negative_file=${training_triplet_emb}_negative_iteration_${iteration}_*.pkl
class_file=${training_triplet_obj}_classes_iteration_${iteration}_*.pkl

sample_centr_csv="${csv_folder}/sample_vs_centroid_distances${appendix}.csv"

python3 py/centroidsStudy.py -p ${positive_file} -a ${anchor_file} -n ${negative_file} -o ${csv_folder} -x ${appendix} -c ${class_file}
python3 py/centroidPlot.py -f ${sample_centr_csv} -o ${output_folder} -x ${appendix}

