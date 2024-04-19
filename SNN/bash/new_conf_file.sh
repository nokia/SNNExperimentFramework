#!/bin/bash

base_folder="Conf/H264/ParTest"
base_appendix="base"
new_appendix="-parTest"

#NUM=1.
#BAD=${2}
#good=$(awk '{print $1-$2}' <<<"${NUM} ${BAD}")
new_file_name=${1}${new_appendix}${2}-${3}.yaml
new_file_path=${base_folder}/${1}/${new_file_name}
#cp ${base_folder}/${1}/${1}${base_appendix}.yaml ${new_file_path};
cp ${base_folder}/${1}/${base_appendix}.yaml ${new_file_path};
sed -i "s|kick_strength: 1.|kick_strength: $2|g" ${new_file_path};
sed -i "s|cv_threshold: 1.|cv_threshold: $3|g" ${new_file_path};
sed -i "s|-base\"|-kick$2-cv$3\"|g" ${new_file_path};
#sed -i "s|rng: 1|rng: $3|g" ${new_file_path};

