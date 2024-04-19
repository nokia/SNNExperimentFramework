#!/bin/bash

LC_NUMERIC=C

root="results-SNN2/MultiGrayFreeze"
conf="Conf/Dkr5G/GrayFrozen"
dst="Conf/Dkr5G-grayFrozen"

for folder in ${root}/*
do
		if [[ -d ${folder} ]]; then
				name=$(basename ${folder})
				out=${dst}/frozen-${name}
				cp -r ${conf} ${out}
				sed -i "s/{!datetime}/${name}/" ${out}/io.ini
				sed -i "s/, 'saveEmbeddings'//" ${out}/experiment.ini
		fi
done
