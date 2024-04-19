#!/bin/bash

appendix="BigTest"
output="01-08-2022-FirstRLTest"
result_folder="results/RL-GrayInf-implemented"
plots_path="${result_folder}/plots"
output_path="${plots_path}/${output}"
pkl_path="${result_folder}/pkl"
register="${pkl_path}/RL-AC-Register_${appendix}.pkl"
test_reg="${pkl_path}/RL-AC-Test_reg_${appendix}.pkl"
loss_reg="${pkl_path}/RL-AC-Loss_reg_${appendix}.pkl"
dictionary="${pkl_path}/RL-AC-State_dictionary_${appendix}.pkl"
test=0
steps=200
tsteps=200
interleave=3
tinterleave=2
evolve=1
tevolve=1
compress=true
security=false
splitted=false

# Check if the output path already exists
if [[ -d ${output_path} ]]
then
		echo "${output_path} exists it must be empty."
		if [[ "$(ls -A ${output_path})" ]] && ${security}
		then
				echo "${output_path} Not Empty"
				exit 1
		fi
else
		echo "${output_path} does not exists."
		echo "Creating ${output_path}."
		mkdir ${output_path}
fi

echo "----- Start plotting -----"
if ${splitted}
then
		python3 SNN/src/plot/rl_plotter.py -p ${register} -d ${dictionary} -l ${loss_reg} -o ${output_path} --interleave ${interleave} --steps ${steps} --evolve ${evolve} --split -t ${test} --tsteps ${tsteps} --tinterleave ${tinterleave} --tevolve ${tevolve} --treg ${test_reg}
else
		python3 SNN/src/plot/rl_plotter.py -p ${register} -d ${dictionary} -l ${loss_reg} -o ${output_path} --interleave ${interleave} --steps ${steps} --evolve ${evolve} -t ${test} --tsteps ${tsteps} --tinterleave ${tinterleave} --tevolve ${tevolve} --treg ${test_reg}
fi

if $compress
then
		echo "----- Compressing the folder -----"
		tar -cvf ${output_path}.tar -C ${plots_path} ${output}
		xz -z 9 -e -v ${output_path}.tar
fi

