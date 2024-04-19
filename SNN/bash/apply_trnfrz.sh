#!/bin/bash

LC_NUMERIC=C

file="/dev/null"
n=10

while getopts "f:n:" OPTNAME
do
    case $OPTNAME in
		f) file=$OPTARG;;
		n) n=$OPTARG;;
    esac
done

trnfrz () {
		local f=$1
		local n=$2

		if (( $n == 0 )); then
				gawk --include inplace -v INPLACE_SUFFIX=.bak -v s="NO_RL" '{print (NR==1?"Phase":s)","$0}' ${file}
		else
				gawk --include inplace -v INPLACE_SUFFIX=.bak -v n="${n}" -v s="RL_Exploitation" '{if ((NR-2)%n==0 && NR!=1) {if (s=="RL_Training") {s="RL_Exploitation"} else {s="RL_Training"}}; print (NR==1?"Phase":s)","$0}' ${file}
		fi
}

trnfrz ${file} ${n}

