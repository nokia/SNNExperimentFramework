#!/bin/bash

LC_NUMERIC=C

input=input.pcap
output=output.pcap
flags=''

while getopts "i:o:f:" OPTNAME
do
    case $OPTNAME in
		i) input="${OPTARG}";;
		o) output="${OPTARG}";;
		f) flags="${OPTARG}";;
    esac
done

su vlcuser -c "echo \"frame,arrival_time,diff_time,frame_len\" > ${output}"
su vlcuser -c "tshark -r ${input} -Y '${flags}' -T fields -E separator=, -e frame.number -e frame.time_relative -e frame.time_delta -e frame.len >> ${output}"
