#!/bin/bash

file="none.mp4"
sdp="sdp_file.sdp"
local_output="local.mp4"
addr="172.16.100.2"
port="8554"
output="log"

while getopts "f:s:l:a:p:o:" OPTNAME
do
   case $OPTNAME in
		f) file=$OPTARG;;
		s) sdp=$OPTARG;;
		l) local_output=$OPTARG;;
		a) addr=$OPTARG;;
		p) port=$OPTARG;;
		o) output=$OPTARG;;
   esac
done

ffmpeg -re -i ${file} \
		-c:v copy -an \
		-flags +global_header \
		-movflags +faststart \
		-fflags +genpts \
		-sdp_file ${sdp} \
		-f tee -map 0:v \
		"${local_output}|[f=rtp]rtp://${addr}:${port}" 2>&1 | tee ${output}_ffmpeg.out
