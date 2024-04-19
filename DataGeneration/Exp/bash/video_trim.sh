#!/bin/bash

LC_NUMERIC=C

reference="none"
original="none"
frames=30
save="output"

while getopts "r:o:f:s:" OPTNAME
do
    case $OPTNAME in
		r) reference=$OPTARG;;
		o) original=$OPTARG;;
		f) frames=$OPTARG;;
		s) save=$OPTARG;;
    esac
done

if [ ! -f $reference ]
then
		echo "${reference} does not exist"
  		exit 1
fi

if [ ! -f $original ]
then
  	    echo "${original} does not exist"
  		exit 2
fi

if [ -f $save ]
then
  	    echo "${save} already exist"
  		exit 3
fi

# Get the duration of the reference video
ref_duration=$(ffprobe -i ${reference} -show_entries format=duration -v quiet | grep "duration" | awk -F '=' '{print $2}')
echo "Reference duration: ${ref_duration}"

# Get the duration of the original file
org_duration=$(ffprobe -i ${original} -show_entries format=duration -v quiet | grep "duration" | awk -F '=' '{print $2}')
echo "Original duration: ${org_duration}"

# Get the time difference
diff=$(echo "${org_duration} -${ref_duration}" | bc)
echo "Difference: ${diff}"

if [[ "$(echo "${diff} < 0" | bc)" = 1 ]]
then
		echo "${original} lower duration than ${reference} (${org_duration} vs ${ref_duration})"
		exit 4
fi

# Get the first I frame position in the reference file
i_frame=$(ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time,flags -of csv=print_section=0 ${reference} | awk -F ',' '/K/ {print $1}' | head -n 1)
echo "First i frame position: ${i_frame}"

# Get the precise time to cut
cut_time=$(echo "${diff} + ${i_frame}" | bc)
cut_time=$(printf '%.*f\n' 4 ${cut_time})
echo "Cut time: ${cut_time}"

# get the frames list
frames=$(ffprobe -loglevel error -select_streams v:0 -show_entries packet=pts_time -of csv=print_section=0 ${original} | awk -v d=${cut_time} '{print $1,$1-d}' | awk -v T=0.0 '{min = 1000} {if ($2 > T) {if ($2 < min) min=$2;}} END {print min}')
echo "${frames}"

# Trim the video
ffmpeg -v error -ss ${cut_time} -i ${original} -c:v copy ${save}
