#!/bin/bash

# vmaf --reference result/server_result/local_30.yuv --distorted result/client_result/udp_test_360p_dv_loss0.yuv
# --width 320 --height 180 --pixel_format 420 --bitdepth 8 --feature psnr
# --feature psnr_hvs --feature float_ssim --feature float_ms_ssim
# --feature ciede --output upd_360_loss25.csv --csv

original=originl.yuv
disturbed=disturbed.yuv
output=output.csv
width=1920
height=1080
model=~/Downloads/VMAF/vmaf-2.1.1/model/vmaf_v0.6.1.json
proc=1

while getopts "f:d:o:w:h:m:p:" OPTNAME
do
   case $OPTNAME in
      f) original=$OPTARG;;
      d) disturbed=$OPTARG;;
      o) output=$OPTARG;;
      w) width=$OPTARG;;
      h) height=$OPTARG;;
      m) model=$OPTARG;;
	  p) proc=$OPTARG;;
   esac
done

if [ ! -f $original ]
then
    echo ""$original" does not exist"
	exit 1
fi

if [ ! -f $disturbed ]
then
    echo ""$disturbed" does not exist"
	exit 2
fi

if [ -f $output ]
then
    echo ""$output" already exist"
	exit 3
fi

frameRate=30

ffmpeg -r ${frameRate} -i $original \
-r ${frameRate} -i $disturbed \
-filter_complex "[0:v]scale=${width}:${height}:flags=bicubic,setpts=PTS-STARTPTS[reference]; \
[1:v]scale=${width}:${height}:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
[distorted][reference]libvmaf=log_fmt=csv:log_path=${output}:model_path=${model}:n_threads=${proc}:n_subsample=1" \
-f null - 2>&1 | grep "VMAF score"

#ffmpeg -r 30 -i $original \
#-r 30 -i $disturbed \
#-filter_complex "[0:v]setpts=PTS-STARTPTS[reference]; \
#[1:v]setpts=PTS-STARTPTS[distorted]; \
#[distorted][reference]libvmaf=log_fmt=csv:log_path="$output":model_path="$model":n_threads=4" \
#-f null - 2>&1 | grep "VMAF score"
