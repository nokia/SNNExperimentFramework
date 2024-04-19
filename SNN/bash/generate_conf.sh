#!/bin/bash

LC_NUMERIC=C
#names=("fixStart" "fix" "disabled" "sigmoid")
names=("KickCV")
kick=$(seq 0.5 0.5 20.0)
cv=$(seq 0.5 0.5 20.0)

parallel --jobs 10 --bar ./bash/new_conf_file.sh ::: ${names[*]} ::: ${kick[*]} ::: ${cv[*]}

