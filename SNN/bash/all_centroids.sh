#!/bin/bash

./bash/centroids.sh -i results-SNN2/tmp/default-tmp -o results-SNN2/tmp/default-tmp/plot -x training_dst_margin0,5
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin1 -o results-SNN2/tmp/default-tmp-margin1/plot -x training_dst_margin1
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin5 -o results-SNN2/tmp/default-tmp-margin5/plot -x training_dst_margin5
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin10 -o results-SNN2/tmp/default-tmp-margin10/plot -x training_dst_margin10
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin50 -o results-SNN2/tmp/default-tmp-margin50/plot -x training_dst_margin50
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin100 -o results-SNN2/tmp/default-tmp-margin100/plot -x training_dst_margin100
./bash/centroids.sh -i results-SNN2/tmp/default-tmp -o results-SNN2/tmp/default-tmp/plot -x validation_dst_margin0,5 -d validation
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin1 -o results-SNN2/tmp/default-tmp-margin1/plot -x validation_dst_margin1 -d validation
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin5 -o results-SNN2/tmp/default-tmp-margin5/plot -x validation_dst_margin5 -d validation
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin10 -o results-SNN2/tmp/default-tmp-margin10/plot -x validation_dst_margin10 -d validation
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin50 -o results-SNN2/tmp/default-tmp-margin50/plot -x validation_dst_margin50 -d validation
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin100 -o results-SNN2/tmp/default-tmp-margin100/plot -x validation_dst_margin100 -d validation
./bash/centroids.sh -i results-SNN2/tmp/default-tmp -o results-SNN2/tmp/default-tmp/plot -x test_dst_margin0,5 -d test
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin1 -o results-SNN2/tmp/default-tmp-margin1/plot -x test_dst_margin1 -d test
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin5 -o results-SNN2/tmp/default-tmp-margin5/plot -x test_dst_margin5 -d test
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin10 -o results-SNN2/tmp/default-tmp-margin10/plot -x test_dst_margin10 -d test
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin50 -o results-SNN2/tmp/default-tmp-margin50/plot -x test_dst_margin50 -d test
./bash/centroids.sh -i results-SNN2/tmp/default-tmp-margin100 -o results-SNN2/tmp/default-tmp-margin100/plot -x test_dst_margin100 -d test
cp results-SNN2/tmp/default-tmp*/plot/* results-SNN2/tmp/violinPlot/
