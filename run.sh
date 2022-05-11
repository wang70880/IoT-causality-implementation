#!/bin/sh
declare -a bk_array=(1)
declare -a partition_array=(20)

evaluatePartitionAccuracy() {
    for bk in ${bk_array[@]}; do
        for partition in ${partition_array[@]}; do
            mpiexec -n 35 python -u pc_discovery.py ${partition} ${bk} &>> output.txt </dev/null
        done
    done
}

rm output.txt; touch output.txt
evaluatePartitionAccuracy