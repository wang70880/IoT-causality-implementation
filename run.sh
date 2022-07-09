#!/bin/sh
declare -a bk_array=(1)
declare -a partition_array=(30)
declare -a datasets=("hh130")

evaluatePartitionAccuracy() {
    for dataset in ${datasets[@]}; do
        python preprocessing.py ${dataset}
        for bk in ${bk_array[@]}; do
            for partition in ${partition_array[@]}; do
                mpiexec -n 5 python -u causalIoT.py ${dataset} ${partition} ${bk} &>> output.txt </dev/null
            done
        done
    done
}

rm output.txt; touch output.txt
evaluatePartitionAccuracy
