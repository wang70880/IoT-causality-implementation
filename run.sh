#!/bin/sh
declare -a bk_array=(0 1)
declare -a partition_array=(5 10 15 20 30)

evaluatePartitionAccuracy() {
    for bk in ${bk_array[@]}; do
        for partition in ${partition_array[@]}; do
            eval 'mpiexec -n 35 python -u run_pcmci_parallel.py ${partition} ${bk} &>> output.txt'
        done
    done
}

rm output.txt; touch output.txt
evaluatePartitionAccuracy