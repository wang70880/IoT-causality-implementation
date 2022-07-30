#!/bin/sh
declare -a bk_array=(1)
declare -a partition_array=(30)
declare -a datasets=("hh130")

data_preprocessing() { # Data preprocessing
    dataset=$1; verbosity=$2
    python preprocessing.py ${dataset} ${verbosity}
}

#causalIoT() {
#    dataset = $1
#    bk_level = $2
#    data_preprocessing ${dataset}
#    for dataset in ${datasets[@]}; do
#        data_preprocessing ${dataset}
#        for bk in ${bk_array[@]}; do
#            for partition in ${partition_array[@]}; do
#                mpiexec -n 8 python -u causalIoT.py ${dataset} ${partition} ${bk} &>> output.txt </dev/null
#            done
#        done
#    done
#}


# 0. Cleanup process and parameter settings
rm output.txt; touch output.txt
dataset="hh130"; preprocessing_verbosity=0
partition_days=30; training_ratio=0.9
# 1. Initiate data preprocessing to generate the sanitized data file
data_preprocessing $dataset $preprocessing_verbosity
# 2. Initiate causal discovery process
mpiexec -n 1 python -u causal_discovery.py ${dataset} ${partition_days} ${training_ratio} &>> output.txt </dev/null
