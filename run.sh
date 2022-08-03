#!/bin/sh
declare -a bk_array=(1)
declare -a partition_array=(30)
declare -a datasets=("hh130")

data_preprocessing() { # Data preprocessing
    python preprocessing.py ${dataset} ${preprocessing_verbosity}
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

evaluate_discovery_process() {
    rm -rf discovery_evaluation.txt; touch discovery_evaluation.txt
    for bk_level in ${bk_levels[@]}; do
        for pc_alpha in ${pc_alphas[@]}; do
            mpiexec -n 8 python -u causalIoT.py ${dataset} \
                                               ${partition_days} ${training_ratio} \
                                               ${tau_max} ${filter_threshold} ${bk_level} \
                                               ${pc_alpha} ${max_conds_dim} ${max_comb} &>> discovery_evaluation.txt </dev/null
        done
    done

}


## Data preprocessing
dataset="hh130"; preprocessing_verbosity=0
## Data loading
partition_days=100; training_ratio=0.8
## Background generator and application level
tau_max=3; filter_threshold=100
declare -a bk_levels=(0 1 2)
## PC discovery process
declare -a pc_alphas=(0.001 0.005 0.01 0.05 0.1)
max_conds_dim=5; max_comb=10

# 0. Cleanup process and parameter settings
rm -rf output.txt; touch output.txt
# 1. Initiate data preprocessing to generate the sanitized data file
data_preprocessing
# 2. Initiate causal discovery process
evaluate_discovery_process
