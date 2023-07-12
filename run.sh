#!/bin/sh

data_preprocessing() { # Data preprocessing
    python preprocessing.py ${dataset} 100 0.8 ${preprocessing_verbosity}
}

evaluate_discovery_process() {
    for partition_day in ${partition_days[@]}; do
            for pc_alpha in ${pc_alphas[@]}; do
                for max_conds_dim in ${max_conds_dims[@]}; do
                        mpiexec -n 10 python -u causalIoT.py ${dataset} \
                                                           ${partition_day} ${training_ratio} \
                                                           ${tau_max} \
                                                           ${pc_alpha} ${max_conds_dim} ${max_comb} &>> discovery_evaluation.txt </dev/null
                done
            done
    done
}

## Data preprocessing
dataset="hh130"; preprocessing_verbosity=0
## Data loading
declare -a partition_days=(30)
training_ratio=0.8
## Background generator and application level
tau_max=2
## PC discovery process
declare -a pc_alphas=(0.001)
declare -a max_conds_dims=(9)
max_comb=10

# 0. Cleanup process and parameter settings
rm -rf discovery_evaluation.txt; touch discovery_evaluation.txt
# 1. Initiate data preprocessing to generate the sanitized data file
#data_preprocessing
# 2. Initiate causal discovery process
evaluate_discovery_process
