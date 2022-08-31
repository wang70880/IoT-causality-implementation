#!/bin/sh

data_preprocessing() { # Data preprocessing
    python preprocessing.py ${dataset} ${preprocessing_verbosity}
}

evaluate_discovery_process() {
    rm -rf discovery_evaluation.txt; touch discovery_evaluation.txt
    for partition_day in ${partition_days[@]}; do
        for bk_level in ${bk_levels[@]}; do
            for pc_alpha in ${pc_alphas[@]}; do
                for filter_threshold in ${filter_thresholds[@]}; do
                    mpiexec -n 8 python -u causalIoT.py ${dataset} \
                                                       ${partition_day} ${training_ratio} \
                                                       ${tau_max} ${partition_day} ${bk_level} \
                                                       ${pc_alpha} ${max_conds_dim} ${max_comb} &>> discovery_evaluation.txt </dev/null
                done
            done
        done
    done
}

## Data preprocessing
dataset="hh130"; preprocessing_verbosity=0
## Data loading
declare -a partition_days=(100)
training_ratio=0.8
## Background generator and application level
tau_max=3
declare -a bk_levels=(0)
declare -a filter_thresholds=(100)
## PC discovery process
declare -a pc_alphas=(0.01)
max_conds_dim=5; max_comb=10

# 0. Cleanup process and parameter settings
#rm -rf output.txt; touch output.txt
# 1. Initiate data preprocessing to generate the sanitized data file
data_preprocessing
# 2. Initiate causal discovery process
evaluate_discovery_process
