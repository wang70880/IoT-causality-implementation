#!/bin/sh
declare -a data_array=("aruba")
#declare -a act_label_array=("Bed_to_Toilet" "Housekeeping" "Eating" "Wash_Dishes" Meal_Preparation "Leave_Home" "Enter_Home" "Work" "Sleeping" "Relax")
declare -a act_label_array=("Bed_to_Toilet")
declare -a rs_array=(0.0)
declare -a re_array=(0)
declare -a alpha_array=(0.01)
# 1: Data Preprocessing
preprocessData () {
    for val in ${act_label_array[@]}; do
        for rs in ${rs_array[@]}; do
            for re in ${re_array[@]}; do
                for alpha in ${alpha_array[@]}; do
                    echo "Now process the activity ${val} with rs=${rs}, re=${re}, alpha=${alpha}"
                    eval 'python preprocess.py ${val} ${rs} ${re}'
                done
            done
        done
    done
}

# 2: Call Discovery Algorithm to generate the skeleton
causalDiscovery() {
    # Initiate the discovery process.
    for val in ${act_label_array[@]}; do
        for rs in ${rs_array[@]}; do
            for re in ${re_array[@]}; do
                for alpha in ${alpha_array[@]}; do
                    eval 'Rscript pc.R ${val} ${alpha}'
                done
            done
        done
    done
}

# 3: Evaluate the causal discovery process
causalDiscoveryEvaluation () {
    for val in ${act_label_array[@]}; do
        echo "Now process the activity ${val}"
        eval 'python evaluation.py ${val}'
    done
}

discoveryProcess() {
    for dataset in ${data_array[@]}; do
        for val in ${act_label_array[@]}; do
            for rs in ${rs_array[@]}; do
                for re in ${re_array[@]}; do
                    for alpha in ${alpha_array[@]}; do
                        echo "Now process the activity ${val} with rs=${rs}, re=${re}, alpha=${alpha}"
#                        eval 'python preprocess.py ${val} ${rs} ${re}'
                        eval 'Rscript pc.R ${val} ${alpha}'
#                        eval 'python evaluation.py ${val}'
                        eval 'python user-inspection.py ${dataset} ${val}'
                    done
                done
            done
        done
    done
}

discoveryProcess
#preprocessData
#causalDiscovery
#causalDiscoveryEvaluation