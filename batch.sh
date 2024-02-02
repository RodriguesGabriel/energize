#!/bin/bash

EXP="./experiments"
CONF="./config"

function change_gpu_mode() {
    MODE=$1
    sudo nvidia-smi -i 0 -c $MODE >> /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error occurred on nvidia-smi"
        exit 1
    fi
}

function handle_signal() {
    echo "Signal detected. Changing GPU mode to default before exiting."
    change_gpu_mode 0
    exit 0
}

function run() {
    CONFIG=$1
    GRAMMAR=$2
    DATASET=$3
    RUN=$4

    EXP_PATH=$(jq -r '.checkpoints_path' $CONFIG)
    echo "Executing run ${RUN}"
    mkdir -p $EXP_PATH/run_$RUN
    python3 -u \
        -m energize.main \
        -d $DATASET \
        -c $CONFIG \
        -g $GRAMMAR \
        --run $RUN \
        --gpu-enabled \
        >> "$EXP_PATH/run_$RUN/energize.log" \
        2>&1
}

if [ $# -eq 0 ]; then
    echo "./batch.sh <CONFIG> <GRAMMAR> <DATASET> <NUM RUNS>"
else
    RUNS=$(($4 - 1))
    CONDA_PATH=$(whereis conda | awk '{print $2}')
    source $CONDA_PATH/bin/activate energize >> /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "Error occurred on conda activate"
        exit 1
    fi

    trap handle_signal SIGINT SIGTERM SIGQUIT SIGABRT SIGKILL SIGHUP SIGSTOP SIGTSTP SIGCONT SIGUSR1 SIGUSR2
    change_gpu_mode 3
    for i in $(seq 0 $RUNS); do
        run $1 $2 $3 $i
    done
    change_gpu_mode 0
fi
