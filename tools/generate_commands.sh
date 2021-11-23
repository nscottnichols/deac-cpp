#!/bin/bash
number_of_reconstructions=1000; #set number of reconstructions
if [ $# -lt 3 ]; then
    echo "generate_commands.sh deac_executable isf_file deac_results_directory <deac_commands_file>";
    exit 1
fi

DEAC_EXE=$1; #path to DEAC executable
ISF_FILE=$2; #path to input binary containing imaginary time, intermediate scattering function, and error
DEAC_RESULTS_DIR=$3; #path to directory to store DEAC results
if [ $# -lt 4 ]; then
    DEAC_COMMAND_FILE="deac_commands";
else
    DEAC_COMMAND_FILE=$4;#path to directory to DEAC commands file
fi

rm -f ${DEAC_COMMAND_FILE}
for seed in $(seq 1 ${number_of_reconstructions})
do
    cmd="${DEAC_EXE} \
    -N 1600000 \
    -T 1.35 \
    -P 8 \
    -M 4096 \
    --normalize \
    --omega_max 512.0 \
    --stop_minimum_fitness 1.0 \
    --save_directory ${DEAC_RESULTS_DIR} \
    --seed ${seed} \
    ${ISF_FILE}";
    echo "${cmd}" >> ${DEAC_COMMAND_FILE};
done
