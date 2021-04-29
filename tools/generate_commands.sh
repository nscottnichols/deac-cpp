#! /bin/bash
deac_commands_file="deac_commands";
save_directory="./deacresults_M512_P16_0.005"
mkdir -p ${save_directory}
for i in {0..9999}
do
    printf -v i_fmt "%06d" $i
    cmd="{ time ./deac.e -P 16 -M 512 --save_directory ${save_directory} --omega_max 64.0 --normalize --temperature 1.2 --number_of_generations 1000000 --stop_minimum_fitness 0.005 --seed ${i} ./isf_data.dat ; } > ${save_directory}/deac_time_${i_fmt}.dat 2>&1"
    echo "$cmd" >> ${deac_commands_file} ;
done
