#!/bin/bash

ITERS=(1 2 5 10 20 50 100 200 500 1000 2000)
SIZES=(1 2 4 8)

SAMPLES=500
SAMPLES_2=1000

DATE=`date +%Y-%m-%d_%H:%M:%S`
baseline_path="results_$DATE/baseline"
CNP_path="results_$DATE/CNP"
graph_path="results_$DATE/graph"
vk_path="results_$DATE/vulkan"
mkdir "results_$DATE"
mkdir "$baseline_path"
mkdir "$CNP_path"
mkdir "$graph_path"
mkdir "$vk_path"

for size in ${SIZES[*]}; do

    echo "Size $size"

    make clean
    make CDEFINES="-DDATA_SIZE=$size"

    cd vkcomp
    make clean
    make CDEFINES="-DDATA_SIZE=$size"
    cd ..

    for iter in ${ITERS[*]}; do

        filename=`printf "i%s_s%s" "$iter" "$size"`

        echo "sudo ./mainfile 0 $iter $SAMPLES"
        sudo ./mainfile 0 $iter $SAMPLES > partial.txt
        tail -n $SAMPLES_2 partial.txt >submit_part.txt
        head -n $SAMPLES submit_part.txt > "$baseline_path/sub_$filename.txt"
        tail -n $SAMPLES submit_part.txt > "$baseline_path/exe_$filename.txt"

        echo "sudo ./mainfile 1 $iter $SAMPLES"
        sudo ./mainfile 1 $iter $SAMPLES > partial.txt
        tail -n $SAMPLES_2 partial.txt >submit_part.txt
        head -n $SAMPLES submit_part.txt > "$CNP_path/sub_$filename.txt"
        tail -n $SAMPLES submit_part.txt > "$CNP_path/exe_$filename.txt"  

        echo "sudo ./mainfile 2 $iter $SAMPLES"
        sudo ./mainfile 2 $iter $SAMPLES > partial.txt
        tail -n $SAMPLES_2 partial.txt >submit_part.txt
        head -n $SAMPLES submit_part.txt > "$graph_path/sub_$filename.txt"
        tail -n $SAMPLES submit_part.txt > "$graph_path/exe_$filename.txt"

        rm submit_part.txt
        rm partial.txt

        cd vkcomp
        echo "sudo ./vkmain $iter $SAMPLES"
        sudo ./vkmain $iter $SAMPLES > partial.txt
        tail -n $SAMPLES_2 partial.txt >> submit_part.txt
        head -n $SAMPLES submit_part.txt > "../$vk_path/sub_$filename.txt"
        tail -n $SAMPLES submit_part.txt > "../$vk_path/exe_$filename.txt"
        rm submit_part.txt
        rm partial.txt
        cd ..

    done

done
