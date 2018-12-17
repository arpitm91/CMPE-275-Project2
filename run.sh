#!/bin/bash 

count="0"
COUNTER=0

while [  $COUNTER -lt 3 ]; do

    for file in ./images/*
    do
        FILESIZE=$(stat -c%s "$file")
        echo -n $count, $file, $FILESIZE,
        count=$((count+1))

        ./bin/contrast-seq  $file
        echo -n ,

        ./bin/contrast-omp  $file
        echo -n ,

        ./bin/contrast-cuda  $file
        echo -n ,

        ./bin/sharpen-seq  $file
        echo -n ,

        ./bin/sharpen-omp  $file
        echo -n ,

        ./bin/sharpen-cuda  $file
        echo -n ,

        ./bin/blur-seq  $file
        echo -n ,

        ./bin/blur-omp $file
        echo -n ,

        ./bin/blur-cuda $file
        echo -n ,

        echo
    done
    COUNTER=$((COUNTER+1))
done 

