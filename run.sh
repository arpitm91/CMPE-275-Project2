for file in ./images/*
do
    utime="$( TIMEFORMAT='%lU';time ( ./bin/sharpen-omp  $file ) 2>&1 1>/dev/null )"
    echo sharpen-omp, $file, $utime

    utime="$( TIMEFORMAT='%lU';time ( ./bin/sharpen-seq  $file ) 2>&1 1>/dev/null )"
    echo sharpen-seq, $file, $utime
done