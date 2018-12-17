count="0"
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
    #echo -n ,

    #./bin/blur-seq ./images/lena_std.tif
    echo -n ,
    echo
done
