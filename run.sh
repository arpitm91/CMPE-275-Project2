count="0"
for file in ./images/*
do
    FILESIZE=$(stat -f%z "$file")
    echo -n $count, $file, $FILESIZE,
    count=$((count+1))

    ./bin/sharpen-omp  $file
    echo -n ,

    ./bin/sharpen-seq  $file
    echo -n ,

    ./bin/sharpen-cuda  $file
    echo -n ,


    echo ,
done