#!/bin/bash

progname="$(echo "$1" | xargs)"
progname_noext="${progname%.cpp}"

echo "Compiling program ..."
nvcc -c -I/usr/local/cuda/include "$progname"

if [[ "$?" -ne 0 ]]; then
    echo "Error in compilation!"
    exit 1
else
    g++ -o "$progname_noext" "$progname_noext.o" -L/usr/local/cuda/lib64 -lcusparse -lcudart
    if [[ "$?" -ne 0 ]]; then
        echo "Error in compilation!"
        exit 1
    else
        echo "Compilation Done."
        echo "Executing..."
        ./"$progname_noext"
    fi
fi
