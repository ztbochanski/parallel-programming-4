#!/bin/bash

array_sizes=(1024 2048 4096 8192 16384 32768 65536 8388608)
output_file="results.csv"

echo "Array Size, Time (s)" > "$output_file"

for size in "${array_sizes[@]}"; do
    g++ -DARRAYSIZE=$size -o simd4 simd4.cpp -lm -fopenmp
    ./simd4
done
