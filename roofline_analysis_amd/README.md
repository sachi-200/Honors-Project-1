# Roofline Analysis

## Peak Memory Bandwidth

Calculated using the STREAM Benchmark (<https://www.cs.virginia.edu/stream/ref.html#start>). Run as follows

```plaintext
gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=100000000 stream.c -o stream_benchmark

./stream_benchmark
```

Convert triad results to GB/s.

## Peak Compute

Calculated using likwid-bench (<https://github.com/RRZE-HPC/likwid>), command is as follows

```plaintext
likwid-bench -t peakflops_sp_avx_fma -W N:100MB
```

The MFLOPs/s value is converted to GFLOP/s for the purpose of this project.

## Ryzen 7 6800HS peaks

PEAK_MEM_BW = 28.9 (GB/s measured using Stream benchmark)
PEAK_COMPUTE = 328.4  (GFLOP/s, float32 single precision, vectorized workload using likwid-bench)

## Prompt History

### v1: Tiling

Please generate a complete C++ program that performs dense matrix multiplication (C=A×B) for two N x N single-precision floating-point matrices.

The program must meet the following requirements:

Algorithm: Implement a tiled (or blocked) matrix multiplication algorithm.
The outer loops should iterate over the tiles, and the inner loops should perform the multiplication for the elements within a tile.

Parallelization:
The computation must be parallelized using OpenMP.
Apply the OpenMP pragma to the outermost loop (i or row loop) to distribute the tile computations across multiple threads.

Main Function:
The program must include a main function that accepts the matrix size N as a single command-line argument.
It should dynamically allocate memory for matrices A, B, and C.
Initialize matrices A and B with random values and matrix C to all zeros.
Call the core matrix multiplication function to perform the computation.
Free the allocated memory before exiting.

Ensure the final output is a single, complete, and runnable C++ file.

### v2: Tiling + Prefetching
