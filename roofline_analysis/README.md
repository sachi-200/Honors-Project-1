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