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

Roofline analysis has provided the following metrics for this code:
Attained Performance: 18.07 GFLOP/s
Operational Intensity: 5.02 FLOPs/Byte
Diagnosis: The code is memory-bound, limited by latency.

The next optimization goal is to hide memory latency.
Please modify the tiled_matmul function to introduce software prefetching for matrices A and B.

Requirements:
Inside the innermost loop, prefetch data for a future iteration. A good strategy is to prefetch data a few iterations ahead (e.g., 8 to 16 iterations).
Add comments explaining the prefetching logic, including why a certain prefetch distance was chosen.

### v3: Improved Tiling + Prefetching

The previous attempt to add software prefetching was not successful. It increased the number of cache misses from 6.7 million to 12.1 million, which lowered the operational intensity and hurt performance. This was likely due to cache pollution.

Please try again, but be more strategic with the prefetching logic.

The prefetch instructions should target data that will be used in the near future, but not so far in the future that it gets evicted.

Critically, add comments to the code explaining the chosen prefetch distance and the specific memory access pattern it is targeting. This will help us understand the logic.

### v4: Tiling + Packing

The previous attempts to add prefetching were not successful. The new strategy is to address the poor cache locality caused by the column-wise memory access of matrix B.

Please modify the original tiled matrix multiplication code to implement a copy optimization (also known as packing) for the tiles of matrix B.

Requirements:

The overall tiling and OpenMP structure should remain the same.

Inside the jt loop (the loop for the j dimension of the tiles), but before the kt loop, do the following: a. Declare a small, local, contiguous array, e.g., float B_tile[TILE_SIZE][TILE_SIZE]; b. Write a nested loop to copy the relevant tile from the global matrix B into B_tile.

Modify the innermost computation to read from B_tile instead of the global matrix B.

Here is the best-performing code to use as the baseline:

```cpp
/**
 * @file tiled_matmul.cpp
 * @brief A complete C++ program for parallel tiled matrix multiplication.
 *
 * This program performs the multiplication of two dense N x N single-precision
 * floating-point matrices (C = A * B) using a tiled (blocked) algorithm.
 * The computation is parallelized with OpenMP by distributing tile computations
 * across multiple threads.
 *
 * The matrix size N is provided as a command-line argument.
 *
 * @author Gemini
 * @date 2025-10-16
 */

#include <iostream>
#include <vector>
#include <cstdlib> // For std::atoi, std::rand, std::srand
#include <ctime>   // For std::time
#include <omp.h>   // For OpenMP
#include <algorithm> // For std::min

// Define the tile size for the blocked algorithm.
// This size is crucial for performance and should be tuned based on the
// specific hardware's cache size. 32 is often a good starting point.
const int TILE_SIZE = 32;

/**
 * @brief Performs tiled matrix multiplication C = A * B.
 *
 * This function implements a cache-friendly, tiled matrix multiplication
 * algorithm. The matrices are divided into smaller blocks (tiles) of size
 * TILE_SIZE x TILE_SIZE. The outer loops iterate over these tiles, and the
 * inner loops perform the matrix multiplication for the elements within a tile.
 * This approach improves data locality and cache utilization.
 *
 * The outermost loop (iterating over rows of tiles) is parallelized using
 * OpenMP, allowing multiple threads to work on different horizontal strips
 * of the resulting matrix C simultaneously.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    // The #pragma directive parallelizes the outermost loop (the 'ii' loop).
    // Each thread will be assigned a different set of 'ii' values, effectively
    // giving each thread a horizontal strip of tiles of matrix C to compute.
    // The shared clause specifies that A, B, C, and N are shared among all threads.
    // Loop variables (ii, jj, kk, i, k, j) are private to each thread by default.
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // This is the mini-kernel that multiplies one tile of A by one tile of B
                // and adds the result to a tile of C.
                // Loop bounds use std::min to handle cases where N is not perfectly
                // divisible by TILE_SIZE.
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        // Pre-load A[i][k] into a register to reduce memory access
                        // within the innermost loop.
                        const float a_ik = A[i * N + k];
                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Main function to drive the matrix multiplication.
 *
 * It handles command-line argument parsing, memory allocation,
 * matrix initialization, calling the core multiplication function,
 * and finally, memory deallocation.
 */
int main(int argc, char* argv[]) {
    // 1. Argument Parsing
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Matrix_Size_N>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: Matrix size N must be a positive integer." << std::endl;
        return 1;
    }

    std::cout << "🚀 Starting tiled matrix multiplication for N = " << N
              << " with tile size " << TILE_SIZE << "." << std::endl;

    // 2. Dynamic Memory Allocation
    // Matrices are stored in a contiguous 1D array in row-major order.
    size_t matrix_size_bytes = (size_t)N * N * sizeof(float);
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // 3. Matrix Initialization
    std::srand(std::time(0)); // Seed for random number generation

    // Initialize A and B with random values between 0.0 and 1.0
    // Initialize C to all zeros.
    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)std::rand() / RAND_MAX;
        B[i] = (float)std::rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    std::cout << "Matrices allocated and initialized." << std::endl;
    std::cout << "Performing computation..." << std::endl;

    // 4. Core Matrix Multiplication
    double start_time = omp_get_wtime();
    perform_tiled_multiplication(A, B, C, N);
    double end_time = omp_get_wtime();

    std::cout << "✅ Computation finished in " << end_time - start_time << " seconds." << std::endl;

    // Optional: Print a few elements of the result matrix for verification.
    // std::cout << "Result C[0][0]: " << C[0] << std::endl;
    // std::cout << "Result C[N-1][N-1]: " << C[(N - 1) * N + (N - 1)] << std::endl;

    // 5. Memory Deallocation
    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}
```
