/**
 * @file tiled_matmul_prefetch.cpp
 * @brief A complete C++ program for parallel tiled matrix multiplication
 * with software prefetching optimization.
 *
 * This program performs the multiplication of two dense N x N single-precision
 * floating-point matrices (C = A * B) using a tiled (blocked) algorithm.
 * The computation is parallelized with OpenMP and optimized with software
 * prefetching to hide memory latency.
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
const int TILE_SIZE = 32;

/**
 * @brief Performs tiled matrix multiplication C = A * B with software prefetching.
 *
 * This function implements a cache-friendly, tiled matrix multiplication
 * algorithm enhanced with software prefetching to hide memory latency.
 * Based on the roofline analysis, the operation is memory-bound. Prefetching
 * instructs the CPU to start loading data from memory into the cache before
 * it is actually needed, reducing processor stalls.
 *
 * The outermost loop (iterating over rows of tiles) is parallelized using
 * OpenMP.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    // Define the prefetch distance. This value determines how far ahead
    // we request data. A distance of 16 is chosen as a practical starting
    // point. It needs to be large enough to hide the memory access latency,
    // which can be hundreds of clock cycles. Tuning this value is often
    // necessary for optimal performance on a specific architecture.
    const int PREFETCH_DISTANCE = 16;

    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // This is the mini-kernel that multiplies one tile of A by one tile of B
                // and adds the result to a tile of C.
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        const float a_ik = A[i * N + k];
                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                            // --- Software Prefetching Logic ---
                            // Prefetching is done inside the innermost loop to issue
                            // requests continuously as we process data.

                            // 1. Prefetch from Matrix B:
                            // The access pattern for B is B[k * N + j]. As 'j'
                            // increments, we are streaming through memory contiguously.
                            // We ask for the data PREFETCH_DISTANCE iterations ahead.
                            // The intrinsic __builtin_prefetch(address, rw, locality) is used:
                            //  - rw=0: Prefetch for a read operation.
                            //  - locality=0: Hint for low temporal locality (_MM_HINT_NTA).
                            //    This is for streaming data that will be used once,
                            //    preventing it from polluting more valuable cache levels.
                            __builtin_prefetch(&B[k * N + j + PREFETCH_DISTANCE], 0, 0);

                            // 2. Prefetch from Matrix A:
                            // The access for A is A[i * N + k]. This address is
                            // invariant inside this 'j' loop. We are prefetching data
                            // for a future iteration of the outer 'k' loop.
                            // NOTE: This issues redundant prefetch requests for the same
                            // address within this loop. A more optimal approach would be
                            // to place this prefetch inside the 'k' loop.
                            //  - locality=3: Hint for high temporal locality (_MM_HINT_T0).
                            //    This brings data into all cache levels, including L1,
                            //    as it will be reused across the 'j' loop.
                            __builtin_prefetch(&A[i * N + k + PREFETCH_DISTANCE], 0, 3);

                            // Note: Prefetching past array boundaries is technically
                            // undefined behavior, but on most modern architectures,
                            // these instructions are treated as no-ops and do not
                            // cause a program fault. This avoids adding a conditional
                            // branch inside this performance-critical loop.

                            // --- Core Computation ---
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
              << " with tile size " << TILE_SIZE << " and software prefetching." << std::endl;

    // 2. Dynamic Memory Allocation
    size_t matrix_size_bytes = (size_t)N * N * sizeof(float);
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // 3. Matrix Initialization
    std::srand(std::time(0));
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

    // 5. Memory Deallocation
    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}