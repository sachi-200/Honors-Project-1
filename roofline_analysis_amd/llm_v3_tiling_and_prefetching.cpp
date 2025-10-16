/**
 * @file tiled_matmul_prefetch_v2.cpp
 * @brief A complete C++ program for parallel tiled matrix multiplication
 * with strategic software prefetching.
 *
 * This program performs the multiplication of two dense N x N single-precision
 * floating-point matrices (C = A * B) using a tiled (blocked) algorithm.
 * The computation is parallelized with OpenMP and optimized with a corrected
 * software prefetching strategy to hide memory latency without causing
 * cache pollution.
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
 * @brief Performs tiled matrix multiplication C = A * B with strategic software prefetching.
 *
 * This version corrects the previous prefetching logic that caused cache pollution.
 * Prefetch instructions are now placed in the appropriate loops to match the
 * memory access patterns of matrices A and B, ensuring data arrives just-in-time
 * without evicting other useful data.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    // A conservative prefetch distance is chosen to avoid prefetching too far ahead,
    // which was the likely cause of cache pollution in the previous attempt. A distance
    // of 8-16 is often a good starting point for tuning, as it's far enough to hide
    // some latency but not so far that the data gets evicted before use.
    const int PREFETCH_DISTANCE = 8;

    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // This is the mini-kernel that multiplies one tile of A by one tile of B
                // and adds the result to a tile of C.
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {

                        // --- Strategic Prefetch for Matrix A ---
                        // The access pattern for A is A[i * N + k]. This value is constant
                        // throughout the inner 'j' loop. The prefetch should therefore target
                        // a future iteration of this 'k' loop. By placing it here, we
                        // fetch the data for A[i*N + k + PREFETCH_DISTANCE] just before
                        // we start the 'j' loop that will use A[i*N + k].
                        // Locality hint '3' (_MM_HINT_T0) is used because A[i][k] has high
                        // temporal locality—it will be reused for every iteration of the 'j' loop.
                        __builtin_prefetch(&A[i * N + k + PREFETCH_DISTANCE], 0, 3);

                        const float a_ik = A[i * N + k];

                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                            // --- Strategic Prefetch for Matrix B ---
                            // The access pattern for B is B[k * N + j]. This is a streaming,
                            // stride-1 access within this innermost loop. This is the ideal
                            // pattern for prefetching. We prefetch the element needed for
                            // a future 'j' iteration.
                            // Locality hint '0' (_MM_HINT_NTA) suggests non-temporal access,
                            // meaning the data is used once and won't be needed again soon.
                            // This prevents it from polluting higher-level caches.
                            __builtin_prefetch(&B[k * N + j + PREFETCH_DISTANCE], 0, 0);

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
              << " with tile size " << TILE_SIZE << " and strategic prefetching." << std::endl;

    // 2. Dynamic Memory Allocation
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