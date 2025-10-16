/**
 * @file tiled_matmul_ikj.cpp
 * @brief A C++ program for parallel tiled matrix multiplication with an i,k,j
 * loop order to maximize temporal locality.
 *
 * This version further optimizes the memory access pattern by reordering the
 * outer tile loops from i,j,k to i,k,j. This change significantly improves
 * temporal locality for the tiles of Matrix A.
 *
 * In an i,k,j ordering, a single tile of A (`A[i][k]`) is loaded into cache
 * and is reused for all computations across the `j` dimension. This minimizes
 * the number of times A's data needs to be fetched from main memory, reducing
 * memory traffic and improving performance, especially when the code is
 * memory-bound.
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
 * @brief Performs tiled matrix multiplication C = A * B using an i,k,j loop order.
 *
 * This function uses three key memory optimizations:
 * 1.  **i,k,j Loop Reordering:** The tile loops are ordered `ii, kk, jj`. This
 * causes a tile of A (`A[ii][kk]`) to be held in cache while the code
 * iterates through the entire `jj` dimension, maximizing its reuse. This is
 * the primary mechanism for improving temporal locality.
 * 2.  **Packing (Copy Optimization):** Tiles of matrix B are still copied into a
 * contiguous buffer to ensure efficient, stride-1 access.
 * 3.  **Software Prefetching:** Prefetch instructions are adjusted for the new
 * loop order to fetch the *next* tiles of A and B ahead of time.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        // The kk loop is now the second loop.
        for (int kk = 0; kk < N; kk += TILE_SIZE) {
            // --- Prefetch for the NEXT A-tile ---
            // We are about to reuse A[ii][kk] across the entire jj loop.
            // This is the ideal place to prefetch the tile for the *next* kk iteration,
            // A[ii][kk + TILE_SIZE], hiding its load latency behind the full jj-loop's execution.
            if (kk + TILE_SIZE < N) {
                __builtin_prefetch(&A[ii * N + (kk + TILE_SIZE)], 0, 3);
            }

            // The jj loop is now the innermost of the three tile loops.
            for (int jj = 0; jj < N; jj += TILE_SIZE) {
                // --- Prefetch for the NEXT B-tile ---
                // Inside the jj loop, we prefetch the B-tile for the next jj iteration.
                if (jj + TILE_SIZE < N) {
                    __builtin_prefetch(&B[kk * N + (jj + TILE_SIZE)], 0, 1);
                }

                // --- Packing (Copy Optimization) for the CURRENT B-tile ---
                float B_tile[TILE_SIZE][TILE_SIZE];
                for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                    for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                        if (kk + k_local < N && jj + j_local < N) {
                           B_tile[k_local][j_local] = B[(kk + k_local) * N + (jj + j_local)];
                        }
                    }
                }

                // --- Inner Computation Kernel ---
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        const float a_ik = A[i * N + k];
                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                            C[i * N + j] += a_ik * B_tile[k - kk][j - jj];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Main function to drive the matrix multiplication.
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

    std::cout << "🚀 Starting tiled matmul for N = " << N << " with i,k,j order." << std::endl;

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