/**
 * @file tiled_matmul_packing_prefetch.cpp
 * @brief A C++ program for parallel tiled matrix multiplication combining
 * packing and software prefetching.
 *
 * This program builds upon the successful "copy optimization" (packing) strategy.
 * After packing a tile of matrix B to ensure ideal cache access in the compute
 * kernel, this version adds strategic software prefetching.
 *
 * The prefetch instructions are designed to hide the memory latency of loading
 * the tiles for the *next* major iteration. While the CPU computes using the
 * current tiles of A and B, we issue requests for the tiles that will be
 * needed in the subsequent k-dimension step, ensuring they are already in or
 * on their way to the cache when needed.
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
 * @brief Performs tiled matrix multiplication C = A * B using packing and prefetching.
 *
 * This function combines two powerful memory optimization techniques:
 * 1.  **Packing (Copy Optimization):** A tile of matrix B is copied into a
 * contiguous local buffer (`B_tile`) to ensure stride-1 access in the
 * inner compute loop.
 * 2.  **Software Prefetching:** While the current tile computation is running,
 * `__builtin_prefetch` instructions are issued to begin fetching the data
 * for the tiles needed in the *next* iteration of the `kk` loop. This
 * hides the latency of reading from main memory.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // --- Software Prefetching Logic ---
                // We prefetch data for the NEXT iteration of the kk-loop (kk + TILE_SIZE).
                // This gives the memory controller time to fetch the data while we are
                // busy with the computation for the CURRENT kk-tile.
                // A boundary check is essential to prevent reading past the matrix bounds.
                if (kk + TILE_SIZE < N) {
                    // 1. Prefetch the next tile of A:
                    // The next tile of A starts at A[ii][kk + TILE_SIZE]. We give a hint
                    // for high temporal locality (3), as this tile will be reused
                    // across the entire inner 'j' loop in the next kk-iteration.
                    __builtin_prefetch(&A[ii * N + (kk + TILE_SIZE)], 0, 3);

                    // 2. Prefetch the next tile of B:
                    // This is the source data from the global B matrix that will be
                    // *packed* in the next kk-iteration. We give a hint for medium
                    // temporal locality (1), as it will be read once for packing.
                    __builtin_prefetch(&B[(kk + TILE_SIZE) * N + jj], 0, 1);
                }

                // --- Copy Optimization (Packing) ---
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

    std::cout << "🚀 Starting tiled matmul for N = " << N << " with packing and prefetching." << std::endl;

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
