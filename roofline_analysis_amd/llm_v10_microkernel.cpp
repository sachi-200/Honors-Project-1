/**
 * @file tiled_matmul_avx.cpp
 * @brief A C++ program for parallel matrix multiplication with a full GEMM micro-kernel.
 *
 * This version represents the culmination of our optimization journey. It introduces
 * a second level of tiling (register blocking) by implementing a highly optimized
 * GEMM (General Matrix Multiply) micro-kernel. This kernel computes a small
 * block of the C matrix entirely within AVX registers, maximizing data reuse
 * in the L1 cache and registers, which is the key to approaching peak performance.
 *
 * The micro-kernel is 6 rows by 16 columns (2 AVX vectors), requiring 12 AVX
 * registers for the C accumulators, which fits within typical hardware limits.
 *
 * Optimization stack:
 * 1.  **Outer Tiling (Macro-Kernel):** For L2/L3 cache locality.
 * 2.  **Loop Reordering:** The `jj, kk, ii` order maximizes reuse of packed B tiles.
 * 3.  **Packing/Copying:** Ensures B's data is contiguous for streaming.
 * 4.  **Register Tiling (Micro-Kernel):** A 6x16 block of C is computed in
 * AVX registers to maximize reuse of A and B data in L1 cache/registers.
 * 5.  **Vectorization & Unrolling:** The micro-kernel is written with AVX
 * intrinsics to expose instruction-level parallelism.
 *
 * @author Gemini
 * @date 2025-10-16
 */

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include <algorithm>
#include <immintrin.h>

const int TILE_SIZE = 32;

// --- Define Micro-Kernel dimensions ---
// MR (rows) and NR (cols) define the size of the C-block held in registers.
// NR must be a multiple of AVX vector width (8 floats).
// MR * (NR/8) should be <= 14 to leave registers for A and B.
// A 6x16 block (6 * 2 = 12 registers) is a good choice.
#define MR 6
#define NR 16

/**
 * @brief The core GEMM micro-kernel.
 *
 * This function computes a small MRxNR block of C using AVX registers.
 * It assumes A is a panel from the original matrix (with row stride N) and
 * B is a panel from a packed, contiguous tile (with row stride TILE_SIZE).
 * C[i:i+MR, j:j+NR] += A[i:i+MR, k:k+KC] * B[k:k+KC, j:j+NR]
 */
void micro_kernel(const float* A, const float* B, float* C, int N, int KC) {
    __m256 c_vec[MR][NR/8]; // Accumulators for the C tile (6 rows, 2 vector-columns)

    // Initialize accumulator registers with current values from C
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR/8; ++j) {
            c_vec[i][j] = _mm256_load_ps(&C[i*N + j*8]);
        }
    }

    // Main loop over the k-dimension (reduction)
    for (int k = 0; k < KC; ++k) {
        // Load two vectors from the packed B tile. These will be reused for all 6 rows of A.
        // CORRECTION: The stride between rows in B_tile is TILE_SIZE, not NR.
        __m256 b_vec0 = _mm256_load_ps(&B[k*TILE_SIZE + 0]);
        __m256 b_vec1 = _mm256_load_ps(&B[k*TILE_SIZE + 8]);

        // For each row of A, broadcast the element and perform FMA on the C accumulators
        for(int i = 0; i < MR; ++i) {
            // CORRECTION: The stride between rows in the original A matrix is N, not KC.
            __m256 a_vec = _mm256_broadcast_ss(&A[i*N + k]);
            c_vec[i][0] = _mm256_fmadd_ps(a_vec, b_vec0, c_vec[i][0]);
            c_vec[i][1] = _mm256_fmadd_ps(a_vec, b_vec1, c_vec[i][1]);
        }
    }

    // Store the results from accumulators back to C
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR/8; ++j) {
            _mm256_store_ps(&C[i*N + j*8], c_vec[i][j]);
        }
    }
}

/**
 * @brief Performs tiled matrix multiplication C = A * B using a GEMM micro-kernel.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for shared(A, B, C, N)
    for (int jj = 0; jj < N; jj += TILE_SIZE) {
        for (int kk = 0; kk < N; kk += TILE_SIZE) {
             // --- Pack a TILE_SIZE x TILE_SIZE tile of B ---
             // This brings a chunk of B into a contiguous buffer, which is great for L1 cache.
            alignas(32) float B_tile[TILE_SIZE][TILE_SIZE];
            for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                    if (kk + k_local < N && jj + j_local < N) {
                       B_tile[k_local][j_local] = B[(kk + k_local) * N + (jj + j_local)];
                    }
                }
            }

            for (int ii = 0; ii < N; ii += TILE_SIZE) {
                // --- Inner loops iterate over the micro-tiles ---
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); i += MR) {
                    for (int j = jj; j < std::min(jj + TILE_SIZE, N); j += NR) {
                        // Check for boundary conditions to use the fast micro-kernel
                        if (i + MR <= N && j + NR <= N) {
                             micro_kernel(&A[i*N + kk], &B_tile[0][j-jj], &C[i*N+j], N, TILE_SIZE);
                        } else {
                            // Boundary case: Use scalar code for partial micro-tiles
                            for (int k_s = kk; k_s < std::min(kk + TILE_SIZE, N); ++k_s) {
                                for(int i_s = i; i_s < std::min(i + MR, N); ++i_s) {
                                    for(int j_s = j; j_s < std::min(j + NR, N); ++j_s) {
                                        C[i_s*N + j_s] += A[i_s*N + k_s] * B_tile[k_s-kk][j_s-jj];
                                    }
                                }
                            }
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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Matrix_Size_N>" << std::endl;
        return 1;
    }
    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: Matrix size N must be a positive integer." << std::endl;
        return 1;
    }

    std::cout << "🚀 Starting matmul for N = " << N << " with GEMM micro-kernel." << std::endl;

    size_t matrix_elements = (size_t)N * N;
    size_t matrix_size_bytes = matrix_elements * sizeof(float);
    float* A = (float*)aligned_alloc(32, matrix_size_bytes);
    float* B = (float*)aligned_alloc(32, matrix_size_bytes);
    float* C = (float*)aligned_alloc(32, matrix_size_bytes);

    std::srand(std::time(0));
    for (size_t i = 0; i < matrix_elements; ++i) {
        A[i] = (float)std::rand() / RAND_MAX;
        B[i] = (float)std::rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    std::cout << "Matrices allocated and initialized." << std::endl;
    std::cout << "Performing computation..." << std::endl;

    double start_time = omp_get_wtime();
    perform_tiled_multiplication(A, B, C, N);
    double end_time = omp_get_wtime();

    std::cout << "✅ Computation finished in " << end_time - start_time << " seconds." << std::endl;

    free(A);
    free(B);
    free(C);

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}