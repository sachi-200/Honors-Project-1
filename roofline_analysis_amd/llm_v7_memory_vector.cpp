/**
 * @file tiled_matmul_avx.cpp
 * @brief A C++ program for parallel matrix multiplication optimized with AVX intrinsics.
 *
 * This version introduces explicit vectorization in the innermost compute kernel
 * using AVX (Advanced Vector Extensions) intrinsics. After optimizing for memory
 * locality with tiling, packing, and loop reordering, the final bottleneck becomes
 * the scalar computation itself.
 *
 * By using AVX, we can perform 8 single-precision floating-point operations
 * (via a fused multiply-add) simultaneously. This dramatically increases the
 * computational density and pushes the performance closer to the hardware's
 * compute ceiling, making much better use of the data already in the cache.
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
#include <immintrin.h> // Required for AVX intrinsics

// Define the tile size for the blocked algorithm.
const int TILE_SIZE = 32;

/**
 * @brief Performs tiled matrix multiplication C = A * B using AVX vectorization.
 *
 * This function uses a fully optimized micro-kernel:
 * 1.  **i,k,j Loop Reordering:** Maximizes temporal locality for A's tiles.
 * 2.  **Packing (Copy Optimization):** Ensures stride-1 access for B's tiles.
 * 3.  **Software Prefetching:** Hides latency for loading the next tiles.
 * 4.  **AVX Vectorization:** The innermost loop (`j`) is replaced with AVX intrinsics
 * to perform 8 floating-point operations per instruction.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int kk = 0; kk < N; kk += TILE_SIZE) {
            if (kk + TILE_SIZE < N) {
                __builtin_prefetch(&A[ii * N + (kk + TILE_SIZE)], 0, 3);
            }
            for (int jj = 0; jj < N; jj += TILE_SIZE) {
                if (jj + TILE_SIZE < N) {
                    __builtin_prefetch(&B[kk * N + (jj + TILE_SIZE)], 0, 1);
                }

                // B_tile is aligned to 32 bytes for efficient AVX loads.
                alignas(32) float B_tile[TILE_SIZE][TILE_SIZE];
                for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                    for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                        if (kk + k_local < N && jj + j_local < N) {
                           B_tile[k_local][j_local] = B[(kk + k_local) * N + (jj + j_local)];
                        }
                    }
                }

                // --- Vectorized Inner Computation Kernel ---
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        // Load a single float from A and broadcast it to all 8 lanes of a 256-bit AVX register.
                        __m256 a_vec = _mm256_broadcast_ss(&A[i * N + k]);

                        // The j-loop is now unrolled by 8, the width of an AVX register (8 floats).
                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); j += 8) {
                            // Ensure we don't read past the boundary of C or B_tile.
                            if (j + 7 < N) {
                                // Load 8 floats from the C matrix (aligned load).
                                __m256 c_vec = _mm256_load_ps(&C[i * N + j]);
                                // Load 8 floats from our packed and aligned B_tile.
                                __m256 b_vec = _mm256_load_ps(&B_tile[k - kk][j - jj]);

                                // Perform the Fused Multiply-Add (FMA) operation: c_vec = (a_vec * b_vec) + c_vec
                                // This is the core computation, performing 8 multiplies and 8 adds.
                                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                                // Store the 8 resulting floats back into the C matrix (aligned store).
                                _mm256_store_ps(&C[i * N + j], c_vec);
                            } else {
                                // If we can't process a full vector of 8, fall back to scalar code
                                // for the remaining elements to handle edge cases.
                                for(int j_scalar = j; j_scalar < std::min(jj + TILE_SIZE, N); ++j_scalar) {
                                    C[i * N + j_scalar] += A[i * N + k] * B_tile[k - kk][j_scalar - jj];
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

    std::cout << "🚀 Starting tiled matmul for N = " << N << " with AVX vectorization." << std::endl;

    // 2. Dynamic Memory Allocation
    // All matrices must be aligned to 32-bytes for efficient AVX instructions.
    size_t matrix_elements = (size_t)N * N;
    size_t matrix_size_bytes = matrix_elements * sizeof(float);
    float* A = (float*)aligned_alloc(32, matrix_size_bytes);
    float* B = (float*)aligned_alloc(32, matrix_size_bytes);
    float* C = (float*)aligned_alloc(32, matrix_size_bytes);


    // 3. Matrix Initialization
    std::srand(std::time(0));
    for (size_t i = 0; i < matrix_elements; ++i) {
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
    free(A); // Use free for memory allocated with aligned_alloc
    free(B);
    free(C);

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}