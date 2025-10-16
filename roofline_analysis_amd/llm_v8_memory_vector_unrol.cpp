/**
 * @file tiled_matmul_avx.cpp
 * @brief A C++ program for parallel matrix multiplication optimized with AVX and loop unrolling.
 *
 * This final version improves compute-bound performance by unrolling the innermost
 * vectorized loop. This reduces loop overhead and exposes more instruction-level
 * parallelism to the CPU, allowing its out-of-order execution engine to better
 * utilize the FMA (Fused Multiply-Add) units.
 *
 * By processing four __m256 vectors per loop iteration, we can hide the latency
 * of the FMA instructions and push performance closer to the processor's
 * theoretical compute peak.
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
 * @brief Performs tiled matrix multiplication C = A * B using an unrolled AVX kernel.
 *
 * This function uses a fully optimized micro-kernel:
 * 1.  **i,k,j Loop Reordering:** Maximizes temporal locality for A's tiles.
 * 2.  **Packing (Copy Optimization):** Ensures stride-1 access for B's tiles.
 * 3.  **Software Prefetching:** Hides latency for loading the next tiles.
 * 4.  **AVX Vectorization with Loop Unrolling:** The innermost loop (`j`) is unrolled
 * by a factor of 4, processing 32 floats (4 vectors) per iteration.
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

                alignas(32) float B_tile[TILE_SIZE][TILE_SIZE];
                for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                    for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                        if (kk + k_local < N && jj + j_local < N) {
                           B_tile[k_local][j_local] = B[(kk + k_local) * N + (jj + j_local)];
                        }
                    }
                }

                // --- Vectorized and Unrolled Inner Computation Kernel ---
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        __m256 a_vec = _mm256_broadcast_ss(&A[i * N + k]);

                        int j = jj;
                        const int bound = std::min(jj + TILE_SIZE, N);

                        // Unroll by 4: Process 32-float (4-vector) chunks.
                        for (; j + 31 < bound; j += 32) {
                            __m256 c_vec0 = _mm256_load_ps(&C[i * N + j + 0]);
                            __m256 c_vec1 = _mm256_load_ps(&C[i * N + j + 8]);
                            __m256 c_vec2 = _mm256_load_ps(&C[i * N + j + 16]);
                            __m256 c_vec3 = _mm256_load_ps(&C[i * N + j + 24]);

                            __m256 b_vec0 = _mm256_load_ps(&B_tile[k - kk][j - jj + 0]);
                            __m256 b_vec1 = _mm256_load_ps(&B_tile[k - kk][j - jj + 8]);
                            __m256 b_vec2 = _mm256_load_ps(&B_tile[k - kk][j - jj + 16]);
                            __m256 b_vec3 = _mm256_load_ps(&B_tile[k - kk][j - jj + 24]);

                            c_vec0 = _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);
                            c_vec1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
                            c_vec2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2);
                            c_vec3 = _mm256_fmadd_ps(a_vec, b_vec3, c_vec3);

                            _mm256_store_ps(&C[i * N + j + 0], c_vec0);
                            _mm256_store_ps(&C[i * N + j + 8], c_vec1);
                            _mm256_store_ps(&C[i * N + j + 16], c_vec2);
                            _mm256_store_ps(&C[i * N + j + 24], c_vec3);
                        }

                        // Cleanup loop for remaining 8-float chunks.
                        for (; j + 7 < bound; j += 8) {
                             __m256 c_vec = _mm256_load_ps(&C[i * N + j]);
                             __m256 b_vec = _mm256_load_ps(&B_tile[k - kk][j - jj]);
                             c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                             _mm256_store_ps(&C[i * N + j], c_vec);
                        }

                        // Scalar cleanup for the last <8 elements.
                        for (; j < bound; ++j) {
                            C[i * N + j] += A[i * N + k] * B_tile[k - kk][j - jj];
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

    std::cout << "🚀 Starting tiled matmul for N = " << N << " with AVX and loop unrolling." << std::endl;

    // 2. Dynamic Memory Allocation
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
    free(A);
    free(B);
    free(C);

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}