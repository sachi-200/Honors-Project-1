/*
 * Optimization for AMD EPYC 9365 (Zen 4/5)
 * Hardware Specs: 144 logical threads, 72 physical cores.
 * ISA: AVX-512 with FMA support.
 *
 * Optimization Strategy:
 * 1. SIMD Vectorization: Utilizing 512-bit ZMM registers via AVX-512 intrinsics.
 * 2. Register Blocking: A 16x16 micro-kernel uses 16 ZMM registers as accumulators
 *    for one block of C, maximizing instruction-level parallelism and register reuse.
 * 3. Cache-Aware Tiling: 3-level tiling (BM, BN, BK) to keep data within L2/L3 caches.
 * 4. Parallelism: OpenMP used to distribute work across 144 logical threads. 
 *    The `collapse(2)` clause ensures a large enough iteration space for high core counts.
 * 5. NUMA-Awareness: Parallel "First Touch" initialization in main() ensures that 
 *    memory pages are allocated on the NUMA node local to the thread that processes them.
 * 6. Edge Handling: Scalar cleanup paths handle matrices with dimensions not divisible by tile/vector sizes.
 *
 * Compile Command:
 * g++ -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
 */

#include <immintrin.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <random>
#include <cassert>
#include <fstream>
#include <string>
#include <iomanip>
#include <omp.h>
#include <algorithm>

// Tiling Parameters
// BM, BN, BK: Tile sizes for M, N, and K dimensions.
// These are chosen to fit within the Zen 4/5 L2 (1MB per core) and L3 caches.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 256;

// Micro-kernel sizes (Fixed by AVX-512 register width and count)
// MR: number of rows processed in one micro-kernel step.
// NR: number of columns processed in one micro-kernel step (1 ZMM = 16 floats).
constexpr int MR = 16;
constexpr int NR = 16;

/**
 * gemm_scalar: Reference C++ implementation of GEMM.
 * Computes C = A * B + C using row-major storage.
 */
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float a_val = A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += a_val * B[k * ldb + j];
            }
        }
    }
}

/**
 * gemm_avx512: Highly optimized AVX-512 GEMM implementation.
 * Performs C = A * B + C.
 */
#if defined(__AVX512F__)
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    // Outer loops parallelized with OpenMP. 
    // collapse(2) provides (M/BM) * (N/BN) tasks to saturate 144 threads.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m += BM) {
        for (int n = 0; n < N; n += BN) {
            
            int m_limit = std::min(m + BM, M);
            int n_limit = std::min(n + BN, N);

            // Iterate through the C-tile using MRxNR micro-panels
            for (int im = m; im < m_limit; im += MR) {
                for (int jn = n; jn < n_limit; jn += NR) {

                    // Optimized path: Full 16x16 micro-kernel
                    if (im + 16 <= m_limit && jn + 16 <= n_limit) {
                        
                        // Register blocking: 16 ZMM registers act as accumulators
                        __m512 c[16];
                        c[0]  = _mm512_loadu_ps(&C[(im + 0)  * ldc + jn]);
                        c[1]  = _mm512_loadu_ps(&C[(im + 1)  * ldc + jn]);
                        c[2]  = _mm512_loadu_ps(&C[(im + 2)  * ldc + jn]);
                        c[3]  = _mm512_loadu_ps(&C[(im + 3)  * ldc + jn]);
                        c[4]  = _mm512_loadu_ps(&C[(im + 4)  * ldc + jn]);
                        c[5]  = _mm512_loadu_ps(&C[(im + 5)  * ldc + jn]);
                        c[6]  = _mm512_loadu_ps(&C[(im + 6)  * ldc + jn]);
                        c[7]  = _mm512_loadu_ps(&C[(im + 7)  * ldc + jn]);
                        c[8]  = _mm512_loadu_ps(&C[(im + 8)  * ldc + jn]);
                        c[9]  = _mm512_loadu_ps(&C[(im + 9)  * ldc + jn]);
                        c[10] = _mm512_loadu_ps(&C[(im + 10) * ldc + jn]);
                        c[11] = _mm512_loadu_ps(&C[(im + 11) * ldc + jn]);
                        c[12] = _mm512_loadu_ps(&C[(im + 12) * ldc + jn]);
                        c[13] = _mm512_loadu_ps(&C[(im + 13) * ldc + jn]);
                        c[14] = _mm512_loadu_ps(&C[(im + 14) * ldc + jn]);
                        c[15] = _mm512_loadu_ps(&C[(im + 15) * ldc + jn]);

                        // K-tiling: process chunks of K to improve L1/L2 data reuse
                        for (int k_outer = 0; k_outer < K; k_outer += BK) {
                            int k_limit = std::min(k_outer + BK, K);

                            for (int k = k_outer; k < k_limit; ++k) {
                                // Load row strip from B (NR=16 columns)
                                __m512 b_vec = _mm512_loadu_ps(&B[k * ldb + jn]);

                                // Perform FMA operations. Broad-cast elements from A rows.
                                // Zen 4/5 can execute 2 FMA per cycle.
                                c[0]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 0)  * lda + k]), b_vec, c[0]);
                                c[1]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 1)  * lda + k]), b_vec, c[1]);
                                c[2]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 2)  * lda + k]), b_vec, c[2]);
                                c[3]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 3)  * lda + k]), b_vec, c[3]);
                                c[4]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 4)  * lda + k]), b_vec, c[4]);
                                c[5]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 5)  * lda + k]), b_vec, c[5]);
                                c[6]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 6)  * lda + k]), b_vec, c[6]);
                                c[7]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 7)  * lda + k]), b_vec, c[7]);
                                c[8]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 8)  * lda + k]), b_vec, c[8]);
                                c[9]  = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 9)  * lda + k]), b_vec, c[9]);
                                c[10] = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 10) * lda + k]), b_vec, c[10]);
                                c[11] = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 11) * lda + k]), b_vec, c[11]);
                                c[12] = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 12) * lda + k]), b_vec, c[12]);
                                c[13] = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 13) * lda + k]), b_vec, c[13]);
                                c[14] = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 14) * lda + k]), b_vec, c[14]);
                                c[15] = _mm512_fmadd_ps(_mm512_set1_ps(A[(im + 15) * lda + k]), b_vec, c[15]);
                            }
                        }

                        // Store result micro-panel back to C
                        _mm512_storeu_ps(&C[(im + 0)  * ldc + jn], c[0]);
                        _mm512_storeu_ps(&C[(im + 1)  * ldc + jn], c[1]);
                        _mm512_storeu_ps(&C[(im + 2)  * ldc + jn], c[2]);
                        _mm512_storeu_ps(&C[(im + 3)  * ldc + jn], c[3]);
                        _mm512_storeu_ps(&C[(im + 4)  * ldc + jn], c[4]);
                        _mm512_storeu_ps(&C[(im + 5)  * ldc + jn], c[5]);
                        _mm512_storeu_ps(&C[(im + 6)  * ldc + jn], c[6]);
                        _mm512_storeu_ps(&C[(im + 7)  * ldc + jn], c[7]);
                        _mm512_storeu_ps(&C[(im + 8)  * ldc + jn], c[8]);
                        _mm512_storeu_ps(&C[(im + 9)  * ldc + jn], c[9]);
                        _mm512_storeu_ps(&C[(im + 10) * ldc + jn], c[10]);
                        _mm512_storeu_ps(&C[(im + 11) * ldc + jn], c[11]);
                        _mm512_storeu_ps(&C[(im + 12) * ldc + jn], c[12]);
                        _mm512_storeu_ps(&C[(im + 13) * ldc + jn], c[13]);
                        _mm512_storeu_ps(&C[(im + 14) * ldc + jn], c[14]);
                        _mm512_storeu_ps(&C[(im + 15) * ldc + jn], c[15]);

                    } else {
                        // Cleanup path: handles edge blocks that don't fit the 16x16 kernel.
                        // Loop order ii-kk-jj ensures contiguous B access (jj).
                        for (int ii = im; ii < std::min(im + MR, m_limit); ++ii) {
                            for (int kk = 0; kk < K; ++kk) {
                                float a_val = A[ii * lda + kk];
                                for (int jj = jn; jj < std::min(jn + NR, n_limit); ++jj) {
                                    C[ii * ldc + jj] += a_val * B[kk * ldb + jj];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
#else
void gemm_avx512(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc) {
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif

/**
 * write_matrix_to_file: Saves matrix contents to a text file for correctness verification.
 */
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream f(filename);
    if (!f.is_open()) return;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            f << std::fixed << std::setprecision(6) << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        f << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    bool dump = (argc > 4 && std::string(argv[4]) == "--dump-matrices");

    // Dimensions for row-major layout
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Aligned memory allocation (64-byte for ZMM loads/stores)
    float* A = (float*)_mm_malloc(M * lda * sizeof(float), 64);
    float* B = (float*)_mm_malloc(K * ldb * sizeof(float), 64);
    float* C = (float*)_mm_malloc(M * ldc * sizeof(float), 64);

    // NUMA-aware First-Touch initialization.
    // Parallelizing initialization pins data pages to the NUMA node local to the thread.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i * lda + j] = static_cast<float>((i + j) % 100) * 0.001f;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * ldb + j] = static_cast<float>((i - j + 100) % 100) * 0.001f;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = 0.0f;
        }
    }

    if (dump) {
        // Test Mode: Verify correctness against scalar reference
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        float* C_ref = (float*)_mm_malloc(M * ldc * sizeof(float), 64);
        std::memset(C_ref, 0, M * ldc * sizeof(float));

        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);

        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

        bool passed = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::abs(C[i] - C_ref[i]) > 1e-2) {
                passed = false;
                break;
            }
        }
        std::cout << "Internal check: " << (passed ? "PASSED" : "FAILED") << std::endl;
        _mm_free(C_ref);
    } else {
        // Perf Mode: Measure GFLOPS using the optimized kernel
        auto start = std::chrono::high_resolution_clock::now();
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);
        std::cout << "Time: " << std::fixed << std::setprecision(6) << diff.count() 
                  << " s, GFLOPS: " << gflops << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}