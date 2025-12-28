/*
 * Optimization for AMD EPYC 9365 (Zen 4) - GEMM AVX-512
 * 
 * Target Architecture: Zen 4 / Zen 5 (AVX-512 support)
 * Logical Threads: 144 (72 Physical Cores)
 * L3 Cache: 384 MiB total
 * 
 * Optimization Strategy:
 * 1. Register Blocking: 6x64 micro-kernel.
 *    - Uses 24 ZMM registers (6 rows x 4 registers/row) to hold the C sub-block.
 *    - 4 ZMM registers for loading B-rows (micro-panels).
 *    - 1 ZMM register for broadcasting elements of A.
 *    - Total 29/32 ZMM registers used, maximizing ILP without register spilling.
 * 2. C-Matrix Reuse: Accumulators stay in registers for the entire K-dimension
 *    to minimize memory traffic for matrix C.
 * 3. Cache-aware Tiling:
 *    - BM=192, BN=384 to balance work distribution across 144 threads and maximize L2/L3 locality.
 *    - BK=512 manages the K-dimension traversal to keep A and B sub-tiles in cache.
 * 4. AVX-512 Masking:
 *    - N-dimension tails are handled natively using _mm512_maskz_loadu_ps and 
 *      _mm512_mask_storeu_ps, ensuring high performance even for non-multiples of 16.
 * 5. NUMA-Awareness:
 *    - Parallel matrix initialization in main() ensures "First Touch" allocation,
 *      pinning memory pages to the NUMA node near the executing threads.
 * 6. Software Pipelining:
 *    - The micro-kernel's row updates are manually unrolled, exposing independent FMA 
 *      operations to the CPU's out-of-order execution unit.
 * 
 * Compile Command (GCC):
 * g++ -O3 -march=znver4 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
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

// Autotuning Parameters
static constexpr int BM = 192;  // M-tile size
static constexpr int BN = 384;  // N-tile size
static constexpr int BK = 512;  // K-tile size
static constexpr int MR = 6;    // Micro-kernel rows (A)
static constexpr int NR = 64;   // Micro-kernel columns (B, 4 x 16-float ZMM)

/**
 * @brief Reference scalar implementation of GEMM.
 * Signatures match requirements exactly.
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
 * @brief Optimized AVX-512 GEMM implementation.
 */
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__)
    // Distribute tiles of M and N across threads.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m += BM) {
        for (int n = 0; n < N; n += BN) {
            int m_limit = std::min(m + BM, M);
            int n_limit = std::min(n + BN, N);

            for (int i = m; i < m_limit; i += MR) {
                int cur_mr = std::min(MR, m_limit - i);
                for (int j = n; j < n_limit; j += NR) {
                    int cur_nr = std::min(NR, n_limit - j);

                    // Pre-calculate masks for N-dimension tails
                    auto get_mask = [](int rem) -> __mmask16 {
                        if (rem <= 0) return 0x0000;
                        if (rem >= 16) return 0xFFFF;
                        return (1U << rem) - 1;
                    };

                    __mmask16 m0 = get_mask(cur_nr);
                    __mmask16 m1 = get_mask(cur_nr - 16);
                    __mmask16 m2 = get_mask(cur_nr - 32);
                    __mmask16 m3 = get_mask(cur_nr - 48);

                    // Register Accumulators: 6 rows x 4 ZMM registers
                    __m512 c[6][4];

                    // Initialize accumulators by loading existing C values
                    for (int r = 0; r < cur_mr; ++r) {
                        c[r][0] = _mm512_maskz_loadu_ps(m0, &C[(i + r) * ldc + j]);
                        c[r][1] = _mm512_maskz_loadu_ps(m1, &C[(i + r) * ldc + j + 16]);
                        c[r][2] = _mm512_maskz_loadu_ps(m2, &C[(i + r) * ldc + j + 32]);
                        c[r][3] = _mm512_maskz_loadu_ps(m3, &C[(i + r) * ldc + j + 48]);
                    }
                    // Safety initialization for unused rows in the micro-kernel
                    for (int r = cur_mr; r < 6; ++r) {
                        for (int v = 0; v < 4; ++v) c[r][v] = _mm512_setzero_ps();
                    }

                    // Process K dimension in tiles to stay within cache levels
                    for (int k_block = 0; k_block < K; k_block += BK) {
                        int k_limit = std::min(k_block + BK, K);

                        for (int k_idx = k_block; k_idx < k_limit; ++k_idx) {
                            // Load micro-panel of B (1x64 block)
                            __m512 b0 = _mm512_maskz_loadu_ps(m0, &B[k_idx * ldb + j]);
                            __m512 b1 = _mm512_maskz_loadu_ps(m1, &B[k_idx * ldb + j + 16]);
                            __m512 b2 = _mm512_maskz_loadu_ps(m2, &B[k_idx * ldb + j + 32]);
                            __m512 b3 = _mm512_maskz_loadu_ps(m3, &B[k_idx * ldb + j + 48]);

                            // Rank-1 update of the micro-block
                            // Manually unrolled for row indices 0..5
                            {
                                __m512 a0 = _mm512_set1_ps(A[(i + 0) * lda + k_idx]);
                                c[0][0] = _mm512_fmadd_ps(a0, b0, c[0][0]);
                                c[0][1] = _mm512_fmadd_ps(a0, b1, c[0][1]);
                                c[0][2] = _mm512_fmadd_ps(a0, b2, c[0][2]);
                                c[0][3] = _mm512_fmadd_ps(a0, b3, c[0][3]);
                            }
                            if (cur_mr > 1) {
                                __m512 a1 = _mm512_set1_ps(A[(i + 1) * lda + k_idx]);
                                c[1][0] = _mm512_fmadd_ps(a1, b0, c[1][0]);
                                c[1][1] = _mm512_fmadd_ps(a1, b1, c[1][1]);
                                c[1][2] = _mm512_fmadd_ps(a1, b2, c[1][2]);
                                c[1][3] = _mm512_fmadd_ps(a1, b3, c[1][3]);
                            }
                            if (cur_mr > 2) {
                                __m512 a2 = _mm512_set1_ps(A[(i + 2) * lda + k_idx]);
                                c[2][0] = _mm512_fmadd_ps(a2, b0, c[2][0]);
                                c[2][1] = _mm512_fmadd_ps(a2, b1, c[2][1]);
                                c[2][2] = _mm512_fmadd_ps(a2, b2, c[2][2]);
                                c[2][3] = _mm512_fmadd_ps(a2, b3, c[2][3]);
                            }
                            if (cur_mr > 3) {
                                __m512 a3 = _mm512_set1_ps(A[(i + 3) * lda + k_idx]);
                                c[3][0] = _mm512_fmadd_ps(a3, b0, c[3][0]);
                                c[3][1] = _mm512_fmadd_ps(a3, b1, c[3][1]);
                                c[3][2] = _mm512_fmadd_ps(a3, b2, c[3][2]);
                                c[3][3] = _mm512_fmadd_ps(a3, b3, c[3][3]);
                            }
                            if (cur_mr > 4) {
                                __m512 a4 = _mm512_set1_ps(A[(i + 4) * lda + k_idx]);
                                c[4][0] = _mm512_fmadd_ps(a4, b0, c[4][0]);
                                c[4][1] = _mm512_fmadd_ps(a4, b1, c[4][1]);
                                c[4][2] = _mm512_fmadd_ps(a4, b2, c[4][2]);
                                c[4][3] = _mm512_fmadd_ps(a4, b3, c[4][3]);
                            }
                            if (cur_mr > 5) {
                                __m512 a5 = _mm512_set1_ps(A[(i + 5) * lda + k_idx]);
                                c[5][0] = _mm512_fmadd_ps(a5, b0, c[5][0]);
                                c[5][1] = _mm512_fmadd_ps(a5, b1, c[5][1]);
                                c[5][2] = _mm512_fmadd_ps(a5, b2, c[5][2]);
                                c[5][3] = _mm512_fmadd_ps(a5, b3, c[5][3]);
                            }
                        }
                    }

                    // Store results back to memory ONCE after full K-dimension accumulation
                    for (int r = 0; r < cur_mr; ++r) {
                        _mm512_mask_storeu_ps(&C[(i + r) * ldc + j],      m0, c[r][0]);
                        _mm512_mask_storeu_ps(&C[(i + r) * ldc + j + 16], m1, c[r][1]);
                        _mm512_mask_storeu_ps(&C[(i + r) * ldc + j + 32], m2, c[r][2]);
                        _mm512_mask_storeu_ps(&C[(i + r) * ldc + j + 48], m3, c[r][3]);
                    }
                }
            }
        }
    }
#else
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

/**
 * @brief Helper to write matrix results.
 */
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;
    ofs << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump_matrices = false;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--dump-matrices") dump_matrices = true;
    }

    // Alignment is vital for AVX-512
    float* A = (float*)aligned_alloc(64, (size_t)M * K * sizeof(float));
    float* B = (float*)aligned_alloc(64, (size_t)K * N * sizeof(float));
    float* C = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));

    // Parallel "First Touch" Initialization
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 gen(42 + tid);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        #pragma omp for nowait
        for (int i = 0; i < M * K; ++i) A[i] = dist(gen);

        #pragma omp for nowait
        for (int i = 0; i < K * N; ++i) B[i] = dist(gen);

        #pragma omp for
        for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
    }

    if (dump_matrices) {
        float* C_ref = (float*)aligned_alloc(64, (size_t)M * N * sizeof(float));
        #pragma omp parallel for
        for (int i = 0; i < M * N; ++i) C_ref[i] = 0.0f;

        write_matrix_to_file("workspace/A.txt", A, M, K, K);
        write_matrix_to_file("workspace/B.txt", B, K, N, N);

        gemm_scalar(A, B, C_ref, M, N, K, K, N, N);
        gemm_avx512(A, B, C, M, N, K, K, N, N);

        write_matrix_to_file("workspace/C.txt", C, M, N, N);

        double max_diff = 0.0;
        for (int i = 0; i < M * N; ++i) {
            max_diff = std::max(max_diff, (double)std::abs(C[i] - C_ref[i]));
        }
        std::cout << "Internal check: " << (max_diff < 1e-3 ? "PASSED" : "FAILED") 
                  << " (Max diff: " << max_diff << ")" << std::endl;
        free(C_ref);
    } else {
        auto start = std::chrono::high_resolution_clock::now();
        gemm_avx512(A, B, C, M, N, K, K, N, N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);

        std::cout << "M: " << M << ", N: " << N << ", K: " << K << "\n";
        std::cout << "Time: " << std::fixed << std::setprecision(6) << diff.count() << " s\n";
        std::cout << "GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
    }

    free(A);
    free(B);
    free(C);
    return 0;
}