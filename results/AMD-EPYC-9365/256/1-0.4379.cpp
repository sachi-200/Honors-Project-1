/*
 * Optimization for AMD EPYC 9365 (Zen 4/5)
 * 
 * Strategy:
 * 1. SIMD: AVX-512 FMA for maximum throughput.
 * 2. Register Blocking: 8x48 micro-kernel (8 rows, 3 AVX-512 vectors wide).
 *    - This uses 24 registers for accumulation, 3 for B-loads, and 1 for A-broadcasts.
 * 3. Cache Blocking: L1/L2/L3 aware tiling (BM, BN, BK).
 * 4. NUMA/Parallelism: OpenMP with parallel initialization (First-Touch) for 144 threads.
 * 5. Memory: Row-major layout assumed.
 * 
 * Build:
 * g++ -O3 -march=x86-64-v4 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
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

// --- Autotuning Parameters ---
constexpr int BM = 128;  // Tile size in M
constexpr int BN = 192;  // Tile size in N
constexpr int BK = 256;  // Tile size in K
constexpr int UNROLL_K = 1; 

// Micro-kernel sizes (Register Blocking)
// Zen 4 has 32 ZMM registers. We use 8x48 (8 rows, 3 vectors of 16 floats).
// Accumulators: 8 * 3 = 24 ZMMs.
// Temporary B: 3 ZMMs.
// Temporary A: 1 ZMM.
// Total: 28/32 ZMMs used.
constexpr int RM = 8;
constexpr int RN = 48; // 3 * 16

/**
 * Reference Scalar GEMM
 * C = A * B + C
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
 * Optimized AVX-512 GEMM
 * Logic: Tiled parallel loops over M and N, then a serial loop over K.
 * Within tiles, uses a high-performance register-blocked micro-kernel.
 */
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__)
    // We parallelize over the M and N blocks
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m0 = 0; m0 < M; m0 += BM) {
        for (int n0 = 0; n0 < N; n0 += BN) {
            int m_limit = (m0 + BM < M) ? m0 + BM : M;
            int n_limit = (n0 + BN < N) ? n0 + BN : N;

            for (int k0 = 0; k0 < K; k0 += BK) {
                int k_limit = (k0 + BK < K) ? k0 + BK : K;

                // Micro-kernel loops
                for (int i = m0; i < m_limit; i += RM) {
                    for (int j = n0; j < n_limit; j += RN) {
                        
                        // Handle boundaries for RM and RN
                        if (i + RM <= m_limit && j + RN <= n_limit) {
                            // Fast Path: Full 8x48 register block
                            __m512 c[8][3];
                            
                            // Load accumulators
                            for (int r = 0; r < 8; ++r) {
                                c[r][0] = _mm512_loadu_ps(&C[(i + r) * ldc + j]);
                                c[r][1] = _mm512_loadu_ps(&C[(i + r) * ldc + j + 16]);
                                c[r][2] = _mm512_loadu_ps(&C[(i + r) * ldc + j + 32]);
                            }

                            for (int k = k0; k < k_limit; ++k) {
                                __m512 b0 = _mm512_loadu_ps(&B[k * ldb + j]);
                                __m512 b1 = _mm512_loadu_ps(&B[k * ldb + j + 16]);
                                __m512 b2 = _mm512_loadu_ps(&B[k * ldb + j + 32]);

                                for (int r = 0; r < 8; ++r) {
                                    __m512 a = _mm512_set1_ps(A[(i + r) * lda + k]);
                                    c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
                                    c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
                                    c[r][2] = _mm512_fmadd_ps(a, b2, c[r][2]);
                                }
                            }

                            // Store accumulators
                            for (int r = 0; r < 8; ++r) {
                                _mm512_storeu_ps(&C[(i + r) * ldc + j], c[r][0]);
                                _mm512_storeu_ps(&C[(i + r) * ldc + j + 16], c[r][1]);
                                _mm512_storeu_ps(&C[(i + r) * ldc + j + 32], c[r][2]);
                            }
                        } else {
                            // Slow Path: Scalar tails for blocks not fitting RMxRN
                            for (int ii = i; ii < (i + RM < m_limit ? i + RM : m_limit); ++ii) {
                                for (int kk = k0; kk < k_limit; ++kk) {
                                    float a_val = A[ii * lda + kk];
                                    for (int jj = j; jj < (j + RN < n_limit ? j + RN : n_limit); ++jj) {
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
    // Fallback if compiled without AVX-512 support
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) return;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << std::fixed << std::setprecision(4) << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
}

int main(int argc, char** argv) {
    int M = 512, N = 512, K = 512;
    bool dump_matrices = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (i == 1) M = std::stoi(arg);
        else if (i == 2) N = std::stoi(arg);
        else if (i == 3) K = std::stoi(arg);
    }

    int lda = K, ldb = N, ldc = N;
    size_t size_a = (size_t)M * lda;
    size_t size_b = (size_t)K * ldb;
    size_t size_c = (size_t)M * ldc;

    // Use aligned allocation for SIMD
    float* A = (float*)_mm_malloc(size_a * sizeof(float), 64);
    float* B = (float*)_mm_malloc(size_b * sizeof(float), 64);
    float* C = (float*)_mm_malloc(size_c * sizeof(float), 64);

    // NUMA-aware initialization (First Touch)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size_a; ++i) A[i] = (float)(rand() % 100) / 10.0f;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size_b; ++i) B[i] = (float)(rand() % 100) / 10.0f;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size_c; ++i) C[i] = 0.0f;

    if (dump_matrices) {
        float* C_ref = (float*)_mm_malloc(size_c * sizeof(float), 64);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size_c; ++i) C_ref[i] = 0.0f;

        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        // Run reference
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

        // Run optimized
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);

        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

        // Check correctness
        bool passed = true;
        for (size_t i = 0; i < size_c; ++i) {
            if (std::abs(C[i] - C_ref[i]) > 1e-2) {
                passed = false;
                break;
            }
        }
        std::cout << "Internal check: " << (passed ? "PASSED" : "FAILED") << std::endl;

        _mm_free(C_ref);
    } else {
        // Performance Mode
        auto start = std::chrono::high_resolution_clock::now();
        
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);
        std::cout << "M: " << M << " N: " << N << " K: " << K << " Time: " << diff.count() << "s GFLOPS: " << gflops << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}