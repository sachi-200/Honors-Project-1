/*
 * Optimization for AMD EPYC 9365 (Zen 4/5) - 144 Threads, AVX-512
 * 
 * Compile instructions:
 * GCC: g++ -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mfma -fopenmp gemm.cpp -o gemm
 * Clang: clang++ -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mfma -fopenmp gemm.cpp -o gemm
 *
 * Strategies:
 * 1. AVX-512 Micro-kernel: 6x64 register blocking (6 rows of A, 4 ZMM registers for B columns).
 * 2. Tiling: BM=48, BN=384, BK=256 to fit L1/L2 caches and maximize L3 reuse.
 * 3. Parallelism: OpenMP collapse(2) on M and N tiles for 144-thread distribution.
 * 4. NUMA: Parallel initialization (First Touch) to pin memory pages to local NUMA nodes.
 * 5. Packing: B-panels are packed into contiguous memory for linear SIMD access.
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

// Autotuning Parameters
#ifndef BM
#define BM 48
#endif
#ifndef BN
#define BN 384
#endif
#ifndef BK
#define BK 256
#endif
#ifndef UNROLL_K
#define UNROLL_K 4
#endif

// Function Signatures
void gemm_scalar(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc);

void gemm_avx512(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc);

// Helper for matrix storage
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream f(filename);
    if (!f.is_open()) return;
    f << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            f << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        f << "\n";
    }
}

// Reference Scalar Implementation
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

#if defined(__AVX512F__)
// Micro-kernel for 6 rows x 64 columns (4 ZMM registers wide)
inline void micro_kernel_6x64(const float* A_ptr, const float* B_packed, float* C_ptr, int ldc, int K_rem, int lda) {
    __m512 c00 = _mm512_setzero_ps(); __m512 c01 = _mm512_setzero_ps(); __m512 c02 = _mm512_setzero_ps(); __m512 c03 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(); __m512 c11 = _mm512_setzero_ps(); __m512 c12 = _mm512_setzero_ps(); __m512 c13 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(); __m512 c21 = _mm512_setzero_ps(); __m512 c22 = _mm512_setzero_ps(); __m512 c23 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(); __m512 c31 = _mm512_setzero_ps(); __m512 c32 = _mm512_setzero_ps(); __m512 c33 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps(); __m512 c41 = _mm512_setzero_ps(); __m512 c42 = _mm512_setzero_ps(); __m512 c43 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps(); __m512 c51 = _mm512_setzero_ps(); __m512 c52 = _mm512_setzero_ps(); __m512 c53 = _mm512_setzero_ps();

    for (int k = 0; k < K_rem; ++k) {
        __m512 b0 = _mm512_loadu_ps(B_packed + k * 64 + 0);
        __m512 b1 = _mm512_loadu_ps(B_packed + k * 64 + 16);
        __m512 b2 = _mm512_loadu_ps(B_packed + k * 64 + 32);
        __m512 b3 = _mm512_loadu_ps(B_packed + k * 64 + 48);

        __m512 a;
        a = _mm512_set1_ps(A_ptr[0 * lda + k]);
        c00 = _mm512_fmadd_ps(a, b0, c00); c01 = _mm512_fmadd_ps(a, b1, c01); c02 = _mm512_fmadd_ps(a, b2, c02); c03 = _mm512_fmadd_ps(a, b3, c03);
        
        a = _mm512_set1_ps(A_ptr[1 * lda + k]);
        c10 = _mm512_fmadd_ps(a, b0, c10); c11 = _mm512_fmadd_ps(a, b1, c11); c12 = _mm512_fmadd_ps(a, b2, c12); c13 = _mm512_fmadd_ps(a, b3, c13);
        
        a = _mm512_set1_ps(A_ptr[2 * lda + k]);
        c20 = _mm512_fmadd_ps(a, b0, c20); c21 = _mm512_fmadd_ps(a, b1, c21); c22 = _mm512_fmadd_ps(a, b2, c22); c23 = _mm512_fmadd_ps(a, b3, c23);
        
        a = _mm512_set1_ps(A_ptr[3 * lda + k]);
        c30 = _mm512_fmadd_ps(a, b0, c30); c31 = _mm512_fmadd_ps(a, b1, c31); c32 = _mm512_fmadd_ps(a, b2, c32); c33 = _mm512_fmadd_ps(a, b3, c33);
        
        a = _mm512_set1_ps(A_ptr[4 * lda + k]);
        c40 = _mm512_fmadd_ps(a, b0, c40); c41 = _mm512_fmadd_ps(a, b1, c41); c42 = _mm512_fmadd_ps(a, b2, c42); c43 = _mm512_fmadd_ps(a, b3, c43);
        
        a = _mm512_set1_ps(A_ptr[5 * lda + k]);
        c50 = _mm512_fmadd_ps(a, b0, c50); c51 = _mm512_fmadd_ps(a, b1, c51); c52 = _mm512_fmadd_ps(a, b2, c52); c53 = _mm512_fmadd_ps(a, b3, c53);
    }

    auto update_c = [&](int row, __m512 r0, __m512 r1, __m512 r2, __m512 r3) {
        float* p = C_ptr + row * ldc;
        _mm512_storeu_ps(p + 0, _mm512_add_ps(_mm512_loadu_ps(p + 0), r0));
        _mm512_storeu_ps(p + 16, _mm512_add_ps(_mm512_loadu_ps(p + 16), r1));
        _mm512_storeu_ps(p + 32, _mm512_add_ps(_mm512_loadu_ps(p + 32), r2));
        _mm512_storeu_ps(p + 48, _mm512_add_ps(_mm512_loadu_ps(p + 48), r3));
    };

    update_c(0, c00, c01, c02, c03);
    update_c(1, c10, c11, c12, c13);
    update_c(2, c20, c21, c22, c23);
    update_c(3, c30, c31, c32, c33);
    update_c(4, c40, c41, c42, c43);
    update_c(5, c50, c51, c52, c53);
}

// Optimized AVX-512 GEMM
void gemm_avx512(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc) {
    
    // Distribute M and N across threads
    #pragma omp parallel
    {
        // Thread-local packing buffer for B-tile (BN x BK)
        // BK is the depth, 64 columns per micro-step
        float* B_packed = (float*)_mm_malloc(BK * BN * sizeof(float), 64);

        #pragma omp for collapse(2) schedule(static)
        for (int m = 0; m < M; m += BM) {
            for (int n = 0; n < N; n += BN) {
                int m_limit = (m + BM > M) ? M : m + BM;
                int n_limit = (n + BN > N) ? N : n + BN;

                for (int k = 0; k < K; k += BK) {
                    int k_limit = (k + BK > K) ? K : k + BK;
                    int k_len = k_limit - k;

                    // Pack B tile into contiguous memory for the N-range
                    // This improves cache access and enables contiguous SIMD loads
                    for (int kb = 0; kb < k_len; ++kb) {
                        int current_k = k + kb;
                        for (int nb = n; nb < n_limit; nb += 64) {
                            int pack_idx = ((nb - n) / 64) * (k_len * 64) + kb * 64;
                            int j = 0;
                            // Vectorized pack if possible
                            for (; j <= (n_limit - nb - 16) && j < 64; j += 16) {
                                _mm512_storeu_ps(B_packed + pack_idx + j, _mm512_loadu_ps(B + current_k * ldb + nb + j));
                            }
                            // Tail pack
                            for (; j < 64 && (nb + j) < n_limit; ++j) {
                                B_packed[pack_idx + j] = B[current_k * ldb + nb + j];
                            }
                            // Zero-pad remainder of 64-block to avoid garbage in micro-kernel FMAs
                            for (; j < 64; ++j) {
                                B_packed[pack_idx + j] = 0.0f;
                            }
                        }
                    }

                    // Micro-kernels
                    for (int mi = m; mi < m_limit; mi += 6) {
                        if (mi + 6 <= m_limit) {
                            for (int ni = n; ni < n_limit; ni += 64) {
                                if (ni + 64 <= n_limit) {
                                    int pack_offset = ((ni - n) / 64) * (k_len * 64);
                                    micro_kernel_6x64(A + mi * lda + k, B_packed + pack_offset, C + mi * ldc + ni, ldc, k_len, lda);
                                } else {
                                    // Tail N: Scalar fallback for the block
                                    for (int r = 0; r < 6; ++r) {
                                        for (int kk = 0; kk < k_len; ++kk) {
                                            float a_val = A[(mi + r) * lda + (k + kk)];
                                            for (int jj = ni; jj < n_limit; ++jj) {
                                                C[(mi + r) * ldc + jj] += a_val * B[(k + kk) * ldb + jj];
                                            }
                                        }
                                    }
                                }
                            }
                        } else {
                            // Tail M: Scalar fallback
                            for (int mi_tail = mi; mi_tail < m_limit; ++mi_tail) {
                                for (int kk = 0; kk < k_len; ++kk) {
                                    float a_val = A[mi_tail * lda + (k + kk)];
                                    for (int jj = n; jj < n_limit; ++jj) {
                                        C[mi_tail * ldc + jj] += a_val * B[(k + kk) * ldb + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _mm_free(B_packed);
    }
}
#else
void gemm_avx512(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc) {
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump_matrices = false;
    if (argc > 4 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    int lda = K;
    int ldb = N;
    int ldc = N;

    // NUMA First-Touch Allocation
    float* A = (float*)_mm_malloc((size_t)M * K * sizeof(float), 64);
    float* B = (float*)_mm_malloc((size_t)K * N * sizeof(float), 64);
    float* C = (float*)_mm_malloc((size_t)M * N * sizeof(float), 64);

    #pragma omp parallel
    {
        unsigned int seed = 42 + omp_get_thread_num();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // First touch for A
        #pragma omp for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < K; ++j) {
                A[i * lda + j] = dist(gen);
            }
        }
        // First touch for B
        #pragma omp for
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < N; ++j) {
                B[i * ldb + j] = dist(gen);
            }
        }
        // First touch for C
        #pragma omp for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] = 0.0f;
            }
        }
    }

    if (dump_matrices) {
        float* C_ref = (float*)_mm_malloc((size_t)M * N * sizeof(float), 64);
        std::memset(C_ref, 0, (size_t)M * N * sizeof(float));

        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);

        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

        bool passed = true;
        for (size_t i = 0; i < (size_t)M * N; ++i) {
            if (std::abs(C[i] - C_ref[i]) > 1e-3f) {
                passed = false;
                break;
            }
        }
        std::cout << "Internal check: " << (passed ? "PASSED" : "FAILED") << std::endl;
        _mm_free(C_ref);
    } else {
        auto start = std::chrono::high_resolution_clock::now();
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);
        std::cout << "Performance: " << diff.count() << " s, " << gflops << " GFLOPS" << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}