/*
 * Optimization for AMD EPYC 9365 (Zen 4)
 * 
 * Compile Instructions:
 * GCC:   g++ -O3 -march=x86-64-v3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
 * Clang: clang++ -O3 -march=x86-64-v3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
 *
 * Strategies used:
 * 1. Register Blocking: 6x64 micro-kernel utilizing 24 AVX-512 accumulators.
 * 2. Cache-aware Tiling: BM=120, BN=512, BK=256 to fit tiles into L2/L3 caches.
 * 3. Memory Packing: Matrix B is packed into a contiguous format [N/64][K][64] to linearize access.
 * 4. Parallelization: OpenMP used to parallelize M and N loops.
 * 5. Masked Stores: AVX-512 masking handles tail dimensions in the N-axis.
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
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// Autotuning Parameters
constexpr int BM = 120;  // M-tile size (multiple of 6)
constexpr int BN = 512;  // N-tile size (multiple of 64)
constexpr int BK = 256;  // K-tile size
constexpr int UNROLL_K = 4; // Inner unrolling factor

// Forward declarations
void gemm_scalar(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc);

void gemm_avx512(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc);

// Helper for saving results
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            out << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        out << "\n";
    }
    out.close();
}

/**
 * Reference Scalar Implementation
 * Row-major: C[i*ldc + j] += A[i*lda + k] * B[k*ldb + j]
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

#if defined(__AVX512F__)
/**
 * Packing function for Matrix B.
 * Reorganizes B from (K, N) to blocks of (BK, 64) to ensure contiguous loads in micro-kernel.
 */
inline void pack_B_tile(int K_start, int K_end, int N_start, int N_end, int ldb, const float* B, float* packed_B) {
    int K_size = K_end - K_start;
    for (int j_block = N_start; j_block < N_end; j_block += 64) {
        float* dst = packed_B + (j_block - N_start) * K_size;
        for (int k = K_start; k < K_end; ++k) {
            int j_limit = std::min(j_block + 64, N_end);
            int j = j_block;
            // Full 16-float vectors
            for (; j <= j_limit - 16; j += 16) {
                _mm512_storeu_ps(dst, _mm512_loadu_ps(&B[k * ldb + j]));
                dst += 16;
            }
            // Tail handling within the 64-block
            if (j < j_limit) {
                int rem = j_limit - j;
                __mmask16 mask = (1U << rem) - 1;
                _mm512_mask_storeu_ps(dst, mask, _mm512_maskz_loadu_ps(mask, &B[k * ldb + j]));
                dst += 16; 
                j += 16;
            }
            // Pad the rest of the 64-float chunk if N_limit was reached early
            for (; j < j_block + 64; j += 16) {
                _mm512_storeu_ps(dst, _mm512_setzero_ps());
                dst += 16;
            }
        }
    }
}

/**
 * AVX-512 Optimized Kernel
 * 6x64 Micro-kernel
 */
void gemm_avx512(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc) {
    
    // Allocate buffer for packing B tiles. Each thread or the whole N-panel?
    // Given the high thread count, we'll pack B panels in the N-dimension.
    float* packed_B = (float*)_mm_malloc(BK * ((N + 63) / 64) * 64 * sizeof(float), 64);

    for (int ko = 0; ko < K; ko += BK) {
        int kend = std::min(ko + BK, K);
        int k_len = kend - ko;

        // Pack the current horizontal panel of B
        pack_B_tile(ko, kend, 0, N, ldb, B, packed_B);

        #pragma omp parallel for collapse(2) schedule(static)
        for (int mo = 0; mo < M; mo += BM) {
            for (int no = 0; no < N; no += BN) {
                int mend = std::min(mo + BM, M);
                int nend = std::min(no + BN, N);

                for (int i = mo; i < mend; i += 6) {
                    int i_rem = std::min(6, mend - i);
                    for (int j = no; j < nend; j += 64) {
                        // Register blocking: 6 rows of A, 64 columns of B (4 ZMMs per row)
                        __m512 acc0[4], acc1[4], acc2[4], acc3[4], acc4[4], acc5[4];
                        
                        // Initialize accumulators
                        if (ko == 0) {
                            for(int v=0; v<4; ++v) {
                                acc0[v] = _mm512_setzero_ps(); acc1[v] = _mm512_setzero_ps();
                                acc2[v] = _mm512_setzero_ps(); acc3[v] = _mm512_setzero_ps();
                                acc4[v] = _mm512_setzero_ps(); acc5[v] = _mm512_setzero_ps();
                            }
                        } else {
                            // Load existing C values if not first K-tile
                            for(int v=0; v<4; ++v) {
                                int col = j + v*16;
                                if (col < N) {
                                    int rem = N - col;
                                    __mmask16 msk = (rem >= 16) ? 0xFFFF : (1U << rem) - 1;
                                    acc0[v] = (i+0 < M) ? _mm512_maskz_loadu_ps(msk, &C[(i+0)*ldc + col]) : _mm512_setzero_ps();
                                    acc1[v] = (i+1 < M) ? _mm512_maskz_loadu_ps(msk, &C[(i+1)*ldc + col]) : _mm512_setzero_ps();
                                    acc2[v] = (i+2 < M) ? _mm512_maskz_loadu_ps(msk, &C[(i+2)*ldc + col]) : _mm512_setzero_ps();
                                    acc3[v] = (i+3 < M) ? _mm512_maskz_loadu_ps(msk, &C[(i+3)*ldc + col]) : _mm512_setzero_ps();
                                    acc4[v] = (i+4 < M) ? _mm512_maskz_loadu_ps(msk, &C[(i+4)*ldc + col]) : _mm512_setzero_ps();
                                    acc5[v] = (i+5 < M) ? _mm512_maskz_loadu_ps(msk, &C[(i+5)*ldc + col]) : _mm512_setzero_ps();
                                } else {
                                    acc0[v] = acc1[v] = acc2[v] = acc3[v] = acc4[v] = acc5[v] = _mm512_setzero_ps();
                                }
                            }
                        }

                        const float* pB = packed_B + j * k_len;
                        const float* pA = &A[i * lda + ko];

                        for (int k = 0; k < k_len; ++k) {
                            __m512 b0 = _mm512_load_ps(pB + k*64 + 0);
                            __m512 b1 = _mm512_load_ps(pB + k*64 + 16);
                            __m512 b2 = _mm512_load_ps(pB + k*64 + 32);
                            __m512 b3 = _mm512_load_ps(pB + k*64 + 48);

                            if (i_rem >= 1) {
                                __m512 a0 = _mm512_set1_ps(pA[0*lda + k]);
                                acc0[0] = _mm512_fmadd_ps(a0, b0, acc0[0]);
                                acc0[1] = _mm512_fmadd_ps(a0, b1, acc0[1]);
                                acc0[2] = _mm512_fmadd_ps(a0, b2, acc0[2]);
                                acc0[3] = _mm512_fmadd_ps(a0, b3, acc0[3]);
                            }
                            if (i_rem >= 2) {
                                __m512 a1 = _mm512_set1_ps(pA[1*lda + k]);
                                acc1[0] = _mm512_fmadd_ps(a1, b0, acc1[0]);
                                acc1[1] = _mm512_fmadd_ps(a1, b1, acc1[1]);
                                acc1[2] = _mm512_fmadd_ps(a1, b2, acc1[2]);
                                acc1[3] = _mm512_fmadd_ps(a1, b3, acc1[3]);
                            }
                            if (i_rem >= 3) {
                                __m512 a2 = _mm512_set1_ps(pA[2*lda + k]);
                                acc2[0] = _mm512_fmadd_ps(a2, b0, acc2[0]);
                                acc2[1] = _mm512_fmadd_ps(a2, b1, acc2[1]);
                                acc2[2] = _mm512_fmadd_ps(a2, b2, acc2[2]);
                                acc2[3] = _mm512_fmadd_ps(a2, b3, acc2[3]);
                            }
                            if (i_rem >= 4) {
                                __m512 a3 = _mm512_set1_ps(pA[3*lda + k]);
                                acc3[0] = _mm512_fmadd_ps(a3, b0, acc3[0]);
                                acc3[1] = _mm512_fmadd_ps(a3, b1, acc3[1]);
                                acc3[2] = _mm512_fmadd_ps(a3, b2, acc3[2]);
                                acc3[3] = _mm512_fmadd_ps(a3, b3, acc3[3]);
                            }
                            if (i_rem >= 5) {
                                __m512 a4 = _mm512_set1_ps(pA[4*lda + k]);
                                acc4[0] = _mm512_fmadd_ps(a4, b0, acc4[0]);
                                acc4[1] = _mm512_fmadd_ps(a4, b1, acc4[1]);
                                acc4[2] = _mm512_fmadd_ps(a4, b2, acc4[2]);
                                acc4[3] = _mm512_fmadd_ps(a4, b3, acc4[3]);
                            }
                            if (i_rem >= 6) {
                                __m512 a5 = _mm512_set1_ps(pA[5*lda + k]);
                                acc5[0] = _mm512_fmadd_ps(a5, b0, acc5[0]);
                                acc5[1] = _mm512_fmadd_ps(a5, b1, acc5[1]);
                                acc5[2] = _mm512_fmadd_ps(a5, b2, acc5[2]);
                                acc5[3] = _mm512_fmadd_ps(a5, b3, acc5[3]);
                            }
                        }

                        // Store results back to C with masking
                        for (int v = 0; v < 4; ++v) {
                            int col = j + v * 16;
                            if (col < N) {
                                int rem = N - col;
                                __mmask16 msk = (rem >= 16) ? 0xFFFF : (1U << rem) - 1;
                                if (i + 0 < M) _mm512_mask_storeu_ps(&C[(i+0)*ldc + col], msk, acc0[v]);
                                if (i + 1 < M) _mm512_mask_storeu_ps(&C[(i+1)*ldc + col], msk, acc1[v]);
                                if (i + 2 < M) _mm512_mask_storeu_ps(&C[(i+2)*ldc + col], msk, acc2[v]);
                                if (i + 3 < M) _mm512_mask_storeu_ps(&C[(i+3)*ldc + col], msk, acc3[v]);
                                if (i + 4 < M) _mm512_mask_storeu_ps(&C[(i+4)*ldc + col], msk, acc4[v]);
                                if (i + 5 < M) _mm512_mask_storeu_ps(&C[(i+5)*ldc + col], msk, acc5[v]);
                            }
                        }
                    }
                }
            }
        }
    }
    _mm_free(packed_B);
}
#else
void gemm_avx512(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc) {
    std::cerr << "AVX-512 is not supported on this architecture." << std::endl;
}
#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    bool dump_matrices = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--dump-matrices") {
            dump_matrices = true;
        }
    }

    // Aligned allocation for matrices
    float* A = (float*)_mm_malloc(M * K * sizeof(float), 64);
    float* B = (float*)_mm_malloc(K * N * sizeof(float), 64);
    float* C = (float*)_mm_malloc(M * N * sizeof(float), 64);

    // Initialize with random data
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < M * K; ++i) A[i] = dist(gen);
    for (int i = 0; i < K * N; ++i) B[i] = dist(gen);
    std::memset(C, 0, M * N * sizeof(float));

    if (dump_matrices) {
        // Test Mode
        float* C_ref = (float*)_mm_malloc(M * N * sizeof(float), 64);
        std::memset(C_ref, 0, M * N * sizeof(float));

        // Create workspace directory if it doesn't exist (platform dependent, assuming it exists per prompt)
        write_matrix_to_file("workspace/A.txt", A, M, K, K);
        write_matrix_to_file("workspace/B.txt", B, K, N, N);

        // Reference
        gemm_scalar(A, B, C_ref, M, N, K, K, N, N);

        // Optimized
        gemm_avx512(A, B, C, M, N, K, K, N, N);
        write_matrix_to_file("workspace/C.txt", C, M, N, N);

        // Correctness check
        bool passed = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::abs(C[i] - C_ref[i]) > 1e-3f) {
                passed = false;
                break;
            }
        }
        std::cout << "Internal check: " << (passed ? "PASSED" : "FAILED") << std::endl;
        _mm_free(C_ref);
    } else {
        // Perf Mode
        auto start = std::chrono::high_resolution_clock::now();
        gemm_avx512(A, B, C, M, N, K, K, N, N);
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> diff = end - start;
        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);
        std::cout << "Time: " << diff.count() << " s, GFLOPS: " << gflops << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}