// Compile with:
// g++ -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mfma -fopenmp gemm.cpp -o gemm

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

/**
 * Optimization Strategy:
 * 1. Register Blocking: 8x32 micro-kernel using 16 ZMM registers for accumulators (8 rows, 2 columns of 16-floats).
 * 2. Cache Blocking: BM=128, BN=128, BK=256 to fit data into L2 cache (1MB on Zen 4).
 * 3. Packing: B-panels are packed into contiguous memory for each tile to ensure unit-stride SIMD loads.
 * 4. Parallelization: OpenMP used on M and N tile loops with 'collapse(2)'.
 * 5. NUMA: Parallel initialization of matrices (First Touch) ensures data is localized to NUMA nodes.
 * 6. ISA: Full AVX-512 with FMA for the inner computation loop.
 */

// Autotuning parameters
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 256;
constexpr int MR = 8;  // Micro-kernel rows
constexpr int NR = 32; // Micro-kernel cols (2 * 16 floats)

void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    ofs << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

#if defined(__AVX512F__)
// Helper to pack a block of B into contiguous memory
// Layout: [BN/32][BK][32]
static inline void pack_B(int K_tile, int N_tile, const float* B_start, int ldb, float* B_packed) {
    for (int j0 = 0; j0 < N_tile; j0 += NR) {
        int rem_n = (N_tile - j0 < NR) ? (N_tile - j0) : NR;
        for (int k = 0; k < K_tile; ++k) {
            const float* src = B_start + k * ldb + j0;
            float* dst = B_packed + (j0 / NR) * BK * NR + k * NR;
            if (rem_n == NR) {
                _mm512_storeu_ps(dst, _mm512_loadu_ps(src));
                _mm512_storeu_ps(dst + 16, _mm512_loadu_ps(src + 16));
            } else {
                // Scalar tail for N
                for (int jj = 0; jj < rem_n; ++jj) dst[jj] = src[jj];
                for (int jj = rem_n; jj < NR; ++jj) dst[jj] = 0.0f;
            }
        }
    }
}

void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    
    #pragma omp parallel
    {
        // Thread-local packing buffer for B tile
        alignas(64) float B_tile_packed[BK * BN];

        #pragma omp for collapse(2) schedule(static)
        for (int i0 = 0; i0 < M; i0 += BM) {
            for (int j0 = 0; j0 < N; j0 += BN) {
                int i_limit = (i0 + BM > M) ? M : i0 + BM;
                int j_limit = (j0 + BN > N) ? N : j0 + BN;

                // Initialize C tile block to zero if it's the first K-block (though GEMM usually assumes C += A*B)
                // For simplicity and to match common benchmarks, we set C to zero elsewhere and accumulate here.

                for (int k0 = 0; k0 < K; k0 += BK) {
                    int k_limit = (k0 + BK > K) ? K : k0 + BK;
                    int K_tile = k_limit - k0;
                    int N_tile = j_limit - j0;

                    // Pack B tile to linearize access
                    pack_B(K_tile, N_tile, B + k0 * ldb + j0, ldb, B_tile_packed);

                    // Micro-kernel loops
                    for (int i = i0; i < i_limit; i += MR) {
                        int current_mr = (i_limit - i < MR) ? (i_limit - i) : MR;
                        
                        for (int j = j0; j < j_limit; j += NR) {
                            int j_idx = (j - j0) / NR;
                            float* pB = B_tile_packed + j_idx * BK * NR;
                            
                            __m512 c[8][2];
                            // Load existing C or zero out
                            for (int r = 0; r < current_mr; ++r) {
                                if (k0 == 0) {
                                    c[r][0] = _mm512_setzero_ps();
                                    c[r][1] = _mm512_setzero_ps();
                                } else {
                                    // Handle N tails with masking for correctness
                                    int rem_n = (N - j);
                                    if (rem_n >= 32) {
                                        c[r][0] = _mm512_loadu_ps(C + (i + r) * ldc + j);
                                        c[r][1] = _mm512_loadu_ps(C + (i + r) * ldc + j + 16);
                                    } else {
                                        __mmask16 m0 = (rem_n > 0) ? (0xFFFF >> (16 - (rem_n > 16 ? 16 : rem_n))) : 0;
                                        __mmask16 m1 = (rem_n > 16) ? (0xFFFF >> (32 - rem_n)) : 0;
                                        c[r][0] = _mm512_maskz_loadu_ps(m0, C + (i + r) * ldc + j);
                                        c[r][1] = _mm512_maskz_loadu_ps(m1, C + (i + r) * ldc + j + 16);
                                    }
                                }
                            }

                            // K-loop
                            for (int k = 0; k < K_tile; ++k) {
                                __m512 b0 = _mm512_loadu_ps(pB + k * NR);
                                __m512 b1 = _mm512_loadu_ps(pB + k * NR + 16);
                                
                                for (int r = 0; r < current_mr; ++r) {
                                    __m512 a = _mm512_set1_ps(A[(i + r) * lda + (k0 + k)]);
                                    c[r][0] = _mm512_fmadd_ps(a, b0, c[r][0]);
                                    c[r][1] = _mm512_fmadd_ps(a, b1, c[r][1]);
                                }
                            }

                            // Store results back to C
                            for (int r = 0; r < current_mr; ++r) {
                                int rem_n = (N - j);
                                if (rem_n >= 32) {
                                    _mm512_storeu_ps(C + (i + r) * ldc + j, c[r][0]);
                                    _mm512_storeu_ps(C + (i + r) * ldc + j + 16, c[r][1]);
                                } else {
                                    __mmask16 m0 = (rem_n > 0) ? (0xFFFF >> (16 - (rem_n > 16 ? 16 : rem_n))) : 0;
                                    __mmask16 m1 = (rem_n > 16) ? (0xFFFF >> (32 - rem_n)) : 0;
                                    _mm512_mask_storeu_ps(C + (i + r) * ldc + j, m0, c[r][0]);
                                    _mm512_mask_storeu_ps(C + (i + r) * ldc + j + 16, m1, c[r][1]);
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
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    // Fallback if AVX-512 is not detected at compile time
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    bool dump_matrices = false;
    if (argc > 4 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    // Leading dimensions (Row Major)
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Use aligned allocation for better performance
    float* A = (float*)_mm_malloc(sizeof(float) * M * K, 64);
    float* B = (float*)_mm_malloc(sizeof(float) * K * N, 64);
    float* C = (float*)_mm_malloc(sizeof(float) * M * N, 64);

    // NUMA-aware First Touch initialization
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            A[i * lda + k] = static_cast<float>(i + k % 7) / 13.0f;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            B[k * ldb + j] = static_cast<float>(k + j % 11) / 17.0f;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = 0.0f;
        }
    }

    if (dump_matrices) {
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        float* C_ref = (float*)_mm_malloc(sizeof(float) * M * N, 64);
        std::memset(C_ref, 0, sizeof(float) * M * N);
        
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

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
        auto start = std::chrono::high_resolution_clock::now();
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
