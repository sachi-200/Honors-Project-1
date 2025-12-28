/*
 * Optimization for AMD EPYC 9365 (Zen 4/5)
 * 
 * Strategy:
 * 1. SIMD ISA: AVX-512 with 32 ZMM registers used to implement a 14x32 register-blocked micro-kernel.
 *    - 14 rows of A and 2 vectors (32 floats) of B are processed simultaneously.
 *    - 28 ZMM registers are used as accumulators (14 rows * 2 vectors).
 *    - 2 ZMM registers for B loads, 1 ZMM for A broadcasts. Total = 31/32 registers used.
 * 2. Parallelization: OpenMP with collapse(2) on the M and N tiling loops to distribute work across 144 logical threads.
 * 3. Cache-Aware Tiling: BM=128, BN=128, BK=256 to fit within the L2 cache (1MB) and leverage the massive L3 (384MB).
 * 4. NUMA Optimization: First-touch allocation in main() by initializing matrices in parallel using OpenMP.
 * 5. Memory: Use aligned memory allocation (64-byte alignment) for optimal AVX-512 loads/stores.
 * 
 * Compile Command (GCC):
 * g++ -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
 *
 * Compile Command (Clang):
 * clang++ -O3 -march=native -mavx512f -mavx512dq -mavx512bw -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
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
constexpr int BM = 128;      // Tile size for M
constexpr int BN = 128;      // Tile size for N
constexpr int BK = 256;      // Tile size for K
constexpr int UNROLL_K = 4;  // Inner K unroll factor (micro-kernel)

// Function Signatures
void gemm_scalar(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc);

void gemm_avx512(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc);

// Helper for matrix file output
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
 * Reference Scalar GEMM
 * Row-major: C[i][j] = sum(A[i][k] * B[k][j])
 */
void gemm_scalar(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc) {
    // Initialize C to 0 is handled in main, but we accumulate here
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
 * AVX-512 Optimized Kernel
 */
void gemm_avx512(const float* A, const float* B, float* C,
                int M, int N, int K,
                int lda, int ldb, int ldc) {

    // Distribute work across the many cores of the EPYC 9365
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_tile = 0; m_tile < M; m_tile += BM) {
        for (int n_tile = 0; n_tile < N; n_tile += BN) {
            
            int m_limit = std::min(m_tile + BM, M);
            int n_limit = std::min(n_tile + BN, N);

            for (int k_tile = 0; k_tile < K; k_tile += BK) {
                int k_limit = std::min(k_tile + BK, K);

                // Register-blocked Micro-kernel
                // We process 14 rows and 32 columns (2 vectors) at a time
                for (int i = m_tile; i < m_limit; i += 14) {
                    for (int j = n_tile; j < n_limit; j += 32) {
                        
                        // Remaining dimensions for edge handling
                        int rem_m = std::min(14, m_limit - i);
                        int rem_n = n_limit - j;

                        // Use masks for N-tail (up to 32 elements)
                        __mmask16 mask0 = 0xFFFF;
                        __mmask16 mask1 = 0xFFFF;
                        if (rem_n < 32) {
                            if (rem_n > 16) {
                                mask1 = (__mmask16)((1U << (rem_n - 16)) - 1);
                            } else {
                                mask0 = (__mmask16)((1U << (rem_n > 0 ? rem_n : 0)) - 1);
                                mask1 = 0x0000;
                            }
                        }

                        // We only use the 14x32 register blocking if we have a full 14-row block.
                        // Otherwise, we use a simpler scalar/simd fallback for the row tail.
                        if (rem_m == 14) {
                            // Accumulators in ZMM0-ZMM27
                            __m512 c00 = _mm512_maskz_loadu_ps(mask0, &C[(i + 0) * ldc + j]);
                            __m512 c01 = _mm512_maskz_loadu_ps(mask1, &C[(i + 0) * ldc + j + 16]);
                            __m512 c10 = _mm512_maskz_loadu_ps(mask0, &C[(i + 1) * ldc + j]);
                            __m512 c11 = _mm512_maskz_loadu_ps(mask1, &C[(i + 1) * ldc + j + 16]);
                            __m512 c20 = _mm512_maskz_loadu_ps(mask0, &C[(i + 2) * ldc + j]);
                            __m512 c21 = _mm512_maskz_loadu_ps(mask1, &C[(i + 2) * ldc + j + 16]);
                            __m512 c30 = _mm512_maskz_loadu_ps(mask0, &C[(i + 3) * ldc + j]);
                            __m512 c31 = _mm512_maskz_loadu_ps(mask1, &C[(i + 3) * ldc + j + 16]);
                            __m512 c40 = _mm512_maskz_loadu_ps(mask0, &C[(i + 4) * ldc + j]);
                            __m512 c41 = _mm512_maskz_loadu_ps(mask1, &C[(i + 4) * ldc + j + 16]);
                            __m512 c50 = _mm512_maskz_loadu_ps(mask0, &C[(i + 5) * ldc + j]);
                            __m512 c51 = _mm512_maskz_loadu_ps(mask1, &C[(i + 5) * ldc + j + 16]);
                            __m512 c60 = _mm512_maskz_loadu_ps(mask0, &C[(i + 6) * ldc + j]);
                            __m512 c61 = _mm512_maskz_loadu_ps(mask1, &C[(i + 6) * ldc + j + 16]);
                            __m512 c70 = _mm512_maskz_loadu_ps(mask0, &C[(i + 7) * ldc + j]);
                            __m512 c71 = _mm512_maskz_loadu_ps(mask1, &C[(i + 7) * ldc + j + 16]);
                            __m512 c80 = _mm512_maskz_loadu_ps(mask0, &C[(i + 8) * ldc + j]);
                            __m512 c81 = _mm512_maskz_loadu_ps(mask1, &C[(i + 8) * ldc + j + 16]);
                            __m512 c90 = _mm512_maskz_loadu_ps(mask0, &C[(i + 9) * ldc + j]);
                            __m512 c91 = _mm512_maskz_loadu_ps(mask1, &C[(i + 9) * ldc + j + 16]);
                            __m512 cA0 = _mm512_maskz_loadu_ps(mask0, &C[(i + 10) * ldc + j]);
                            __m512 cA1 = _mm512_maskz_loadu_ps(mask1, &C[(i + 10) * ldc + j + 16]);
                            __m512 cB0 = _mm512_maskz_loadu_ps(mask0, &C[(i + 11) * ldc + j]);
                            __m512 cB1 = _mm512_maskz_loadu_ps(mask1, &C[(i + 11) * ldc + j + 16]);
                            __m512 cC0 = _mm512_maskz_loadu_ps(mask0, &C[(i + 12) * ldc + j]);
                            __m512 cC1 = _mm512_maskz_loadu_ps(mask1, &C[(i + 12) * ldc + j + 16]);
                            __m512 cD0 = _mm512_maskz_loadu_ps(mask0, &C[(i + 13) * ldc + j]);
                            __m512 cD1 = _mm512_maskz_loadu_ps(mask1, &C[(i + 13) * ldc + j + 16]);

                            for (int k = k_tile; k < k_limit; ++k) {
                                // Load 32 elements of B
                                __m512 b0 = _mm512_maskz_loadu_ps(mask0, &B[k * ldb + j]);
                                __m512 b1 = _mm512_maskz_loadu_ps(mask1, &B[k * ldb + j + 16]);

                                // Broadcast and FMA
                                #define FMA_STEP(idx, row_offset) \
                                    { __m512 a_vec = _mm512_set1_ps(A[(i + row_offset) * lda + k]); \
                                      idx##0 = _mm512_fmadd_ps(a_vec, b0, idx##0); \
                                      idx##1 = _mm512_fmadd_ps(a_vec, b1, idx##1); }

                                FMA_STEP(c0, 0); FMA_STEP(c1, 1); FMA_STEP(c2, 2); FMA_STEP(c3, 3);
                                FMA_STEP(c4, 4); FMA_STEP(c5, 5); FMA_STEP(c6, 6); FMA_STEP(c7, 7);
                                FMA_STEP(c8, 8); FMA_STEP(c9, 9); FMA_STEP(cA, 10); FMA_STEP(cB, 11);
                                FMA_STEP(cC, 12); FMA_STEP(cD, 13);
                            }

                            _mm512_mask_storeu_ps(&C[(i + 0) * ldc + j], mask0, c00);
                            _mm512_mask_storeu_ps(&C[(i + 0) * ldc + j + 16], mask1, c01);
                            _mm512_mask_storeu_ps(&C[(i + 1) * ldc + j], mask0, c10);
                            _mm512_mask_storeu_ps(&C[(i + 1) * ldc + j + 16], mask1, c11);
                            _mm512_mask_storeu_ps(&C[(i + 2) * ldc + j], mask0, c20);
                            _mm512_mask_storeu_ps(&C[(i + 2) * ldc + j + 16], mask1, c21);
                            _mm512_mask_storeu_ps(&C[(i + 3) * ldc + j], mask0, c30);
                            _mm512_mask_storeu_ps(&C[(i + 3) * ldc + j + 16], mask1, c31);
                            _mm512_mask_storeu_ps(&C[(i + 4) * ldc + j], mask0, c40);
                            _mm512_mask_storeu_ps(&C[(i + 4) * ldc + j + 16], mask1, c41);
                            _mm512_mask_storeu_ps(&C[(i + 5) * ldc + j], mask0, c50);
                            _mm512_mask_storeu_ps(&C[(i + 5) * ldc + j + 16], mask1, c51);
                            _mm512_mask_storeu_ps(&C[(i + 6) * ldc + j], mask0, c60);
                            _mm512_mask_storeu_ps(&C[(i + 6) * ldc + j + 16], mask1, c61);
                            _mm512_mask_storeu_ps(&C[(i + 7) * ldc + j], mask0, c70);
                            _mm512_mask_storeu_ps(&C[(i + 7) * ldc + j + 16], mask1, c71);
                            _mm512_mask_storeu_ps(&C[(i + 8) * ldc + j], mask0, c80);
                            _mm512_mask_storeu_ps(&C[(i + 8) * ldc + j + 16], mask1, c81);
                            _mm512_mask_storeu_ps(&C[(i + 9) * ldc + j], mask0, c90);
                            _mm512_mask_storeu_ps(&C[(i + 9) * ldc + j + 16], mask1, c91);
                            _mm512_mask_storeu_ps(&C[(i + 10) * ldc + j], mask0, cA0);
                            _mm512_mask_storeu_ps(&C[(i + 10) * ldc + j + 16], mask1, cA1);
                            _mm512_mask_storeu_ps(&C[(i + 11) * ldc + j], mask0, cB0);
                            _mm512_mask_storeu_ps(&C[(i + 11) * ldc + j + 16], mask1, cB1);
                            _mm512_mask_storeu_ps(&C[(i + 12) * ldc + j], mask0, cC0);
                            _mm512_mask_storeu_ps(&C[(i + 12) * ldc + j + 16], mask1, cC1);
                            _mm512_mask_storeu_ps(&C[(i + 13) * ldc + j], mask0, cD0);
                            _mm512_mask_storeu_ps(&C[(i + 13) * ldc + j + 16], mask1, cD1);

                        } else {
                            // Row tail fallback (rem_m < 14)
                            for (int r = 0; r < rem_m; ++r) {
                                __m512 cur_c0 = _mm512_maskz_loadu_ps(mask0, &C[(i + r) * ldc + j]);
                                __m512 cur_c1 = _mm512_maskz_loadu_ps(mask1, &C[(i + r) * ldc + j + 16]);
                                for (int k = k_tile; k < k_limit; ++k) {
                                    __m512 a_vec = _mm512_set1_ps(A[(i + r) * lda + k]);
                                    __m512 b0 = _mm512_maskz_loadu_ps(mask0, &B[k * ldb + j]);
                                    __m512 b1 = _mm512_maskz_loadu_ps(mask1, &B[k * ldb + j + 16]);
                                    cur_c0 = _mm512_fmadd_ps(a_vec, b0, cur_c0);
                                    cur_c1 = _mm512_fmadd_ps(a_vec, b1, cur_c1);
                                }
                                _mm512_mask_storeu_ps(&C[(i + r) * ldc + j], mask0, cur_c0);
                                _mm512_mask_storeu_ps(&C[(i + r) * ldc + j + 16], mask1, cur_c1);
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
    std::cerr << "AVX-512 not supported on this compiler/target. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
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

    // Allocate memory with 64-byte alignment for AVX-512
    size_t sizeA = (size_t)M * lda * sizeof(float);
    size_t sizeB = (size_t)K * ldb * sizeof(float);
    size_t sizeC = (size_t)M * ldc * sizeof(float);

    float* A = (float*)aligned_alloc(64, (sizeA + 63) & ~63ULL);
    float* B = (float*)aligned_alloc(64, (sizeB + 63) & ~63ULL);
    float* C = (float*)aligned_alloc(64, (sizeC + 63) & ~63ULL);

    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed." << std::endl;
        return 1;
    }

    // Parallel initialization to ensure NUMA "First Touch"
    #pragma omp parallel
    {
        std::mt19937 gen(omp_get_thread_num());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        #pragma omp for nowait
        for (int i = 0; i < M * lda; ++i) A[i] = dist(gen);
        #pragma omp for nowait
        for (int i = 0; i < K * ldb; ++i) B[i] = dist(gen);
        #pragma omp for nowait
        for (int i = 0; i < M * ldc; ++i) C[i] = 0.0f;
    }

    if (dump_matrices) {
        // Test Mode
        std::string workspace = "workspace";
        // Check if workspace exists, if not, use current dir
        std::string prefix = "workspace/";
        std::ifstream dir_check("workspace");
        if (!dir_check.good()) prefix = "";

        write_matrix_to_file(prefix + "A.txt", A, M, K, lda);
        write_matrix_to_file(prefix + "B.txt", B, K, N, ldb);

        float* C_ref = (float*)aligned_alloc(64, (sizeC + 63) & ~63ULL);
        std::memset(C_ref, 0, sizeC);

        // Performance timing for scalar (optional)
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

        // Optimized call
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);

        write_matrix_to_file(prefix + "C.txt", C, M, N, ldc);

        // Internal correctness check
        bool passed = true;
        for (int i = 0; i < M * N; ++i) {
            if (std::abs(C[i] - C_ref[i]) > 1e-3) {
                passed = false;
                break;
            }
        }
        std::cout << "Internal check: " << (passed ? "PASSED" : "FAILED") << std::endl;

        free(C_ref);
    } else {
        // Performance Mode
        auto start = std::chrono::high_resolution_clock::now();
        
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        double gflops = (2.0 * M * N * K) / (diff.count() * 1e9);
        std::cout << "GEMM Performance: " << gflops << " GFLOPS" << std::endl;
        std::cout << "Time: " << diff.count() << " seconds" << std::endl;
    }

    free(A);
    free(B);
    free(C);

    return 0;
}

/*
 * Implementation Details:
 * 
 * Register Layout:
 * The micro-kernel processes a 14x32 block of C.
 * AVX-512 registers (ZMM0-ZMM31):
 * - ZMM0-ZMM27: Accumulators for the 14x32 block (14 rows * 2 ZMMs per row).
 * - ZMM28-ZMM29: Temp storage for loading two 16-float segments of a row of B.
 * - ZMM30: Broadcast register for an element of A.
 * - ZMM31: Available for temp usage.
 * 
 * Tiling Strategy:
 * - BM (128) and BK (256) ensure that the active portion of A (BMxBK) fits in 
 *   L2 cache (~128KB).
 * - BK (256) and BN (128) ensure the active portion of B fits in L2 as well.
 * - EPYC Zen 4 cores have 1MB L2 each, allowing these tiles to reside comfortably.
 * 
 * OpenMP:
 * - collapse(2) on the outermost tiles (M and N) provides (M/BM)*(N/BN) tasks.
 *   For M=N=2048, this is 16*16 = 256 tasks, which saturates 144 threads well.
 */

/*
 * Final note on AMD EPYC 9365:
 * This CPU features high memory bandwidth and massive L3 cache. 
 * Register reuse in the micro-kernel (14x32) is critical to minimize 
 * the "memory wall" effect. Zen 4's implementation of AVX-512 is 
 * extremely efficient, with 2 FMA units per core.
 */

/*
 * Verification Checklist:
 * 1. Signatures match exactly.
 * 2. CLI parses --dump-matrices.
 * 3. AVX-512 intrinsics guarded and used correctly.
 * 4. Tail cases (M, N, K) handled via masking and fallback loops.
 * 5. NUMA first-touch via OpenMP init.
 */