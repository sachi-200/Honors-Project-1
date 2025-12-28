// Compilation Instructions:
// GCC: g++ -O3 -march=native -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
// Clang: clang++ -O3 -march=native -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -fopenmp gemm.cpp -o gemm
// Note: Optimized for AMD EPYC 9365 (Zen 4) featuring 72 Cores (144 Threads), AVX-512, and large L2/L3 caches.

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
#include <cmath>

/**
 * GEMM Optimization for AMD EPYC 9365 (Zen 4)
 * 
 * Hardware Context:
 * - 72 Cores / 144 Logical Threads.
 * - 512-bit ZMM registers (AVX-512).
 * - 1 MB L2 cache per core.
 * 
 * Strategy:
 * 1. Cache Tiling (BM=192, BN=192, BK=128):
 *    - These tile sizes ensure that the working set (A-packed, B-packed, and C-tile) 
 *      fits within the 1 MB L2 cache (~344 KB total), maximizing L2 hits.
 * 2. Double Packing:
 *    - Matrix A is packed into a contiguous thread-local buffer [BM x BK] to remove strided access.
 *    - Matrix B is packed into micro-panels of [BK x 48] to ensure linear access in the micro-kernel.
 * 3. Micro-Kernel (6x48):
 *    - Uses 18 ZMM registers for accumulation, 6 for A-broadcasting, and 3 for B-loading.
 *    - Total: 27/32 registers used, enabling high Instruction Level Parallelism (ILP).
 * 4. Parallelization:
 *    - OpenMP parallel for collapse(2) over M and N tiles to utilize all 144 logical threads.
 *    - "First Touch" NUMA allocation via parallel initialization.
 * 5. Edge Handling:
 *    - Full support for tail cases in M, N, and K dimensions using masking and length checks.
 */

// Autotuning Parameters
constexpr int BM = 192;   // M-tile size
constexpr int BN = 192;   // N-tile size
constexpr int BK = 128;   // K-tile size
constexpr int MR = 6;     // Micro-kernel rows
constexpr int NR = 48;    // Micro-kernel columns (3 * 16 floats)

/**
 * Aligned allocation to ensure 64-byte alignment for AVX-512.
 */
template <typename T>
T* aligned_alloc_type(size_t size, size_t alignment = 64) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size * sizeof(T)) != 0) return nullptr;
    return static_cast<T*>(ptr);
}

/**
 * AVX-512 Mask generation for tail handling.
 */
inline uint16_t get_simd_mask(int rem) {
    if (rem <= 0) return 0;
    if (rem >= 16) return 0xFFFF;
    return (uint16_t)((1U << (uint32_t)rem) - 1);
}

/**
 * Scalar reference GEMM (C = A * B + C).
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
 * Packs a tile of A into contiguous memory: [BM x BK].
 */
inline void pack_A_tile(const float* __restrict__ A, float* __restrict__ A_packed,
                         int M_start, int M_end, int K_start, int K_end, int lda) {
    const int M_tile = M_end - M_start;
    const int K_tile = K_end - K_start;
    for (int i = 0; i < M_tile; ++i) {
        std::memcpy(A_packed + i * K_tile, A + (M_start + i) * lda + K_start, K_tile * sizeof(float));
    }
}

/**
 * Vectorized packing of B into micro-panels: [N/48][BK][48].
 * Uses AVX-512 masked loads/stores to handle N-tails and zero-fill.
 */
inline void pack_B_tile(const float* __restrict__ B, float* __restrict__ B_packed, 
                        int K_start, int K_end, int N_start, int N_end, int ldb) {
    const int K_tile = K_end - K_start;
    const int N_tile = N_end - N_start;
    
    for (int j = 0; j < N_tile; j += NR) {
        int n_len = std::min(NR, N_tile - j);
        float* panel_ptr = B_packed + (j / NR) * K_tile * NR;

        __mmask16 masks[3];
        masks[0] = get_simd_mask(n_len);
        masks[1] = get_simd_mask(n_len - 16);
        masks[2] = get_simd_mask(n_len - 32);

        for (int k = 0; k < K_tile; ++k) {
            const float* src = B + (K_start + k) * ldb + N_start + j;
            float* dst = panel_ptr + k * NR;
            // Load from original B and store in micro-panel (aligned)
            _mm512_store_ps(dst + 0,  _mm512_maskz_loadu_ps(masks[0], src + 0));
            _mm512_store_ps(dst + 16, _mm512_maskz_loadu_ps(masks[1], src + 16));
            _mm512_store_ps(dst + 32, _mm512_maskz_loadu_ps(masks[2], src + 32));
        }
    }
}

/**
 * Consolidated AVX-512 Micro-Kernel.
 * Accumulates results for a block up to 6x48 into C.
 */
inline void micro_kernel(const float* __restrict__ A_ptr, int lda_p, 
                         const float* __restrict__ B_panel, 
                         float* __restrict__ C_ptr, int ldc, 
                         int K, int m_len, int n_rem) {
    __m512 c[6][3];
    // Always initialize 18 registers to zero. 
    // Small overhead for tail cases vs significantly better code generation.
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 3; ++j) c[i][j] = _mm512_setzero_ps();
    }

    // Inner K-loop: Contiguous access for A and B
    for (int k = 0; k < K; ++k) {
        // Contiguous aligned loads from packed B micro-panel
        __m512 b0 = _mm512_load_ps(B_panel + k * 48 + 0);
        __m512 b1 = _mm512_load_ps(B_panel + k * 48 + 16);
        __m512 b2 = _mm512_load_ps(B_panel + k * 48 + 32);

        for (int i = 0; i < m_len; ++i) {
            __m512 a = _mm512_set1_ps(A_ptr[i * lda_p + k]);
            c[i][0] = _mm512_fmadd_ps(a, b0, c[i][0]);
            c[i][1] = _mm512_fmadd_ps(a, b1, c[i][1]);
            c[i][2] = _mm512_fmadd_ps(a, b2, c[i][2]);
        }
    }

    // Write-back with AVX-512 masks for N-tails
    __mmask16 masks[3];
    masks[0] = get_simd_mask(n_rem);
    masks[1] = get_simd_mask(n_rem - 16);
    masks[2] = get_simd_mask(n_rem - 32);

    for (int i = 0; i < m_len; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (masks[j] != 0) {
                float* target = C_ptr + i * ldc + j * 16;
                __m512 old_c = _mm512_maskz_loadu_ps(masks[j], target);
                __m512 new_c = _mm512_add_ps(old_c, c[i][j]);
                _mm512_mask_storeu_ps(target, masks[j], new_c);
            }
        }
    }
}

/**
 * AVX-512 Optimized GEMM implementation.
 */
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__)
    #pragma omp parallel
    {
        // Thread-local scratchpads for packing
        float* B_packed = aligned_alloc_type<float>(BK * BN);
        float* A_packed = aligned_alloc_type<float>(BM * BK);

        #pragma omp for collapse(2) schedule(static)
        for (int m = 0; m < M; m += BM) {
            for (int n = 0; n < N; n += BN) {
                const int m_end = std::min(m + BM, M);
                const int n_end = std::min(n + BN, N);

                for (int k = 0; k < K; k += BK) {
                    const int k_end = std::min(k + BK, K);
                    const int cur_bk = k_end - k;

                    // Pack A and B slices into contiguous local buffers
                    pack_A_tile(A, A_packed, m, m_end, k, k_end, lda);
                    pack_B_tile(B, B_packed, k, k_end, n, n_end, ldb);

                    // Divide current tile into micro-panels
                    for (int n_m = 0; n_m < (n_end - n); n_m += NR) {
                        const int n_rem = std::min(NR, (n_end - n) - n_m);
                        const float* b_panel = B_packed + (n_m / NR) * cur_bk * NR;

                        for (int m_m = 0; m_m < (m_end - m); m_m += MR) {
                            const int m_len = std::min(MR, (m_end - m) - m_m);
                            float* c_target = C + (m + m_m) * ldc + (n + n_m);
                            const float* a_source = A_packed + m_m * cur_bk;

                            micro_kernel(a_source, cur_bk, b_panel, c_target, ldc, cur_bk, m_len, n_rem);
                        }
                    }
                }
            }
        }
        free(A_packed);
        free(B_packed);
    }
#else
    // Fallback if AVX-512 is not detected at runtime (safety)
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

/**
 * Utility: Saves matrix to text file.
 */
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

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]\n";
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump = (argc > 4 && std::string(argv[4]) == "--dump-matrices");

    // Standard leading dimensions for row-major storage
    int lda = K, ldb = N, ldc = N;

    float* A = aligned_alloc_type<float>(M * K);
    float* B = aligned_alloc_type<float>(K * N);
    float* C = aligned_alloc_type<float>(M * N);

    /**
     * NUMA First-Touch Initialization:
     * Parallelizing the initialization ensures memory pages are mapped 
     * near the cores that will process them.
     */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A[i * K + j] = static_cast<float>((i + j) % 11) / 11.0f;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B[i * N + j] = static_cast<float>((i * j) % 13) / 13.0f;
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0.0f;
        }
    }

    if (dump) {
        // --- Test Mode ---
        float* C_ref = aligned_alloc_type<float>(M * N);
        std::memset(C_ref, 0, M * N * sizeof(float));

        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);

        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

        // Verification logic
        bool passed = true;
        for (int i = 0; i < M * N; ++i) {
            float diff = std::abs(C[i] - C_ref[i]);
            float ref_v = std::abs(C_ref[i]) + 1e-9f;
            if (diff / ref_v > 1e-2f && diff > 1e-2f) {
                passed = false;
                break;
            }
        }
        std::cout << "Internal check: " << (passed ? "PASSED" : "FAILED") << std::endl;
        free(C_ref);
    } else {
        // --- Performance Mode ---
        auto start = std::chrono::high_resolution_clock::now();
        
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        double gflops = (2.0 * M * N * K) / (duration.count() * 1e9);
        std::cout << "Performance: " << std::fixed << std::setprecision(2) << gflops << " GFLOPS (" << duration.count() << " s)" << std::endl;
    }

    free(A);
    free(B);
    free(C);

    return 0;
}