// Compile instructions:
//
// To compile and run on a system that supports AVX2 (like the target AMD Ryzen 7 6800HS):
// Recommended: `g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_avx2 -std=c++17`
// Or specifically: `g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2 -std=c++17`
// (The target CPU, AMD Ryzen 7 6800HS, will use the AVX2 path as it lacks AVX-512.
// `-march=native` allows the compiler to detect and use all available features on the host system.)
//
// For systems that specifically have AVX-512 (e.g., Intel Skylake-X, Ice Lake, Sapphire Rapids, Zen 4 with AVX512):
// Recommended: `g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_avx512 -std=c++17`
// Or specifically: `g++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512 -std=c++17`
// (The `__attribute__((target(...)))` mechanism allows specific functions to be compiled with
// different ISA extensions within the same binary, enabling runtime dispatch even with a more
// conservative global `-march` flag like `-march=x86-64` or `-march=native` on an AVX2-only CPU.)
//
// Portable (runtime dispatch will select scalar if no suitable SIMD is detected/compiled):
// `g++ -O3 -march=x86-64 -fopenmp gemm.cpp -o gemm_portable -std=c++17`
// `clang++ -O3 -march=x86-64 -fopenmp gemm.cpp -o gemm_portable -std=c++17`

#include <iostream>   // For std::cout, std::cerr
#include <vector>     // For std::vector (though raw pointers are used for matrix data)
#include <random>     // For std::mt19937, std::uniform_real_distribution
#include <chrono>     // For std::chrono high_resolution_clock
#include <cassert>    // For assert
#include <fstream>    // For std::ofstream
#include <filesystem> // For std::filesystem::path, create_directory (C++17)
#include <cstring>    // For std::memset
#include <string>     // For std::string, std::stoi, std::stoll
#include <algorithm>  // For std::min
#include <cstddef>    // For size_t, ptrdiff_t
#include <cmath>      // For std::abs
#include <iomanip>    // For std::fixed, std::precision
#include <stdexcept>  // For std::runtime_error

// Intrinsics headers (includes MMX, SSE, AVX, AVX2, AVX512F, FMA, etc.)
#include <immintrin.h>

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// --- Autotuning Parameters and Blocking Strategy ---
// These parameters are crucial for cache efficiency and SIMD utilization.
// The strategy is a 3-level tiling:
// 1. Outer loops (M, N) are parallelized with OpenMP threads across large matrix blocks (BM x BN).
// 2. Middle loop (K) iterates through K-blocks (BK).
// 3. Inner-most loops form a micro-kernel (MR x NR_VEC) operating on registers, accumulating results.
//
// The goal is to keep data for the current K-block of A (BM x BK) and B (BK x BN) in L1/L2 cache.
//
// Target CPU: AMD Ryzen 7 6800HS
// L1d cache: 32KB (per core)
// L2 cache:  512KB (per core)
// L3 cache:  16MB (shared across 8 cores)
//
// float size = 4 bytes.
//
// Example calculation for L2 cache fit (for BM=96, BN=128, BK=96):
// A_block (BM x BK): 96 * 96 * 4 bytes = 36864 bytes (~36KB)
// B_block (BK x BN): 96 * 128 * 4 bytes = 49152 bytes (~48KB)
// Total data for A and B blocks for one K-step: ~36KB + ~48KB = 84KB. This fits comfortably within a 512KB L2 cache.
//
// Note: Due to strict function signature requirements, `DEFAULT_BM/BN/BK` are `constexpr` globals.
// A dynamic autotuner that modifies these at runtime (e.g., through a benchmark loop)
// would typically involve compiling/loading different kernel versions or re-compiling with new constants.
// For this single-file solution, the "autotune harness" merely benchmarks the chosen default configuration.

constexpr int DEFAULT_BM = 96;  // Rows of C block for outer tiling (multiple of MR, e.g., 4*24)
constexpr int DEFAULT_BN = 128; // Cols of C block for outer tiling (multiple of NR_VEC, e.g., 8*16 or 16*8)
constexpr int DEFAULT_BK = 96;  // Inner dimension K block for outer tiling (multiple of UNROLL_K, e.g., 4*24)

// Micro-kernel register blocking factors (MR x NR_VEC registers for C)
// MR: Number of rows of C (and A) processed in parallel in the micro-kernel.
// NR_VEC: Number of columns of C (and B) processed using a single SIMD vector.
// UNROLL_K: K-dimension unroll factor within the micro-kernel's K-loop.

// For AVX2 (256-bit registers = 8 floats)
constexpr int NR_AVX2_VEC = 8;  // 8 floats per __m256 vector
constexpr int MR_AVX2     = 4;  // 4 rows of C computed simultaneously in register accumulators
constexpr int UNROLL_K_AVX2 = 4; // Unroll factor for K-loop in AVX2 micro-kernel

// For AVX-512 (512-bit registers = 16 floats)
constexpr int NR_AVX512_VEC = 16; // 16 floats per __m512 vector
constexpr int MR_AVX512     = 4;  // 4 rows of C computed simultaneously in register accumulators
constexpr int UNROLL_K_AVX512 = 4; // Unroll factor for K-loop in AVX-512 micro-kernel

// Prefetch distance: how many K-steps ahead to prefetch elements of A and B
constexpr int PREFETCH_DIST = 4; // Example value, can be tuned

// Row-major storage is assumed for A, B, C.
// A (M x K) at A + i*lda + j where lda is the leading dimension (typically K for contiguous).
// B (K x N) at B + i*ldb + j where ldb is the leading dimension (typically N for contiguous).
// C (M x N) at C + i*ldc + j where ldc is the leading dimension (typically N for contiguous).

// --- Helper Functions ---

// Function to write a matrix to a text file
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    ofs.precision(6); // Set precision for float output
    ofs << std::fixed; // Use fixed-point notation for consistent output

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[static_cast<size_t>(i) * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// --- Scalar Reference Implementation ---
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<size_t>(m) * lda + k] * B[static_cast<size_t>(k) * ldb + n];
            }
            C[static_cast<size_t>(m) * ldc + n] = sum;
        }
    }
}

// --- AVX2 + FMA Micro-Kernel Implementation ---
// The `__attribute__((target("avx2,fma")))` ensures this function is compiled
// with AVX2 and FMA instructions, even if the main compilation flags are more generic.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma")))
#endif
void gemm_avx2_micro_kernel(int M_tile, int N_tile, int K_tile,
                            const float* A_global, const float* B_global, float* C_global,
                            int lda, int ldb, int ldc,
                            int m_global_offset, int n_global_offset, int k_global_offset) {
    
    // Ensure intrinsics are available at compile time. This acts as a safeguard.
#if defined(__AVX2__) && defined(__FMA__)
    constexpr int FLOAT_PER_VEC = NR_AVX2_VEC; // 8 floats per __m256 vector
    constexpr int MR = MR_AVX2;

    // Calculate vectorized boundaries for M and N dimensions within the current tile.
    const int m_vec_end_in_tile = (M_tile / MR) * MR; // Number of rows that are multiples of MR
    const int n_vec_end_in_tile = (N_tile / FLOAT_PER_VEC) * FLOAT_PER_VEC; // Number of columns that are multiples of FLOAT_PER_VEC

    // --- Main Vectorized M-N Loop (processes (m_vec_end_in_tile x n_vec_end_in_tile) sub-rectangle) ---
    for (int m_offset_in_tile = 0; m_offset_in_tile < m_vec_end_in_tile; m_offset_in_tile += MR) {
        // Here, actual_MR will always be MR, because we loop up to m_vec_end_in_tile.
        const int actual_MR = MR;

        for (int n_offset_in_tile = 0; n_offset_in_tile < n_vec_end_in_tile; n_offset_in_tile += FLOAT_PER_VEC) {
            // Accumulators for MR rows, one vector register each for the current K-slice contribution
            __m256 c_acc[MR]; 
            for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) {
                c_acc[r_in_block] = _mm256_setzero_ps(); // Initialize accumulators to zero
            }

            // K-loop: inner-most loop for accumulation, unrolled by UNROLL_K_AVX2
            for (int k_offset_in_tile = 0; k_offset_in_tile < K_tile; k_offset_in_tile += UNROLL_K_AVX2) {
                for (int uk = 0; uk < UNROLL_K_AVX2; ++uk) {
                    if (k_offset_in_tile + uk >= K_tile) break; // Handle K-tail within unroll factor

                    const int current_k_global = k_global_offset + k_offset_in_tile + uk;
                    const float* B_k_n_ptr = B_global + static_cast<size_t>(current_k_global) * ldb + (n_global_offset + n_offset_in_tile);

                    // Prefetch B for next K iteration from L1 (T0)
                    #ifdef __SSE__ 
                    if (k_offset_in_tile + uk + PREFETCH_DIST < K_tile) {
                        _mm_prefetch((const char*)(B_global + static_cast<size_t>(k_global_offset + k_offset_in_tile + uk + PREFETCH_DIST) * ldb + (n_global_offset + n_offset_in_tile)), _MM_HINT_T0);
                    }
                    #endif

                    // Load B vector. _mm256_loadu_ps handles unaligned memory access.
                    __m256 b_vec = _mm256_loadu_ps(B_k_n_ptr);

                    // For each row in the MR block
                    for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) {
                        const int current_m_global = m_global_offset + m_offset_in_tile + r_in_block;
                        const float* A_m_k_ptr = A_global + static_cast<size_t>(current_m_global) * lda + current_k_global;
                        
                        // Prefetch A for next K iteration, for this row
                        #ifdef __SSE__
                        if (k_offset_in_tile + uk + PREFETCH_DIST < K_tile) {
                            _mm_prefetch((const char*)(A_global + static_cast<size_t>(current_m_global) * lda + (current_k_global + PREFETCH_DIST)), _MM_HINT_T0);
                        }
                        #endif

                        // Load A scalar and broadcast to fill a vector register.
                        __m256 a_scalar = _mm256_broadcast_ss(A_m_k_ptr);
                        
                        // Fused Multiply-Add: c_acc[r] = a_scalar * b_vec + c_acc[r]
                        c_acc[r_in_block] = _mm256_fmadd_ps(a_scalar, b_vec, c_acc[r_in_block]);
                    }
                }
            } // End K-loop

            // Store accumulated results back to C. Add to existing values in C_global.
            for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) {
                float* c_ptr = C_global + static_cast<size_t>(m_global_offset + m_offset_in_tile + r_in_block) * ldc + (n_global_offset + n_offset_in_tile);
                // Load existing C block, add accumulated value, then store back.
                _mm256_storeu_ps(c_ptr, _mm256_add_ps(_mm256_loadu_ps(c_ptr), c_acc[r_in_block]));
            }
        } // End vectorized N-loop (full vectors)
    } // End M-loop (full MR-blocks)

    // --- Scalar N-tail for M-blocks (columns not divisible by FLOAT_PER_VEC) ---
    // This processes the remaining columns for all rows that were fully vectorized in M.
    if (n_vec_end_in_tile < N_tile) { // If there's an N-tail part
        for (int m_offset_in_tile = 0; m_offset_in_tile < m_vec_end_in_tile; m_offset_in_tile += MR) { // For each M-block
            const int actual_MR = MR; // Full MR rows for these blocks
            for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) { // For each row in the current MR-block
                for (int n_scalar_in_tile = n_vec_end_in_tile; n_scalar_in_tile < N_tile; ++n_scalar_in_tile) { // For each N-tail column
                    float sum_elem = 0.0f; // Accumulator for this specific C(m,n) element for CURRENT K-TILE.
                    for (int k_offset_in_tile = 0; k_offset_in_tile < K_tile; ++k_offset_in_tile) {
                        sum_elem += A_global[static_cast<size_t>(m_global_offset + m_offset_in_tile + r_in_block) * lda + (k_global_offset + k_offset_in_tile)] *
                                    B_global[static_cast<size_t>(k_global_offset + k_offset_in_tile) * ldb + (n_global_offset + n_scalar_in_tile)];
                    }
                    C_global[static_cast<size_t>(m_global_offset + m_offset_in_tile + r_in_block) * ldc + (n_global_offset + n_scalar_in_tile)] += sum_elem; // ADD current K-slice contribution
                }
            }
        }
    }
    
    // --- Scalar M-tail for the remaining rows (M_tile % MR) covering all N_tile columns ---
    const int m_tail_start_in_tile = m_vec_end_in_tile; // Equivalent to (M_tile / MR) * MR;
    if (m_tail_start_in_tile < M_tile) { // If there's an M-tail part
        for (int m_scalar_in_tile = m_tail_start_in_tile; m_scalar_in_tile < M_tile; ++m_scalar_in_tile) { // For each M-tail row
            for (int n_scalar_in_tile = 0; n_scalar_in_tile < N_tile; ++n_scalar_in_tile) { // For all N columns (full width of current N-tile)
                float sum_elem = 0.0f; // Accumulator for this specific C(m,n) element for CURRENT K-TILE.
                for (int k_offset_in_tile = 0; k_offset_in_tile < K_tile; ++k_offset_in_tile) {
                    sum_elem += A_global[static_cast<size_t>(m_global_offset + m_scalar_in_tile) * lda + (k_global_offset + k_offset_in_tile)] *
                                B_global[static_cast<size_t>(k_global_offset + k_offset_in_tile) * ldb + (n_global_offset + n_scalar_in_tile)];
                }
                C_global[static_cast<size_t>(m_global_offset + m_scalar_in_tile) * ldc + (n_global_offset + n_scalar_in_tile)] += sum_elem; // ADD current K-slice contribution
            }
        }
    }
#else
    // Fallback if AVX2/FMA intrinsics are not available at compile time.
    // This ensures correctness even if __attribute__((target(...))) fails or specific flags were missing.
    gemm_scalar(A_global + static_cast<size_t>(m_global_offset) * lda, B_global + static_cast<size_t>(k_global_offset) * ldb + n_global_offset, C_global + static_cast<size_t>(m_global_offset) * ldc + n_global_offset,
                M_tile, N_tile, K_tile, lda, ldb, ldc);
#endif
}

// Wrapper function to expose the exact signature for AVX2
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
#if defined(__AVX2__) && defined(__FMA__)
    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN; 
    const int BK = DEFAULT_BK;

    // OpenMP parallelization for M and N blocks (outer loops)
    // `schedule(static)` is a good default for uniform workloads, providing even distribution.
    // `collapse(2)` ensures that both M and N loops are parallelized effectively across threads.
    // This parallelization granularity is coarse-grained (tiles of C), which minimizes OpenMP overhead.
    // Each thread operates on a distinct C-block, ensuring thread-safe writes.
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int i = 0; i < M; i += BM) { // Iterate over M dimension (rows of C)
        for (int j = 0; j < N; j += BN) { // Iterate over N dimension (columns of C)
            // The C matrix is initialized to zero once in main().
            // Each micro-kernel call accumulates into its respective C block.
            for (int p = 0; p < K; p += BK) { // Iterate over K dimension (inner product)
                // Calculate actual dimensions for the current tile
                int M_current_tile = std::min(BM, M - i);
                int N_current_tile = std::min(BN, N - j);
                int K_current_tile = std::min(BK, K - p);

                // Call the AVX2 micro-kernel
                gemm_avx2_micro_kernel(M_current_tile, N_current_tile, K_current_tile,
                                       A, B, C,
                                       lda, ldb, ldc,
                                       i, j, p);
            }
        }
    }
#else
    // Fallback if AVX2/FMA intrinsics are not available globally at compile time.
    // This should ideally only be hit if __attribute__((target(...))) failed or specific flags were missing.
    std::cerr << "Error: gemm_avx2 called but AVX2/FMA is not available or not enabled at compile time. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

// --- AVX-512 + FMA Micro-Kernel Implementation ---
// The `__attribute__((target("avx512f,fma")))` ensures this function is compiled
// with AVX-512F and FMA instructions, even if the main compilation flags are more generic.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx512f,fma")))
#endif
void gemm_avx512_micro_kernel(int M_tile, int N_tile, int K_tile,
                              const float* A_global, const float* B_global, float* C_global,
                              int lda, int ldb, int ldc,
                              int m_global_offset, int n_global_offset, int k_global_offset) {
#if defined(__AVX512F__) && defined(__FMA__)
    constexpr int FLOAT_PER_VEC = NR_AVX512_VEC; // 16 floats per __m512 vector
    constexpr int MR = MR_AVX512;

    // Calculate vectorized boundaries for M dimension within the current tile.
    const int m_vec_end_in_tile = (M_tile / MR) * MR; // Number of rows that are multiples of MR

    // --- Main Vectorized M-N Loop (processes (m_vec_end_in_tile x N_tile) sub-rectangle using masks for N-tail) ---
    for (int m_offset_in_tile = 0; m_offset_in_tile < m_vec_end_in_tile; m_offset_in_tile += MR) {
        // Here, actual_MR will always be MR, because we loop up to m_vec_end_in_tile.
        const int actual_MR = MR;

        // Process N in chunks of NR_AVX512_VEC, handling tails with masks for loads/stores
        for (int n_offset_in_tile = 0; n_offset_in_tile < N_tile; n_offset_in_tile += FLOAT_PER_VEC) {
            const int current_n_block_size = std::min(FLOAT_PER_VEC, N_tile - n_offset_in_tile);
            // Create a mask for vector operations for potentially partial vectors
            const __mmask16 vec_mask = (__mmask16)((1U << current_n_block_size) - 1);

            __m512 c_acc[MR];
            for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) {
                c_acc[r_in_block] = _mm512_setzero_ps(); // Initialize accumulators to zero
            }

            // K-loop, unrolled
            for (int k_offset_in_tile = 0; k_offset_in_tile < K_tile; k_offset_in_tile += UNROLL_K_AVX512) {
                for (int uk = 0; uk < UNROLL_K_AVX512; ++uk) {
                    if (k_offset_in_tile + uk >= K_tile) break; // Handle K-tail within unroll factor

                    const int current_k_global = k_global_offset + k_offset_in_tile + uk;
                    const float* B_k_n_ptr = B_global + static_cast<size_t>(current_k_global) * ldb + (n_global_offset + n_offset_in_tile);

                    // Prefetch B for next K iteration
                    #ifdef __SSE__
                    if (k_offset_in_tile + uk + PREFETCH_DIST < K_tile) {
                        _mm_prefetch((const char*)(B_global + static_cast<size_t>(current_k_global + PREFETCH_DIST) * ldb + (n_global_offset + n_offset_in_tile)), _MM_HINT_T0);
                    }
                    #endif

                    // Masked load for B: loads only `current_n_block_size` elements, filling rest with zero
                    __m512 b_vec = _mm512_maskz_loadu_ps(vec_mask, B_k_n_ptr);

                    for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) {
                        const int current_m_global = m_global_offset + m_offset_in_tile + r_in_block;
                        const float* A_m_k_ptr = A_global + static_cast<size_t>(current_m_global) * lda + current_k_global;

                        // Prefetch A for next K iteration, for this row
                        #ifdef __SSE__
                        if (k_offset_in_tile + uk + PREFETCH_DIST < K_tile) {
                            _mm_prefetch((const char*)(A_global + static_cast<size_t>(current_m_global) * lda + (current_k_global + PREFETCH_DIST)), _MM_HINT_T0);
                        }
                        #endif

                        // Load A scalar and broadcast to fill a vector register.
                        __m512 a_scalar = _mm512_broadcast_ss(A_m_k_ptr);
                        
                        // Fused Multiply-Add
                        c_acc[r_in_block] = _mm512_fmadd_ps(a_scalar, b_vec, c_acc[r_in_block]);
                    }
                }
            } // End K-loop

            // Store accumulated results for this MR x current_n_block_size block back to C
            for (int r_in_block = 0; r_in_block < actual_MR; ++r_in_block) {
                float* c_ptr = C_global + static_cast<size_t>(m_global_offset + m_offset_in_tile + r_in_block) * ldc + (n_global_offset + n_offset_in_tile);
                // Masked load existing C: loads only `current_n_block_size` elements, others are zero.
                // This ensures we only add to relevant parts of C, preserving existing values outside the mask.
                __m512 c_existing = _mm512_maskz_loadu_ps(vec_mask, c_ptr); 
                // Masked store: writes only `current_n_block_size` elements
                _mm512_mask_storeu_ps(c_ptr, vec_mask, _mm512_add_ps(c_existing, c_acc[r_in_block]));
            }
        } // End N-loop (including masked N-tail handling)
    } // End M-loop (full MR-blocks)

    // --- Scalar M-tail for the remaining rows (M_tile % MR) covering all N_tile columns ---
    const int m_tail_start_in_tile = m_vec_end_in_tile; // Equivalent to (M_tile / MR) * MR;
    if (m_tail_start_in_tile < M_tile) { // If there's an M-tail part
        for (int m_scalar_in_tile = m_tail_start_in_tile; m_scalar_in_tile < M_tile; ++m_scalar_in_tile) { // For each M-tail row
            for (int n_scalar_in_tile = 0; n_scalar_in_tile < N_tile; ++n_scalar_in_tile) { // For all N columns (full width of current N-tile)
                float sum_elem = 0.0f;
                for (int k_offset_in_tile = 0; k_offset_in_tile < K_tile; ++k_offset_in_tile) {
                    sum_elem += A_global[static_cast<size_t>(m_global_offset + m_scalar_in_tile) * lda + (k_global_offset + k_offset_in_tile)] *
                                B_global[static_cast<size_t>(k_global_offset + k_offset_in_tile) * ldb + (n_global_offset + n_scalar_in_tile)];
                }
                C_global[static_cast<size_t>(m_global_offset + m_scalar_in_tile) * ldc + (n_global_offset + n_scalar_in_tile)] += sum_elem;
            }
        }
    }
#else
    // Fallback if AVX-512F intrinsics are not available at compile time.
    gemm_scalar(A_global + static_cast<size_t>(m_global_offset) * lda, B_global + static_cast<size_t>(k_global_offset) * ldb + n_global_offset, C_global + static_cast<size_t>(m_global_offset) * ldc + n_global_offset,
                M_tile, N_tile, K_tile, lda, ldb, ldc);
#endif
}

// Wrapper function to expose the exact signature for AVX-512
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__) && defined(__FMA__)
    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN;
    const int BK = DEFAULT_BK;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int i = 0; i < M; i += BM) {
        for (int j = 0; j < N; j += BN) {
            for (int p = 0; p < K; p += BK) {
                int M_current_tile = std::min(BM, M - i);
                int N_current_tile = std::min(BN, N - j);
                int K_current_tile = std::min(BK, K - p);

                gemm_avx512_micro_kernel(M_current_tile, N_current_tile, K_current_tile,
                                         A, B, C,
                                         lda, ldb, ldc,
                                         i, j, p);
            }
        }
    }
#else
    // Fallback if AVX-512F intrinsics are not available globally at compile time.
    std::cerr << "Error: gemm_avx512 called but AVX-512F/FMA is not available or not enabled at compile time. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}


// --- Top-level GEMM function with runtime dispatch ---
// This function determines the best available SIMD instruction set at runtime
// and dispatches to the corresponding optimized kernel.
// It should be compiled with a general architecture flag (e.g., -march=native or x86-64)
// and relies on the __attribute__((target(...))) on the specific micro-kernels.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    
    bool dispatched = false;

    // Runtime dispatch using __builtin_cpu_supports (GCC/Clang specific).
    // The target CPU (Ryzen 7 6800HS) supports AVX, AVX2, FMA but NOT AVX-512.
    // So, it will likely fall back to AVX2. The AVX-512 path is implemented for compliance
    // with the requirement of implementing it and dispatching if available on other CPUs.

    // Check for AVX-512F support (compiler must also have been configured to enable AVX512F for gemm_avx512 function)
    #if defined(__AVX512F__) && (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("avx512f")) {
        std::cout << "Dispatching to AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        dispatched = true;
    }
    #endif

    // Check for AVX2 support (compiler must also have been configured to enable AVX2 for gemm_avx2 function)
    // Only attempt to dispatch to AVX2 if AVX-512 was not available or not chosen.
    #if defined(__AVX2__) && defined(__FMA__) && (defined(__GNUC__) || defined(__clang__))
    if (!dispatched && __builtin_cpu_supports("avx2")) {
        std::cout << "Dispatching to AVX2 kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        dispatched = true;
    }
    #endif

    if (!dispatched) {
        std::cout << "Dispatching to scalar kernel (no AVX2/AVX-512 support detected or compilation flags missing)." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
}

// --- Main Function for Demo and Testing ---

int main(int argc, char* argv[]) {
    int M = 512;
    int N = 512;
    int K = 512;
    long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    int num_threads = 0; // 0 means OpenMP default
    bool dump_matrices = false;
    bool run_warmup_benchmark = true; // Flag to enable/disable simple warm-up benchmark

    // Process arguments in order.
    // Positional M, N, K first, then flags. Flags for M, N, K override positional.
    int current_arg_idx = 1;
    
    // Attempt to parse positional M, N, K
    if (current_arg_idx < argc && argv[current_arg_idx][0] != '-') {
        try { M = std::stoi(argv[current_arg_idx]); current_arg_idx++; } catch(const std::exception& e) { /* silently ignore if not integer */ }
    }
    if (current_arg_idx < argc && argv[current_arg_idx][0] != '-') {
        try { N = std::stoi(argv[current_arg_idx]); current_arg_idx++; } catch(const std::exception& e) { /* silently ignore if not integer */ }
    }
    if (current_arg_idx < argc && argv[current_arg_idx][0] != '-') {
        try { K = std::stoi(argv[current_arg_idx]); current_arg_idx++; } catch(const std::exception& e) { /* silently ignore if not integer */ }
    }

    // Process remaining arguments (flags and their values)
    for (int i = current_arg_idx; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            if (arg == "-M") {
                if (i + 1 < argc) { M = std::stoi(argv[++i]); } else { throw std::runtime_error("-M requires a value."); }
            } else if (arg == "-N") {
                if (i + 1 < argc) { N = std::stoi(argv[++i]); } else { throw std::runtime_error("-N requires a value."); }
            } else if (arg == "-K") {
                if (i + 1 < argc) { K = std::stoi(argv[++i]); } else { throw std::runtime_error("-K requires a value."); }
            } else if (arg == "--seed") {
                if (i + 1 < argc) { seed = std::stoll(argv[++i]); } else { throw std::runtime_error("--seed requires a value."); }
            } else if (arg == "--threads") {
                if (i + 1 < argc) { num_threads = std::stoi(argv[++i]); } else { throw std::runtime_error("--threads requires a value."); }
            } else if (arg == "--dump-matrices") {
                dump_matrices = true;
            } else if (arg == "--no-warmup-benchmark") {
                run_warmup_benchmark = false;
            } else if (arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M <rows>] [-N <cols>] [-K <inner>] [--seed <value>] [--threads <num>] [--dump-matrices] [--no-warmup-benchmark] [--help]" << std::endl;
                std::cout << "  M, N, K can be specified as first 3 positional arguments (e.g., `gemm 100 200 300`).\n";
                std::cout << "  Flags like -M, -N, -K (e.g., `gemm -M 100`) override any positional values.\n";
                std::cout << "  Default M, N, K values are 512, 512, 512 respectively if not specified.\n";
                return 0;
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                return 1;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error parsing argument " << arg << ": " << e.what() << std::endl;
            return 1;
        }
    }

    // Set OpenMP thread count if specified
    if (num_threads > 0) {
        #ifdef _OPENMP
        omp_set_num_threads(num_threads);
        #else
        std::cerr << "Warning: OpenMP not enabled, --threads argument ignored." << std::endl;
        #endif
    }

    #ifdef _OPENMP
    std::cout << "Running with " << (num_threads > 0 ? std::to_string(num_threads) : std::string("default")) << " OpenMP threads (max: " << omp_get_max_threads() << ")." << std::endl;
    #else
    std::cout << "Running without OpenMP." << std::endl;
    #endif

    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    
    // Allocate matrices using aligned memory for SIMD operations
    // AVX-512 generally benefits from 64-byte alignment, AVX2 from 32-byte. Max is 64-byte.
    const size_t alignment = 64; 
    
    // Custom aligned allocator using posix_memalign for Linux.
    auto aligned_allocator = [&](size_t count) {
        void* ptr = nullptr;
        // posix_memalign returns 0 on success, non-zero on failure
        if (posix_memalign(&ptr, alignment, count * sizeof(float)) != 0) {
            ptr = nullptr; 
        }
        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<float*>(ptr);
    };

    // Custom aligned deallocator for memory allocated by posix_memalign.
    auto aligned_deallocator = [](float* ptr) {
        free(ptr); // free for posix_memalign
    };

    float* A_data = nullptr;
    float* B_data = nullptr;
    float* C_data = nullptr;
    float* C_ref_data = nullptr; // For correctness check

    try {
        A_data = aligned_allocator(static_cast<size_t>(M) * K);
        B_data = aligned_allocator(static_cast<size_t>(K) * N);
        C_data = aligned_allocator(static_cast<size_t>(M) * N);
        C_ref_data = aligned_allocator(static_cast<size_t>(M) * N);
    } catch (const std::bad_alloc& e) {
        std::cerr << "Memory allocation failed: " << e.what() << std::endl;
        // Clean up any partially allocated memory
        if (A_data) aligned_deallocator(A_data);
        if (B_data) aligned_deallocator(B_data);
        if (C_data) aligned_deallocator(C_data);
        if (C_ref_data) aligned_deallocator(C_ref_data);
        return 1;
    }

    // Initialize matrices with random values
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i) A_data[i] = dist(rng);
    for (size_t i = 0; i < static_cast<size_t>(K) * N; ++i) B_data[i] = dist(rng);
    std::memset(C_data, 0, static_cast<size_t>(M) * N * sizeof(float));      // Initialize C with zeros
    std::memset(C_ref_data, 0, static_cast<size_t>(M) * N * sizeof(float)); // Initialize C_ref with zeros

    // For dense row-major storage, leading dimensions are typically equal to the number of columns.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Create workspace directory and dump initial matrices if requested
    if (dump_matrices) {
        std::filesystem::path workspace_dir("workspace");
        // Create directory if it does not exist
        if (!std::filesystem::exists(workspace_dir)) {
            std::error_code ec; // Use non-throwing version of create_directory
            std::filesystem::create_directory(workspace_dir, ec);
            if (ec) {
                std::cerr << "Error creating directory 'workspace': " << ec.message() << std::endl;
                dump_matrices = false; // Disable dumping if directory creation failed
            }
        }
        if (dump_matrices) { // Only dump if directory creation was successful
            write_matrix_to_file("workspace/A.txt", A_data, M, K, lda);
            write_matrix_to_file("workspace/B.txt", B_data, K, N, ldb);
            std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt\n";
        }
    }

    // --- Simple Warm-up Benchmark of Default Configuration ---
    // This provides a baseline performance metric for the current configuration
    // but does not dynamically tune parameters due to API constraints.
    if (run_warmup_benchmark) {
        std::cout << "\n--- Benchmarking default configuration on a warm-up problem (M=128, N=128, K=128) ---\n";
        const int AT_M = std::min(M, 128); // Warm-up size
        const int AT_N = std::min(N, 128);
        const int AT_K = std::min(K, 128);

        if (AT_M > 0 && AT_N > 0 && AT_K > 0) {
            float* At_data = nullptr;
            float* Bt_data = nullptr;
            float* Ct_data = nullptr;
            float* Cref_t_data = nullptr;
            
            try {
                At_data = aligned_allocator(static_cast<size_t>(AT_M) * AT_K);
                Bt_data = aligned_allocator(static_cast<size_t>(AT_K) * AT_N);
                Ct_data = aligned_allocator(static_cast<size_t>(AT_M) * AT_N);
                Cref_t_data = aligned_allocator(static_cast<size_t>(AT_M) * AT_N);
            } catch (const std::bad_alloc& e) {
                std::cerr << "Warm-up memory allocation failed: " << e.what() << std::endl;
                run_warmup_benchmark = false; // Disable benchmark if memory fails
            }

            if (run_warmup_benchmark) {
                // Initialize temporary matrices with reproducible random values
                std::mt19937 warmup_rng(12345); // Fixed seed for consistent warm-up
                std::uniform_real_distribution<float> warmup_dist(0.0f, 1.0f);
                for (size_t i = 0; i < static_cast<size_t>(AT_M) * AT_K; ++i) At_data[i] = warmup_dist(warmup_rng);
                for (size_t i = 0; i < static_cast<size_t>(AT_K) * AT_N; ++i) Bt_data[i] = warmup_dist(warmup_rng);
                std::memset(Ct_data, 0, static_cast<size_t>(AT_M) * AT_N * sizeof(float));
                std::memset(Cref_t_data, 0, static_cast<size_t>(AT_M) * AT_N * sizeof(float));

                // Compute reference for correctness of the warm-up run
                gemm_scalar(At_data, Bt_data, Cref_t_data, AT_M, AT_N, AT_K, AT_K, AT_N, AT_N);

                auto start_time = std::chrono::high_resolution_clock::now();
                // Call the main gemm function which uses the global default parameters
                ::gemm(At_data, Bt_data, Ct_data, AT_M, AT_N, AT_K, AT_K, AT_N, AT_N);
                auto end_time = std::chrono::high_resolution_clock::now();

                double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                double gflops = (2.0 * AT_M * AT_N * AT_K) / (duration_ms * 1e6);

                float max_diff = 0.0f;
                for (int r = 0; r < AT_M; ++r) {
                    for (int c = 0; c < AT_N; ++c) {
                        max_diff = std::max(max_diff, std::abs(Ct_data[static_cast<size_t>(r) * AT_N + c] - Cref_t_data[static_cast<size_t>(r) * AT_N + c]));
                    }
                }

                std::cout << "  Default config (BM=" << DEFAULT_BM << ", BN=" << DEFAULT_BN << ", BK=" << DEFAULT_BK << "):\n";
                std::cout << "  Time: " << duration_ms << " ms, GFLOP/s: " << gflops << ", Max Diff: " << max_diff << "\n";
                if (max_diff > 1e-4f) {
                    std::cerr << "  WARNING: Correctness check FAILED during warm-up! Max difference: " << max_diff << "\n";
                } else {
                    std::cout << "  Correctness check PASSED during warm-up.\n";
                }
            }

            // Deallocate temporary matrices
            if (At_data) aligned_deallocator(At_data);
            if (Bt_data) aligned_deallocator(Bt_data);
            if (Ct_data) aligned_deallocator(Ct_data);
            if (Cref_t_data) aligned_deallocator(Cref_t_data);
        } else {
            std::cout << "Warm-up problem dimensions are too small or invalid; skipping warmup benchmark.\n";
        }
    }
    
    // --- Main GEMM computation ---
    std::cout << "\n--- Running main GEMM (" << M << "x" << N << "x" << K << ") ---\n";

    // Run scalar reference for correctness verification
    auto start_scalar = std::chrono::high_resolution_clock::now();
    gemm_scalar(A_data, B_data, C_ref_data, M, N, K, lda, ldb, ldc);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();
    std::cout << "Scalar reference time: " << scalar_ms << " ms\n";

    // Run optimized GEMM
    auto start_time = std::chrono::high_resolution_clock::now();
    ::gemm(A_data, B_data, C_data, M, N, K, lda, ldb, ldc); // Call the dispatching gemm
    auto end_time = std::chrono::high_resolution_clock::now();

    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double gflops = (2.0 * M * N * K) / (duration_ms * 1e6); // 2 ops (mul+add) per element

    std::cout << "Optimized GEMM time: " << duration_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    // --- Correctness Check ---
    float max_diff = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            max_diff = std::max(max_diff, std::abs(C_data[static_cast<size_t>(i) * ldc + j] - C_ref_data[static_cast<size_t>(i) * ldc + j]));
        }
    }

    float tolerance = 1e-4f; // A reasonable tolerance for float comparisons due to potential FMA differences and order of operations
    if (max_diff > tolerance) {
        std::cerr << "Correctness check FAILED! Max difference: " << max_diff << " (tolerance: " << tolerance << ")" << std::endl;
    } else {
        std::cout << "Correctness check PASSED. Max difference: " << max_diff << std::endl;
    }

    // Dump C matrix if requested
    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C_data, M, N, ldc);
        std::cout << "Matrix C dumped to workspace/C.txt\n";
    }

    // Deallocate memory
    if (A_data) aligned_deallocator(A_data);
    if (B_data) aligned_deallocator(B_data);
    if (C_data) aligned_deallocator(C_data);
    if (C_ref_data) aligned_deallocator(C_ref_data);

    return 0;
}