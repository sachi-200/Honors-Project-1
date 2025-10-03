// =================================================================================================
// C++ CPU-Optimized Dense Matrix Multiplication (GEMM) for x86-64 with SIMD and Multi-threading
//
// Target Platform: x86_64 (Intel 11th Gen Core i7-1195G7)
// SIMD ISA: AVX2, FMA, AVX-512 (runtime dispatched)
// OS: Linux (GCC/Clang)
//
// This file implements a `float` (single-precision) GEMM (C = A * B) optimized for cache locality,
// SIMD vectorization, and multi-threading. It features runtime CPU capability dispatch.
//
// Matrix storage convention: Row-Major.
// A: M x K (stored as A[row_idx * lda + col_idx])
// B: K x N (stored as B[row_idx * ldb + col_idx])
// C: M x N (stored as C[row_idx * ldc + col_idx])
//
// Compile instructions (GCC/Clang):
//
// 1. For AVX-512 enabled build (target specifically AVX-512 capable CPUs):
//    g++ -std=c++17 -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//    (Note: `x86-64-v3` implies AVX2/FMA. `-mavx512f` explicitly enables AVX-512 F.)
//
// 2. For AVX2 enabled build (target specifically AVX2 capable CPUs, or as a fallback):
//    g++ -std=c++17 -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//    (Note: `x86-64-v2` implies SSE4.2, AVX, CLMUL. `-mavx2 -mfma` explicitly enables AVX2/FMA.)
//
// 3. For Portable Runtime Dispatch (recommended for broadest compatibility on modern CPUs):
//    g++ -std=c++17 -O3 -march=native -fopenmp gemm.cpp -o gemm_native
//    (Compiles with instruction sets supported by the *host* CPU. Runtime dispatch still works.)
//
// 4. For generic CPU (scalar fallback only):
//    g++ -std=c++17 -O3 -fopenmp gemm.cpp -o gemm_scalar
//
// =================================================================================================

#include <iostream>     // For std::cout, std::cerr
#include <vector>       // For std::vector
#include <string>       // For std::string, std::stoi, std::stoul
#include <random>       // For std::mt19937, std::uniform_real_distribution
#include <chrono>       // For std::chrono high_resolution_clock
#include <cassert>      // For assert
#include <numeric>      // Not strictly used, but common in numerical code
#include <algorithm>    // For std::min, std::max, std::fill
#include <filesystem>   // For std::filesystem::create_directories (C++17)
#include <fstream>      // For std::ofstream
#include <cmath>        // For std::abs
#include <limits>       // For std::numeric_limits
#include <cstring>      // For std::memcpy

#ifdef _OPENMP
#include <omp.h>        // For OpenMP pragmas and functions
#else
// Define dummy OpenMP functions/macros if OpenMP is not available
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {}
#endif

// Include SIMD intrinsics. This header generally includes SSE, AVX, AVX2, AVX-512 depending on compiler flags.
#include <immintrin.h>

// --- Tunable Parameters ---
// These constants define the blocking (tiling) strategy and register blocking for the micro-kernels.
// They are chosen to optimize for cache reuse (L1, L2, L3) and register utilization.
// BM, BN, BK relate to L2/L3 cache blocking. MR, NR relate to L1 cache and register blocking.

// Target CPU (Intel 11th Gen Core i7-1195G7) characteristics:
// - Architecture: Tiger Lake
// - Cores/Threads: 4 Cores / 8 Threads (SMT/HT = 2)
// - Cache Hierarchy:
//   - L1 Data Cache: 48 KB per core
//   - L2 Cache: 1.25 MB per core
//   - L3 Cache: 12 MB shared
//
// The chosen BM, BN, BK values aim to keep the working set (blocks of A, B, C) within L2 cache.
// Specifically, (BM * BK + BK * BN) * sizeof(float) should ideally fit within L2 cache.
// MR, NR define the register block size, optimizing for L1 cache and register pressure.

// AVX-512 specific parameters (optimized for 512-bit vectors, 16 floats)
constexpr int VEC_FLOATS_AVX512 = 16;
constexpr int MR_AVX512 = 8;     // M-dimension register block size (8 rows of C/A processed concurrently)
constexpr int NR_AVX512 = 16;    // N-dimension register block size (1 vector of 16 floats for B/C)
constexpr int BM_AVX512 = 128;   // M-dimension L2 cache block size (multiple of MR_AVX512, e.g., 128 = 16 * 8)
constexpr int BN_AVX512 = 192;   // N-dimension L2 cache block size (multiple of NR_AVX512, e.g., 192 = 12 * 16)
constexpr int BK_AVX512 = 64;    // K-dimension L2 cache block size
constexpr int UNROLL_K_AVX512 = 4; // K-loop unroll factor within micro-kernel for better instruction-level parallelism

// AVX2 specific parameters (optimized for 256-bit vectors, 8 floats)
// Similar considerations as AVX-512, but with smaller vector sizes and potentially adjusted cache blocks.
constexpr int VEC_FLOATS_AVX2 = 8;
constexpr int MR_AVX2 = 8;       // M-dimension register block size (8 rows of C/A processed concurrently)
constexpr int NR_AVX2 = 8;       // N-dimension register block size (1 vector of 8 floats for B/C)
constexpr int BM_AVX2 = 64;      // M-dimension L2 cache block size (multiple of MR_AVX2, e.g., 64 = 8 * 8)
constexpr int BN_AVX2 = 128;     // N-dimension L2 cache block size (multiple of NR_AVX2, e.g., 128 = 16 * 8)
constexpr int BK_AVX2 = 64;      // K-dimension L2 cache block size
constexpr int UNROLL_K_AVX2 = 4; // K-loop unroll factor

// Prefetching distance (in number of floats)
// _MM_HINT_T0: Prefetch to all levels of the cache hierarchy (L1/L2/L3).
// This value helps bring B matrix data into cache slightly ahead of time.
constexpr int PREFETCH_DISTANCE_K = 16; 

// --- Helper for memory alignment ---
// Used to allocate matrices A, B, C with required alignment for SIMD operations.
// AVX-512 requires 64-byte alignment, AVX2 requires 32-byte. 64-byte alignment satisfies both.
void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
#else // Linux/Unix-like systems
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else // Linux/Unix-like systems
    free(ptr);
#endif
}

// --- Function to write a matrix to a text file ---
// This helper function writes matrix contents to a specified file, respecting leading dimension.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::filesystem::path dir = std::filesystem::path(filename).parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

// --- Scalar Reference Implementation ---
// This is a basic triple-nested loop GEMM, used as a fallback and for correctness verification.
// It computes C = A * B in row-major format (C[m][n] = sum(A[m][k] * B[k][n])).
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float c_val = 0.0f;
            for (int k = 0; k < K; ++k) {
                c_val += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = c_val;
        }
    }
}

// --- AVX2 Implementation ---
#if defined(__AVX2__) && defined(__FMA__)
// This implementation uses AVX2 (256-bit vectors, 8 floats) and FMA instructions.
// It applies cache blocking and register blocking for performance.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Outer loops for M and N blocks, parallelized with OpenMP.
    // `collapse(2)` parallelizes both loops, distributing M x N blocks among threads.
    // `schedule(static)` provides static load balancing, suitable for dense GEMM
    // where workload per block is relatively uniform.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM_AVX2) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN_AVX2) {
            
            // Inner loops iterating over MR_AVX2 x NR_AVX2 micro-panels within the current M_block x N_block.
            for (int mm = m_block_start; mm < std::min(m_block_start + BM_AVX2, M); mm += MR_AVX2) {
                // `mr_actual` handles the M-dimension tail (if M_block_size is not a multiple of MR_AVX2)
                int mr_actual = std::min(MR_AVX2, std::min(m_block_start + BM_AVX2, M) - mm);

                for (int nn = n_block_start; nn < std::min(n_block_start + BN_AVX2, N); nn += NR_AVX2) {
                    // `nr_actual` handles the N-dimension tail
                    int nr_actual = std::min(NR_AVX2, std::min(n_block_start + BN_AVX2, N) - nn);

                    // Declare and initialize MR_AVX2 registers for accumulating C values.
                    // These will hold `MR_AVX2` rows, each containing `NR_AVX2` floats (1 AVX2 vector).
                    // This is an MR_AVX2 x NR_AVX2 register block for C.
                    __m256 c_regs[MR_AVX2];
                    for (int r = 0; r < mr_actual; ++r) {
                        c_regs[r] = _mm256_setzero_ps(); // Initialize accumulators to zero
                    }

                    // K-BLOCK loop (L2 cache blocking for K dimension).
                    // This loop processes `K` in chunks of `BK_AVX2` to improve B and A reuse in L2 cache.
                    for (int k_block_start = 0; k_block_start < K; k_block_start += BK_AVX2) {
                        int k_block_end = std::min(k_block_start + BK_AVX2, K);

                        // Inner K loop (L1 cache and register-level unrolling for K dimension).
                        // `UNROLL_K_AVX2` determines how many K iterations are processed concurrently.
                        for (int k = k_block_start; k < k_block_end; k += UNROLL_K_AVX2) {
                            for (int k_unroll_idx = 0; k_unroll_idx < UNROLL_K_AVX2; ++k_unroll_idx) {
                                int k_val = k + k_unroll_idx;
                                if (k_val >= k_block_end) break; // K-tail handling for unroll factor

                                // Prefetch B data for the next K iteration from memory into cache (T0 = L1/L2/L3)
                                // Pre-fetching B[k_val * ldb + nn + PREFETCH_DISTANCE_K] aims to bring in the next
                                // vector of B for future K loops.
                                _mm_prefetch((const char*)&B[k_val * ldb + nn + PREFETCH_DISTANCE_K], _MM_HINT_T0);

                                // Load B vector (NR_AVX2 floats).
                                // For N-tails (`nr_actual < NR_AVX2`), use a temporary buffer for safe loading and zero-padding.
                                __m256 b_vec;
                                alignas(32) float b_tmp_buffer[NR_AVX2]; // 32-byte aligned temporary buffer
                                if (nr_actual == NR_AVX2) {
                                    // Full vector load, unaligned load is fine as modern CPUs handle it efficiently.
                                    b_vec = _mm256_loadu_ps(&B[k_val * ldb + nn]);
                                } else {
                                    // Load only `nr_actual` elements from B, zero-pad the rest of the vector.
                                    // This prevents reading garbage data beyond matrix bounds.
                                    std::memcpy(b_tmp_buffer, &B[k_val * ldb + nn], nr_actual * sizeof(float));
                                    std::fill(b_tmp_buffer + nr_actual, b_tmp_buffer + NR_AVX2, 0.0f);
                                    b_vec = _mm256_load_ps(b_tmp_buffer); // Aligned load from temporary buffer
                                }

                                // Perform Fused Multiply-Add (FMA) for MR_AVX2 rows.
                                // Each A[r][k_val] is broadcast (replicated) to an AVX2 vector using _mm256_set1_ps
                                // and then multiplied by the loaded B vector. The result is added to the corresponding
                                // C accumulator register (c_regs[r]).
                                for (int r = 0; r < mr_actual; ++r) {
                                    __m256 a_val = _mm256_set1_ps(A[(mm + r) * lda + k_val]); // Broadcast A scalar
                                    c_regs[r] = _mm256_fmadd_ps(a_val, b_vec, c_regs[r]); // c_regs[r] = a_val * b_vec + c_regs[r]
                                }
                            }
                        } // End of inner K loop
                    } // End of K-BLOCK loop (all K contributions accumulated for this MRxNR micro-panel)

                    // Store the accumulated MR x NR C values back to memory.
                    for (int r = 0; r < mr_actual; ++r) {
                        float* c_ptr = &C[(mm + r) * ldc + nn];
                        if (nr_actual == NR_AVX2) {
                            // Full vector store.
                            _mm256_storeu_ps(c_ptr, c_regs[r]);
                        } else {
                            // For N-tails, store the full vector to a temporary buffer, then copy `nr_actual` elements.
                            // This ensures that only valid elements are written to C, preventing out-of-bounds writes
                            // and preserving existing data beyond the matrix boundary if C is part of a larger buffer.
                            alignas(32) float c_tmp_buffer[NR_AVX2];
                            _mm256_store_ps(c_tmp_buffer, c_regs[r]); // Store full vector to aligned temporary
                            std::memcpy(c_ptr, c_tmp_buffer, nr_actual * sizeof(float)); // Copy only the relevant `nr_actual` parts to C
                        }
                    }
                } // end nn (N-micro-panel loop)
            } // end mm (M-micro-panel loop)
        } // end n_block_start (N-block loop)
    } // end m_block_start (M-block loop)
}
#else // AVX2 and FMA not defined, provide a dummy function
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "AVX2 and FMA support not detected or enabled at compile time. Falling back to scalar GEMM for gemm_avx2 call." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX2__ && __FMA__

// --- AVX-512 Implementation ---
#if defined(__AVX512F__)
// This implementation uses AVX-512 (512-bit vectors, 16 floats) and FMA instructions.
// It leverages AVX-512's masking capabilities for efficient tail handling.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    // Outer loops for M and N blocks, parallelized with OpenMP.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM_AVX512) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN_AVX512) {
            
            // Inner loops iterating over MR_AVX512 x NR_AVX512 micro-panels within the current M_block x N_block.
            for (int mm = m_block_start; mm < std::min(m_block_start + BM_AVX512, M); mm += MR_AVX512) {
                int mr_actual = std::min(MR_AVX512, std::min(m_block_start + BM_AVX512, M) - mm);

                for (int nn = n_block_start; nn < std::min(n_block_start + BN_AVX512, N); nn += NR_AVX512) {
                    int nr_actual = std::min(NR_AVX512, std::min(n_block_start + BN_AVX512, N) - nn);

                    // Declare and initialize MR_AVX512 registers for accumulating C values.
                    // These will hold `MR_AVX512` rows, each containing `NR_AVX512` floats (1 AVX-512 vector).
                    // This forms an MR_AVX512 x NR_AVX512 register block for C.
                    __m512 c_regs[MR_AVX512];
                    for (int r = 0; r < mr_actual; ++r) {
                        c_regs[r] = _mm512_setzero_ps(); // Initialize accumulators to zero
                    }

                    // K-BLOCK loop (L2 cache blocking for K dimension).
                    for (int k_block_start = 0; k_block_start < K; k_block_start += BK_AVX512) {
                        int k_block_end = std::min(k_block_start + BK_AVX512, K);

                        // Inner K loop (L1 cache and register-level unrolling for K dimension).
                        for (int k = k_block_start; k < k_block_end; k += UNROLL_K_AVX512) {
                            for (int k_unroll_idx = 0; k_unroll_idx < UNROLL_K_AVX512; ++k_unroll_idx) {
                                int k_val = k + k_unroll_idx;
                                if (k_val >= k_block_end) break; // K-tail handling for unroll factor

                                // Prefetch B data for the next K iteration from memory into cache (T0 = L1/L2/L3)
                                // Prefetching along the N-dimension to ensure the next vector of B is in cache.
                                _mm_prefetch((const char*)&B[k_val * ldb + nn + PREFETCH_DISTANCE_K], _MM_HINT_T0);

                                // Load B vector (NR_AVX512 floats).
                                // For N-tails (`nr_actual < NR_AVX512`), `_mm512_maskz_loadu_ps` is used for masked loading,
                                // which loads `nr_actual` elements and zero-pads the rest of the vector. This prevents
                                // reading out-of-bounds and sets non-valid elements to zero for correct accumulation.
                                __m512 b_vec;
                                if (nr_actual == NR_AVX512) {
                                    b_vec = _mm512_loadu_ps(&B[k_val * ldb + nn]); // Unaligned load of 16 floats
                                } else {
                                    // Create a mask for `nr_actual` elements (e.g., if nr_actual=10, mask is 0x03FF)
                                    __mmask16 k_mask = (__mmask16)((1ULL << nr_actual) - 1);
                                    b_vec = _mm512_maskz_loadu_ps(k_mask, &B[k_val * ldb + nn]);
                                }

                                // Perform Fused Multiply-Add (FMA) for MR_AVX512 rows.
                                // Each A[r][k_val] is broadcast using _mm512_set1_ps and multiplied by the loaded B vector.
                                // The result is added to the corresponding C accumulator (c_regs[r]).
                                for (int r = 0; r < mr_actual; ++r) {
                                    __m512 a_val = _mm512_set1_ps(A[(mm + r) * lda + k_val]); // Broadcast A scalar
                                    c_regs[r] = _mm512_fmadd_ps(a_val, b_vec, c_regs[r]); // c_regs[r] = a_val * b_vec + c_regs[r]
                                }
                            }
                        } // End of inner K loop
                    } // End of K-BLOCK loop

                    // Store the accumulated MR x NR C values back to memory.
                    for (int r = 0; r < mr_actual; ++r) {
                        float* c_ptr = &C[(mm + r) * ldc + nn];
                        if (nr_actual == NR_AVX512) {
                            _mm512_storeu_ps(c_ptr, c_regs[r]); // Unaligned store of 16 floats
                        } else {
                            // Use a mask for partial store to C (N-tail).
                            // _mm512_mask_storeu_ps only writes to elements where the mask bit is set,
                            // leaving other elements in memory untouched. This prevents writing out-of-bounds
                            // and correctly handles the edges of the matrix block.
                            __mmask16 k_mask = (__mmask16)((1ULL << nr_actual) - 1);
                            _mm512_mask_storeu_ps(c_ptr, k_mask, c_regs[r]);
                        }
                    }
                } // end nn (N-micro-panel loop)
            } // end mm (M-micro-panel loop)
        } // end n_block_start (N-block loop)
    } // end m_block_start (M-block loop)
}
#else // AVX512 not defined, provide a dummy function
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    std::cerr << "AVX-512 support not detected or enabled at compile time. Falling back to AVX2 or scalar for gemm_avx512 call." << std::endl;
    // If this function is called when AVX-512 is not compiled, dispatch to AVX2 if available, else scalar.
    // This provides a robust fallback path even if __builtin_cpu_supports is bypassed.
#if defined(__AVX2__) && defined(__FMA__)
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}
#endif // __AVX512F__

// --- Top-level GEMM with Runtime Dispatch ---
// This function dispatches to the most optimized GEMM kernel available at runtime.
// It checks for AVX-512 first, then AVX2, falling back to scalar if neither is available.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    // Function pointer to hold the chosen GEMM implementation
    void (*gemm_impl)(const float*, const float*, float*, int, int, int, int, int, int) = nullptr;

#if defined(__AVX512F__)
    // Check for AVX-512F support at runtime (GCC/Clang intrinsic).
    // Note: Intel 11th Gen Core i7 (Tiger Lake) supports AVX-512F.
    if (__builtin_cpu_supports("avx512f")) {
        gemm_impl = gemm_avx512;
        // std::cout << "Using AVX-512 kernel." << std::endl; // Uncomment for debugging dispatch
    } else
#endif
#if defined(__AVX2__) && defined(__FMA__)
    // Check for AVX2 support at runtime.
    if (__builtin_cpu_supports("avx2")) {
        gemm_impl = gemm_avx2;
        // std::cout << "Using AVX2 kernel." << std::endl; // Uncomment for debugging dispatch
    } else
#endif
    {
        // Fallback to scalar implementation if no SIMD support is detected at runtime
        gemm_impl = gemm_scalar;
        // std::cout << "Using scalar kernel." << std::endl; // Uncomment for debugging dispatch
    }

    // Call the chosen GEMM implementation
    gemm_impl(A, B, C, M, N, K, lda, ldb, ldc);
}

// --- Main function for demo and testing ---
int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 512, N = 512, K = 512;
    unsigned int seed = 42; // Default random seed
    int num_threads = omp_get_max_threads(); // Default to all available logical cores
    bool dump_matrices = false; // Flag to dump matrices to files
    bool check_correctness = false; // Flag to compare with scalar reference

    int current_arg_idx = 1;

    // Parse M, N, K as optional positional arguments first
    // Only parse if the argument doesn't start with '-' (indicating it's a flag)
    if (current_arg_idx < argc && argv[current_arg_idx][0] != '-') {
        M = std::stoi(argv[current_arg_idx++]);
    }
    if (current_arg_idx < argc && argv[current_arg_idx][0] != '-') {
        N = std::stoi(argv[current_arg_idx++]);
    }
    if (current_arg_idx < argc && argv[current_arg_idx][0] != '-') {
        K = std::stoi(argv[current_arg_idx++]);
    }

    // Now parse the remaining arguments as flags
    for (; current_arg_idx < argc; ++current_arg_idx) {
        std::string arg = argv[current_arg_idx];
        if (arg == "-s" && current_arg_idx + 1 < argc) {
            seed = std::stoul(argv[++current_arg_idx]);
        } else if (arg == "-t" && current_arg_idx + 1 < argc) {
            num_threads = std::stoi(argv[++current_arg_idx]);
        } else if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "--check-correctness") {
            check_correctness = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-s <seed>] [-t <threads>] [--dump-matrices] [--check-correctness] [--help]\n";
            std::cout << "  [M], [N], [K]: Optional matrix dimensions (default: " << M << "x" << N << "x" << K << ").\n";
            std::cout << "  -s: Random seed (default: " << seed << ").\n";
            std::cout << "  -t: Number of OpenMP threads (default: max available, currently " << omp_get_max_threads() << ").\n";
            std::cout << "  --dump-matrices: Write matrices A, B, C to 'workspace/A.txt', 'workspace/B.txt', 'workspace/C.txt'.\n";
            std::cout << "  --check-correctness: Compare result with scalar reference implementation.\n";
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    // Set number of OpenMP threads
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        std::cout << "Using " << omp_get_max_threads() << " OpenMP threads." << std::endl;
    } else {
        std::cerr << "Invalid number of threads specified: " << num_threads << ". Using default max threads." << std::endl;
        num_threads = omp_get_max_threads();
        omp_set_num_threads(num_threads);
    }
    
    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Allocate aligned matrices.
    // Use 64-byte alignment to satisfy AVX-512 requirements (also works for AVX2's 32-byte).
    size_t alignment = 64; 
    float* A = (float*)aligned_malloc(M * K * sizeof(float), alignment);
    float* B = (float*)aligned_malloc(K * N * sizeof(float), alignment);
    float* C = (float*)aligned_malloc(M * N * sizeof(float), alignment);
    float* C_ref = nullptr; // For correctness checking

    if (!A || !B || !C) {
        std::cerr << "Error: Memory allocation failed! Please check requested matrix sizes.\n";
        aligned_free(A); aligned_free(B); aligned_free(C);
        return 1;
    }

    // Initialize matrices A and B with random float values, and C with zeros.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) B[i] = dis(gen);
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f; // C must be initialized to zero for C = A * B

    // Define leading dimensions. Assuming tightly packed row-major for this example.
    int lda = K; // For M x K matrix A, lda is K if tightly packed
    int ldb = N; // For K x N matrix B, ldb is N if tightly packed
    int ldc = N; // For M x N matrix C, ldc is N if tightly packed
    
    // Dump matrices A and B to files if requested
    if (dump_matrices) {
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B written to workspace/A.txt and workspace/B.txt\n";
    }

    // Optional: Warm-up run to prime caches and ensure frequency scaling is stable.
    // Uses small dummy matrices to avoid affecting benchmark results for main problem size.
    if (M > 0 && N > 0 && K > 0) {
        constexpr int WARMUP_DIM = 64; // Smaller dimensions for warm-up
        float* A_dummy = (float*)aligned_malloc(WARMUP_DIM * WARMUP_DIM * sizeof(float), alignment);
        float* B_dummy = (float*)aligned_malloc(WARMUP_DIM * WARMUP_DIM * sizeof(float), alignment);
        float* C_dummy = (float*)aligned_malloc(WARMUP_DIM * WARMUP_DIM * sizeof(float), alignment);
        if (A_dummy && B_dummy && C_dummy) {
            for (int i = 0; i < WARMUP_DIM * WARMUP_DIM; ++i) { 
                A_dummy[i] = dis(gen); B_dummy[i] = dis(gen); C_dummy[i] = 0.0f; 
            }
            gemm(A_dummy, B_dummy, C_dummy, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM, WARMUP_DIM);
            aligned_free(A_dummy); aligned_free(B_dummy); aligned_free(C_dummy);
            // std::cout << "Warm-up run completed.\n"; // Can be uncommented for verbose output
        } else {
            std::cerr << "Warning: Failed to allocate memory for warm-up. Skipping warm-up.\n";
        }
    }

    // Perform the GEMM computation and time it
    std::cout << "Starting GEMM computation...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time and performance
    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    double time_s = elapsed_ms.count() / 1000.0;

    double flops = 2.0 * static_cast<double>(M) * N * K; // 2 operations (mul, add) per element
    double gflops_per_sec = (flops / time_s) / 1e9;

    std::cout << "\n--- Performance Report ---\n";
    std::cout << "Computation finished in: " << elapsed_ms.count() << " ms\n";
    std::cout << "Effective GFLOP/s: " << gflops_per_sec << std::endl;
    std::cout << "--------------------------\n";

    // Dump matrix C to file if requested
    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Matrix C written to workspace/C.txt\n";
    }

    // Optional: Correctness check against the scalar reference implementation
    if (check_correctness) {
        std::cout << "Running correctness check with scalar reference...\n";
        C_ref = (float*)aligned_malloc(M * N * sizeof(float), alignment);
        if (!C_ref) {
            std::cerr << "Error: Memory allocation for C_ref failed during correctness check!\n";
        } else {
            for (int i = 0; i < M * N; ++i) C_ref[i] = 0.0f; // Ensure C_ref is zeroed before scalar GEMM
            gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

            float max_diff = 0.0f;
            for (int i = 0; i < M * N; ++i) {
                max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
            }

            // Define a tolerance based on matrix dimensions and float precision (machine epsilon)
            // A common rule of thumb for GEMM is K * epsilon * max_element_magnitude
            // Assuming A, B values are around [-1, 1], so max_element_magnitude is approx 1.0f.
            // The result can grow up to K * (max_A * max_B), so the error also scales with K.
            float max_val_expected = static_cast<float>(K); 
            float tolerance = K * std::numeric_limits<float>::epsilon() * max_val_expected * 10.0f; // Multiplier 10 for robustness

            if (max_diff < tolerance) {
                std::cout << "Correctness check PASSED. Max difference: " << max_diff << " (tolerance: " << tolerance << ")\n";
            } else {
                std::cerr << "Correctness check FAILED. Max difference: " << max_diff << " (tolerance: " << tolerance << ")\n";
                // Optionally print first few differing elements for debugging
                // int diff_count = 0;
                // for (int i = 0; i < M; ++i) {
                //     for (int j = 0; j < N; ++j) {
                //         if (std::abs(C[i*ldc + j] - C_ref[i*ldc + j]) >= tolerance) {
                //             std::cerr << "C[" << i << "][" << j << "] ours=" << C[i*ldc + j] << ", ref=" << C_ref[i*ldc + j] << ", diff=" << std::abs(C[i*ldc + j] - C_ref[i*ldc + j]) << "\n";
                //             diff_count++;
                //             if (diff_count > 10) { std::cerr << "...and more\n"; break; }
                //         }
                //     }
                //     if (diff_count > 10) break;
                // }
            }
        }
    }

    // Clean up dynamically allocated memory
    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    if (C_ref) aligned_free(C_ref);

    return 0;
}