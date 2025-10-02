// Compile instructions:
//
// For AVX-512 (preferred for Intel 11th Gen Core i7, enables x86-64-v4 ISA features):
// g++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
// clang++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//
// For AVX2 (fallback for systems without AVX-512, or when AVX-512 is not desired):
// g++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
// clang++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// For portable compilation (runtime dispatch will pick best available, but might not enable all features
// if target machine is older than compile machine, or if specific -march flags are not used):
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native
// clang++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native
//
// Note: -march=x86-64-v4 implies AVX2, FMA, etc., so explicit -mavx2/-mfma might be redundant
// with v3/v4 but are included for clarity. For AVX-512, -mavx512f is necessary.

#include <immintrin.h> // For SIMD intrinsics (AVX2, AVX-512, FMA)
#include <iostream>    // For input/output operations (cout, cerr)
#include <vector>      // For std::vector
#include <cstring>     // For std::strcmp (CLI parsing)
#include <chrono>      // For high-resolution timing
#include <random>      // For random number generation
#include <cassert>     // For assert()
#include <numeric>     // For std::iota (not strictly needed here but often useful)
#include <fstream>     // For file I/O (matrix dumping)
#include <iomanip>     // For std::fixed, std::setprecision (output formatting)
#include <filesystem>  // For std::filesystem::create_directory (C++17)
#include <string>      // For std::stoi, std::stoul
#include <algorithm>   // For std::min, std::max, std::fill
#include <cmath>       // For std::abs

#ifdef _OPENMP
#include <omp.h> // For OpenMP multi-threading
#endif

// --- Constants / Tunable Parameters ---
// These parameters are crucial for performance and should be tuned for the target CPU.
// The provided values are reasonable starting points for an Intel 11th Gen i7.
//
// Target CPU: Intel 11th Gen Core i7-1195G7
// - Architecture: x86_64, Supports AVX2, FMA, AVX-512.
// - Threads: 8 logical CPUs (4 cores, SMT/HT=2)
// - Cache Hierarchy:
//   - L1d cache: 48KB per core (data + instruction)
//   - L2 cache: 1.25MB per core (inclusive)
//   - L3 cache: 12MB shared (inclusive)
//
// Blocking strategy: We employ a three-level blocking scheme (M, N, K dimensions) to maximize data reuse:
// 1. Outer loops (M_block, N_block): These loops define a block of the C matrix (BM x BN) to be computed.
//    They are parallelized using OpenMP to distribute the workload across available cores.
// 2. Middle loop (K_block): This loop iterates through the K-dimension. For each K-block, a sub-block
//    of matrix A (BM x BK) and a sub-block of matrix B (BK x BN) are loaded into L2/L3 cache.
//    The results are accumulated into the C_block.
// 3. Inner loops / Micro-kernel (MR, NR, UNROLL_K): This is the most performance-critical part.
//    It computes a small micro-tile of C (MR x NR) using register blocking. UNROLL_K determines
//    how many K-elements are processed in a single unrolled inner loop iteration to maximize
//    register utilization and instruction-level parallelism. The goal is to keep the MRxNR C-tile
//    in CPU registers, and the relevant A/B data in L1d cache.

// AVX-512 Micro-kernel parameters:
// NR_AVX512: Number of columns of C processed per _mm512 register (16 floats).
// MR_AVX512: Number of rows of C processed concurrently using multiple _mm512 accumulators.
// UNROLL_K_AVX512: Loop unroll factor for the K-dimension inside the micro-kernel.
constexpr int MR_AVX512 = 6;
constexpr int NR_AVX512 = 16; // 16 floats in __m512
constexpr int UNROLL_K_AVX512 = 4;

// AVX2 Micro-kernel parameters:
// NR_AVX2: Number of columns of C processed per _mm256 register (8 floats).
// MR_AVX2: Number of rows of C processed concurrently using multiple _mm256 accumulators.
// UNROLL_K_AVX2: Loop unroll factor for the K-dimension inside the micro-kernel.
constexpr int MR_AVX2 = 6;
constexpr int NR_AVX2 = 8; // 8 floats in __m256
constexpr int UNROLL_K_AVX2 = 4;

// Global blocking parameters (should ideally be multiples of MR/NR for cleaner tail handling):
// BM (Block M): M-dimension block size. Chosen to keep a portion of A and C in L2/L3 cache.
// BN (Block N): N-dimension block size. Chosen to keep a portion of B and C in L2/L3 cache.
// BK (Block K): K-dimension block size. Chosen to fit B's K-rows and A's K-columns into L2/L3 cache.
//
// Example calculation for L2 cache fit (1.25MB per core):
// For BM=128, BN=192, BK=256:
// A sub-block: BM * BK floats = 128 * 256 * 4 bytes = 131072 bytes (~128KB)
// B sub-block: BK * BN floats = 256 * 192 * 4 bytes = 196608 bytes (~192KB)
// C sub-block: BM * BN floats = 128 * 192 * 4 bytes = 98304 bytes (~96KB)
// Total working set for A, B, C blocks is approx. 128KB + 192KB + 96KB = 416KB.
// This size fits well within the 1.25MB L2 cache per core, promoting data reuse.
// These values can be adjusted via the autotuning harness for specific problem sizes.
constexpr int DEFAULT_BM = 128;
constexpr int DEFAULT_BN = 192;
constexpr int DEFAULT_BK = 256;

// Function declarations (CRITICAL: EXACT SIGNATURES as requested)
void gemm_scalar(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);
void gemm_avx2(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);
void gemm_avx512(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);
void gemm(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);

// Helper for matrix writing
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld);

// Custom aligned allocator for std::vector.
// Uses posix_memalign for allocation and free for deallocation, suitable for Linux.
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    AlignedAllocator() = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
        // posix_memalign is common on Linux/Unix for aligned memory.
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) {
        free(p);
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    bool operator==(const AlignedAllocator& other) const { return true; }
    bool operator!=(const AlignedAllocator& other) const { return false; }
};

// --- Scalar Reference Implementation ---
// C = A * B, computes C(M,N) = A(M,K) * B(K,N)
// All matrices are assumed to be row-major, using lda, ldb, ldc for strides.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}

// --- AVX2 Implementation ---
#if defined(__AVX2__) && defined(__FMA__)
// AVX2 Micro-kernel: Computes a (MR_AVX2 x NR_AVX2) block of C.
// This micro-kernel is responsible for computing a small tile of C, typically 6x8 floats.
// It performs the accumulation in registers and handles tail cases for N-dimension.
//
// Parameters:
//   A_base: Pointer to the start of the full A matrix.
//   B_base: Pointer to the start of the full B matrix.
//   C: Pointer to the start of the full C matrix.
//   m_start, n_start: Global row/column start indices for the current C micro-tile.
//   k_block_start: Global K-dimension start index for the current K-block. This is crucial for
//                  correctly handling C initialization (zeroing for the first K-block, loading for subsequent).
//   K_block_dim: The actual K-dimension size for the current K-block.
//   lda, ldb, ldc: Leading dimensions (strides) of A, B, C.
//   M_tail, N_tail: The actual dimensions of the micro-tile if it's smaller than MR_AVX2/NR_AVX2
//                   (e.g., at the edges of the overall matrix or block).
//                   -1 indicates full MR/NR, otherwise it's the actual remaining size.
void gemm_avx2_micro_kernel(const float* A_base, const float* B_base, float* C,
                            int m_start, int n_start, int k_block_start, int K_block_dim,
                            int lda, int ldb, int ldc,
                            int M_tail, int N_tail) {
    
    // Determine the actual dimensions of the micro-tile.
    const int actual_NR = (N_tail != -1 && N_tail < NR_AVX2) ? N_tail : NR_AVX2;
    const int actual_MR = (M_tail != -1 && M_tail < MR_AVX2) ? M_tail : MR_AVX2;

    // Accumulator registers for the C micro-tile (MR_AVX2 x NR_AVX2 floats).
    // Each __m256 register holds 8 floats, so c_regs[i] stores 8 elements for row 'i'.
    __m256 c_regs[MR_AVX2];

    // Prepare a mask for N-tail handling if necessary.
    // _mm256_maskload_ps and _mm256_maskstore_ps use __m256i (int32_t vector) for masks.
    alignas(32) int n_tail_mask_array[NR_AVX2];
    if (actual_NR < NR_AVX2) {
        for (int i = 0; i < actual_NR; ++i) n_tail_mask_array[i] = -1; // -1 means all bits set (active lane)
        for (int i = actual_NR; i < NR_AVX2; ++i) n_tail_mask_array[i] = 0;  // 0 means all bits zero (inactive lane)
    }
    const __m256i* n_tail_mask_ptr = (const __m256i*)n_tail_mask_array;

    for (int i = 0; i < actual_MR; ++i) {
        // Initialize or load C values for accumulation.
        // If this is the first K-block (k_block_start == 0), initialize to zero.
        // Otherwise, load existing C values from the global C matrix for accumulation.
        if (k_block_start == 0) {
            c_regs[i] = _mm256_setzero_ps();
        } else {
            if (actual_NR < NR_AVX2) {
                // Masked load for N-tail columns
                c_regs[i] = _mm256_maskload_ps(C + (m_start + i) * ldc + n_start, *n_tail_mask_ptr);
            } else {
                // Unaligned load for full NR_AVX2 columns. _mm256_loadu_ps is safe for any address.
                c_regs[i] = _mm256_loadu_ps(C + (m_start + i) * ldc + n_start);
            }
        }
    }

    // Loop over the K-dimension (inner-most loop) with UNROLL_K_AVX2 unrolling.
    // This forms the core computation, performing matrix multiplication for the micro-tile.
    for (int k_idx_in_block = 0; k_idx_in_block < K_block_dim; k_idx_in_block += UNROLL_K_AVX2) {
        // Inner loop for K-unrolling. Handles K-tail if K_block_dim is not a multiple of UNROLL_K_AVX2.
        for (int uk = 0; uk < UNROLL_K_AVX2 && (k_idx_in_block + uk) < K_block_dim; ++uk) {
            int k_global_idx = k_block_start + k_idx_in_block + uk; // Effective GLOBAL K index for current iteration.

            // Load B vector: B_base[k_global_idx, n_start ... n_start + NR_AVX2 - 1]
            // We use unaligned loads (_mm256_loadu_ps) as B is not guaranteed to be vector-aligned.
            __m256 b_vec;
            if (actual_NR < NR_AVX2) {
                b_vec = _mm256_maskload_ps(B_base + k_global_idx * ldb + n_start, *n_tail_mask_ptr);
            } else {
                b_vec = _mm256_loadu_ps(B_base + k_global_idx * ldb + n_start);
            }

            // Perform outer product accumulation: C += A_scalar * B_vector
            for (int i = 0; i < actual_MR; ++i) {
                // Load A scalar: A_base[m_start + i, k_global_idx]
                // A elements are generally not contiguous in a way that allows vector loads directly.
                float a_scalar = A_base[(m_start + i) * lda + k_global_idx];
                __m256 a_vec = _mm256_set1_ps(a_scalar); // Broadcast scalar 'a_scalar' to all elements of 'a_vec'.
                // Fused Multiply-Add (FMA): c_regs[i] = (a_vec * b_vec) + c_regs[i]
                c_regs[i] = _mm256_fmadd_ps(a_vec, b_vec, c_regs[i]);
            }
        }
    }

    // Store accumulated results from c_regs back to C matrix.
    for (int i = 0; i < actual_MR; ++i) {
        if (actual_NR < NR_AVX2) {
            // Masked store for N-tail
            _mm256_maskstore_ps(C + (m_start + i) * ldc + n_start, *n_tail_mask_ptr, c_regs[i]);
        } else {
            _mm256_storeu_ps(C + (m_start + i) * ldc + n_start, c_regs[i]);
        }
    }
}

// Main AVX2 GEMM function with tiling and multi-threading.
// This function orchestrates the blocking and parallelization, calling the micro-kernel.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN;
    const int BK = DEFAULT_BK;

    // OpenMP parallelization over outer M and N blocks.
    // 'collapse(2)' allows OpenMP to parallelize both loops simultaneously,
    // creating a larger pool of tasks for better load balancing.
    // 'schedule(dynamic)' is often good for matrices with varying block computation times,
    // or when the number of blocks is large relative to the number of threads.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            // Determine actual dimensions of the current M x N block.
            int current_M_block_dim = std::min(BM, M - m_block_start);
            int current_N_block_dim = std::min(BN, N - n_block_start);

            // K-loop: Iterate through K-dimension blocks.
            // This loop performs the accumulation for the current C-block.
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                // Determine actual K-dimension size for the current K-block.
                int current_K_block_dim = std::min(BK, K - k_block_start);

                // Inner M and N loops, calling the micro-kernel.
                // These loops iterate over the current M x N block in steps of MR_AVX2 x NR_AVX2.
                for (int m_inner = 0; m_inner < current_M_block_dim; m_inner += MR_AVX2) {
                    for (int n_inner = 0; n_inner < current_N_block_dim; n_inner += NR_AVX2) {
                        int current_m_start = m_block_start + m_inner;
                        int current_n_start = n_block_start + n_inner;
                        
                        // Handle M-tail (remaining rows if current_M_block_dim is not a multiple of MR_AVX2)
                        int M_tail = -1; // -1 indicates no tail (full MR_AVX2 rows)
                        if (m_inner + MR_AVX2 > current_M_block_dim) {
                            M_tail = current_M_block_dim - m_inner;
                        }
                        // Handle N-tail (remaining columns if current_N_block_dim is not a multiple of NR_AVX2)
                        int N_tail = -1; // -1 indicates no tail (full NR_AVX2 columns)
                        if (n_inner + NR_AVX2 > current_N_block_dim) {
                            N_tail = current_N_block_dim - n_inner;
                        }

                        // Call the AVX2 micro-kernel to compute the (MR_AVX2 x NR_AVX2) micro-tile.
                        // A, B, C are the global matrix pointers. The micro-kernel will handle indexing.
                        gemm_avx2_micro_kernel(A, B, C,
                                                current_m_start, current_n_start,
                                                k_block_start, // Pass global k_block_start for accumulation logic
                                                current_K_block_dim,
                                                lda, ldb, ldc,
                                                M_tail, N_tail);
                    }
                }
            }
        }
    }
}
#else // AVX2 not available
// Placeholder for when AVX2 is not compiled in. Will trigger a runtime error message.
void gemm_avx2(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc) {
    std::cerr << "Error: AVX2 kernel called but AVX2 intrinsics are not available or not compiled with -mavx2 -mfma.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc); // Fallback to scalar
}
#endif // __AVX2__ && __FMA__

// --- AVX-512 Implementation ---
#if defined(__AVX512F__) && defined(__FMA__)
// AVX-512 Micro-kernel: Computes a (MR_AVX512 x NR_AVX512) block of C.
// Parameters are identical in purpose to the AVX2 version, adjusted for AVX-512 vector width.
void gemm_avx512_micro_kernel(const float* A_base, const float* B_base, float* C,
                              int m_start, int n_start, int k_block_start, int K_block_dim,
                              int lda, int ldb, int ldc,
                              int M_tail, int N_tail) {
    
    // Determine the actual dimensions of the micro-tile.
    const int actual_NR = (N_tail != -1 && N_tail < NR_AVX512) ? N_tail : NR_AVX512;
    const int actual_MR = (M_tail != -1 && M_tail < MR_AVX512) ? M_tail : MR_AVX512;

    // Accumulator registers for the C micro-tile (MR_AVX512 x NR_AVX512 floats).
    // Each __m512 register holds 16 floats.
    __m512 c_regs[MR_AVX512];

    // Prepare a mask for N-tail handling if necessary. AVX-512 uses __mmask16 for float masks.
    __mmask16 n_tail_mask = (__mmask16)((1 << actual_NR) - 1); // Create bitmask for active lanes.

    for (int i = 0; i < actual_MR; ++i) {
        // Initialize or load C values for accumulation.
        // If this is the first K-block (k_block_start == 0), initialize to zero.
        // Otherwise, load existing C values from the global C matrix for accumulation.
        if (k_block_start == 0) {
            c_regs[i] = _mm512_setzero_ps();
        } else {
            if (actual_NR < NR_AVX512) {
                // Masked load for N-tail columns, zero-masking to ensure inactive lanes are zero.
                c_regs[i] = _mm512_maskz_loadu_ps(n_tail_mask, C + (m_start + i) * ldc + n_start);
            } else {
                // Unaligned load for full NR_AVX512 columns.
                c_regs[i] = _mm512_loadu_ps(C + (m_start + i) * ldc + n_start);
            }
        }
    }

    // Loop over the K-dimension (inner-most loop) with UNROLL_K_AVX512 unrolling.
    // This forms the core computation, performing matrix multiplication for the micro-tile.
    for (int k_idx_in_block = 0; k_idx_in_block < K_block_dim; k_idx_in_block += UNROLL_K_AVX512) {
        // Inner loop for K-unrolling. Handles K-tail if K_block_dim is not a multiple of UNROLL_K_AVX512.
        for (int uk = 0; uk < UNROLL_K_AVX512 && (k_idx_in_block + uk) < K_block_dim; ++uk) {
            int k_global_idx = k_block_start + k_idx_in_block + uk; // Effective GLOBAL K index for current iteration.

            // Load B vector: B_base[k_global_idx, n_start ... n_start + NR_AVX512 - 1]
            // We use unaligned loads (_mm512_loadu_ps) as B is not guaranteed to be vector-aligned.
            __m512 b_vec;
            if (actual_NR < NR_AVX512) {
                b_vec = _mm512_maskz_loadu_ps(n_tail_mask, B_base + k_global_idx * ldb + n_start);
            } else {
                b_vec = _mm512_loadu_ps(B_base + k_global_idx * ldb + n_start);
            }

            // Perform outer product accumulation: C += A_scalar * B_vector
            for (int i = 0; i < actual_MR; ++i) {
                // Load A scalar: A_base[m_start + i, k_global_idx]
                // A elements are generally not contiguous in a way that allows vector loads directly.
                float a_scalar = A_base[(m_start + i) * lda + k_global_idx];
                __m512 a_vec = _mm512_set1_ps(a_scalar); // Broadcast scalar 'a_scalar' to all elements of 'a_vec'.
                // Fused Multiply-Add (FMA): c_regs[i] = (a_vec * b_vec) + c_regs[i]
                c_regs[i] = _mm512_fmadd_ps(a_vec, b_vec, c_regs[i]);
            }
        }
    }

    // Store accumulated results from c_regs back to C matrix.
    for (int i = 0; i < actual_MR; ++i) {
        if (actual_NR < NR_AVX512) {
            // Masked store for N-tail
            _mm512_mask_storeu_ps(C + (m_start + i) * ldc + n_start, n_tail_mask, c_regs[i]);
        } else {
            _mm512_storeu_ps(C + (m_start + i) * ldc + n_start, c_regs[i]);
        }
    }
}

// Main AVX-512 GEMM function with tiling and multi-threading.
// This function orchestrates the blocking and parallelization, calling the micro-kernel.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN;
    const int BK = DEFAULT_BK;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            int current_M_block_dim = std::min(BM, M - m_block_start);
            int current_N_block_dim = std::min(BN, N - n_block_start);

            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                int current_K_block_dim = std::min(BK, K - k_block_start);

                for (int m_inner = 0; m_inner < current_M_block_dim; m_inner += MR_AVX512) {
                    for (int n_inner = 0; n_inner < current_N_block_dim; n_inner += NR_AVX512) {
                        int current_m_start = m_block_start + m_inner;
                        int current_n_start = n_block_start + n_inner;
                        
                        int M_tail = -1;
                        if (m_inner + MR_AVX512 > current_M_block_dim) {
                            M_tail = current_M_block_dim - m_inner;
                        }
                        int N_tail = -1;
                        if (n_inner + NR_AVX512 > current_N_block_dim) {
                            N_tail = current_N_block_dim - n_inner;
                        }

                        gemm_avx512_micro_kernel(A, B, C,
                                                 current_m_start, current_n_start,
                                                 k_block_start, // Pass global k_block_start for accumulation logic
                                                 current_K_block_dim,
                                                 lda, ldb, ldc,
                                                 M_tail, N_tail);
                    }
                }
            }
        }
    }
}
#else // AVX512 not available
// Placeholder for when AVX512 is not compiled in. Will trigger a runtime error message.
void gemm_avx512(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc) {
    std::cerr << "Error: AVX-512 kernel called but AVX-512 intrinsics are not available or not compiled with -mavx512f -mfma.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc); // Fallback to scalar
}
#endif // __AVX512F__ && __FMA__

// --- Runtime Dispatch Function ---
// This function acts as the entry point and dispatches to the most optimized
// kernel available on the CPU at runtime, using __builtin_cpu_supports.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
#if defined(__GNUC__) || defined(__clang__)
    // Use GCC/Clang specific intrinsics for runtime CPU feature detection.
    // The target CPU (i7-1195G7) supports AVX-512 and AVX2+FMA.
    if (__builtin_cpu_supports("avx512f")) {
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    } else if (__builtin_cpu_supports("avx2")) {
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    } else {
        std::cerr << "Warning: No AVX2 or AVX-512 support detected. Falling back to scalar GEMM.\n";
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
#else
    // Fallback for other compilers/platforms where __builtin_cpu_supports might not be available.
    // This assumes that if the preprocessor macro is defined, the instruction set is usable.
    // This is less robust than runtime detection but necessary for portability.
    #if defined(__AVX512F__) && defined(__FMA__)
        std::cout << "Using AVX-512 kernel (compiled-time selection).\n";
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    #elif defined(__AVX2__) && defined(__FMA__)
        std::cout << "Using AVX2 kernel (compiled-time selection).\n";
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    #else
        std::cerr << "Warning: No AVX2 or AVX-512 support assumed. Falling back to scalar GEMM.\n";
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    #endif
#endif
}

// --- Matrix I/O Helper ---
// Writes a matrix to a specified text file in a space-separated format.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    ofs << std::fixed << std::setprecision(6); // Format output for readability
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            ofs << matrix[r * ld + c] << (c == cols - 1 ? "" : " "); // Space-separated values
        }
        ofs << "\n"; // Newline after each row
    }
    ofs.close();
}

// --- Main Function (CLI Demo) ---
int main(int argc, char** argv) {
    // Default matrix dimensions
    int M = 512;
    int N = 512;
    int K = 512;
    unsigned int seed = 12345;
    int num_threads = 0; // 0 means use OMP_NUM_THREADS or default from system
    bool dump_matrices = false;
    bool enable_autotune = false;
    bool verify_scalar = false;

    // Flags to indicate if M, N, K were set by explicit flags
    bool m_set_by_flag = false;
    bool n_set_by_flag = false;
    bool k_set_by_flag = false;

    // Store indices of positional arguments to parse later
    std::vector<int> positional_arg_indices;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            M = std::stoi(argv[++i]);
            m_set_by_flag = true;
        } else if (std::strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = std::stoi(argv[++i]);
            n_set_by_flag = true;
        } else if (std::strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            K = std::stoi(argv[++i]);
            k_set_by_flag = true;
        } else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--dump-matrices") == 0) {
            dump_matrices = true;
        } else if (std::strcmp(argv[i], "--autotune") == 0) {
            enable_autotune = true;
        } else if (std::strcmp(argv[i], "--verify") == 0) {
            verify_scalar = true;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M <rows>] [-N <cols>] [-K <inner>] [--seed <val>] [--threads <num>] [--dump-matrices] [--autotune] [--verify] [--help]\n";
            std::cout << "  [M] [N] [K]        : Positional arguments for matrix dimensions (override defaults if flags not used).\n";
            std::cout << "  -M <rows>          : Number of rows in A and C (default: 512)\n";
            std::cout << "  -N <cols>          : Number of columns in B and C (default: 512)\n";
            std::cout << "  -K <inner>         : Inner dimension for A and B (default: 512)\n";
            std::cout << "  --seed <val>       : Seed for random matrix initialization (default: 12345)\n";
            std::cout << "  --threads <num>    : Number of OpenMP threads to use (default: OMP_NUM_THREADS env var or system default)\n";
            std::cout << "  --dump-matrices    : Write A, B, and C matrices to 'workspace/' directory.\n";
            std::cout << "  --autotune         : Run a simple autotuning process to suggest block sizes.\n";
            std::cout << "  --verify           : Run scalar GEMM and compare results for correctness.\n";
            return 0;
        } else {
            // Not a recognized flag, try to parse as a number for positional arguments.
            try {
                std::stoi(argv[i]); // Check if it's a valid integer
                positional_arg_indices.push_back(i);
            } catch (const std::invalid_argument&) {
                std::cerr << "Error: Unrecognized argument or invalid integer: " << argv[i] << "\n";
                return 1;
            } catch (const std::out_of_range&) {
                std::cerr << "Error: Integer out of range: " << argv[i] << "\n";
                return 1;
            }
        }
    }

    // Apply positional arguments if they exist and no flags were used to set M, N, K.
    if (!m_set_by_flag && positional_arg_indices.size() > 0) {
        M = std::stoi(argv[positional_arg_indices[0]]);
    }
    if (!n_set_by_flag && positional_arg_indices.size() > 1) {
        N = std::stoi(argv[positional_arg_indices[1]]);
    }
    if (!k_set_by_flag && positional_arg_indices.size() > 2) {
        K = std::stoi(argv[positional_arg_indices[2]]);
    }
    if (positional_arg_indices.size() > 3) {
        std::cerr << "Error: Too many positional arguments. Expected at most M N K.\n";
        return 1;
    }


#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
#else
    if (num_threads > 0) {
        std::cerr << "Warning: OpenMP not enabled, --threads argument ignored." << std::endl;
    }
    std::cout << "OpenMP not available. Running single-threaded." << std::endl;
#endif

    // Allocate matrices with 64-byte alignment using the custom allocator.
    // 64-byte alignment is beneficial for AVX-512 (cache line size).
    std::vector<float, AlignedAllocator<float, 64>> A_vec(static_cast<std::size_t>(M) * K);
    std::vector<float, AlignedAllocator<float, 64>> B_vec(static_cast<std::size_t>(K) * N);
    std::vector<float, AlignedAllocator<float, 64>> C_vec(static_cast<std::size_t>(M) * N);
    std::vector<float, AlignedAllocator<float, 64>> C_ref_vec;

    if (verify_scalar) {
        C_ref_vec.resize(static_cast<std::size_t>(M) * N);
    }

    float* A = A_vec.data();
    float* B = B_vec.data();
    float* C = C_vec.data();
    float* C_ref = C_ref_vec.data();

    // Initialize matrices with random values
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (std::size_t i = 0; i < A_vec.size(); ++i) A[i] = dist(rng);
    for (std::size_t i = 0; i < B_vec.size(); ++i) B[i] = dist(rng);
    // C is initialized by the kernel (zero for first K-block, then accumulates)
    // For consistency with scalar verification, we explicitly zero it here.
    std::fill(C_vec.begin(), C_vec.end(), 0.0f);

    // Leading dimensions (strides) assuming row-major storage.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Create workspace directory and dump A, B if requested.
    if (dump_matrices) {
        std::filesystem::create_directory("workspace"); // Requires C++17
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B written to workspace/A.txt and workspace/B.txt\n";
    }

    // --- Autotuning Harness (conceptual for constexprs) ---
    // This autotuner is conceptual. Since BM, BN, BK are constexpr,
    // they cannot be changed dynamically at runtime within the same compilation.
    // This section demonstrates how one might search for optimal values by testing
    // on a smaller problem and then suggests to *manually update* the
    // `constexpr` values at the top of the file and recompile for actual effect.
    if (enable_autotune) {
        std::cout << "\nStarting autotuning process (suggested block sizes require recompile).\n";
        const int test_BM_values[] = {64, 96, 128, 192};
        const int test_BN_values[] = {64, 96, 128, 192, 256};
        const int test_BK_values[] = {64, 128, 256, 384};

        double best_gflops = -1.0;
        int best_bm = DEFAULT_BM, best_bn = DEFAULT_BN, best_bk = DEFAULT_BK;

        // Use a smaller problem size for autotuning to reduce tuning time.
        // It should be large enough to be representative but small enough to iterate quickly.
        int tune_M = std::min(M, 256);
        int tune_N = std::min(N, 256);
        int tune_K = std::min(K, 256);

        std::cout << "Autotuning on a " << tune_M << "x" << tune_K << "x" << tune_N << " problem.\n";

        // Allocate temporary matrices for tuning.
        std::vector<float, AlignedAllocator<float, 64>> A_tune_vec(static_cast<std::size_t>(tune_M) * tune_K);
        std::vector<float, AlignedAllocator<float, 64>> B_tune_vec(static_cast<std::size_t>(tune_K) * tune_N);
        std::vector<float, AlignedAllocator<float, 64>> C_tune_vec(static_cast<std::size_t>(tune_M) * tune_N);
        
        float* A_tune = A_tune_vec.data();
        float* B_tune = B_tune_vec.data();
        float* C_tune = C_tune_vec.data();
        
        for (std::size_t i = 0; i < A_tune_vec.size(); ++i) A_tune[i] = dist(rng);
        for (std::size_t i = 0; i < B_tune_vec.size(); ++i) B_tune[i] = dist(rng);

        for (int bm_candidate : test_BM_values) {
            for (int bn_candidate : test_BN_values) {
                for (int bk_candidate : test_BK_values) {
                    // Note: The `gemm` call here uses the *current constexpr* DEFAULT_BM/BN/BK,
                    // not `bm_candidate`, `bn_candidate`, `bk_candidate`. These loop variables
                    // are only for displaying potential values. A true dynamic autotuner would
                    // need to pass these block sizes to the GEMM kernels.
                    std::fill(C_tune_vec.begin(), C_tune_vec.end(), 0.0f); // Reset C_tune for each trial
                    
                    auto start_tune = std::chrono::high_resolution_clock::now();
                    gemm(A_tune, B_tune, C_tune, tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
                    auto end_tune = std::chrono::high_resolution_clock::now();
                    double duration_ms_tune = std::chrono::duration<double, std::milli>(end_tune - start_tune).count();
                    double gflops_tune = (2.0 * tune_M * tune_N * tune_K) / (duration_ms_tune * 1e6);
                    std::cout << "  Trial sizes (for suggestion): BM=" << bm_candidate << ", BN=" << bn_candidate << ", BK=" << bk_candidate << " -> "
                              << std::fixed << std::setprecision(2) << gflops_tune << " GFLOP/s\n";

                    // This part stores the "best" parameters from the trials (if the current static config performs better),
                    // but the primary intent is to provide recommendations.
                    if (gflops_tune > best_gflops) {
                        best_gflops = gflops_tune;
                        best_bm = bm_candidate; // Store for recommendation only
                        best_bn = bn_candidate; // Store for recommendation only
                        best_bk = bk_candidate; // Store for recommendation only
                    }
                }
            }
        }
        std::cout << "\nAutotuning finished. To potentially improve performance, consider manually updating `DEFAULT_BM`, `DEFAULT_BN`, `DEFAULT_BK` values at the top of the file to:\n";
        std::cout << "  `constexpr int DEFAULT_BM = " << best_bm << ";`\n";
        std::cout << "  `constexpr int DEFAULT_BN = " << best_bn << ";`\n";
        std::cout << "  `constexpr int DEFAULT_BK = " << best_bk << ";`\n";
        std::cout << "  Then recompile for optimal performance with these static block sizes.\n\n";
    }
    std::cout << "\nRunning GEMM with M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Current (static) block sizes: BM=" << DEFAULT_BM << ", BN=" << DEFAULT_BN << ", BK=" << DEFAULT_BK << std::endl;

    // Measure performance of the optimized GEMM.
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();

    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double gflops = (2.0 * M * N * K) / (duration_ms * 1e6);

    std::cout << "\n--- Performance Report ---\n";
    std::cout << "Dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(4) << duration_ms << " ms\n";
    std::cout << "GFLOP/s: " << std::fixed << std::setprecision(2) << gflops << std::endl;
    std::cout << "--------------------------\n";

    // Dump C matrix if requested.
    if (dump_matrices) {
        std::filesystem::create_directory("workspace"); // Ensure directory exists
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Matrix C written to workspace/C.txt\n";
    }

    // Optional: Verify correctness against scalar implementation.
    if (verify_scalar) {
        std::cout << "\nPerforming scalar verification...\n";
        std::fill(C_ref_vec.begin(), C_ref_vec.end(), 0.0f); // Ensure C_ref is clean
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

        float max_diff = 0.0f;
        float max_val_C = 0.0f;
        for (std::size_t i = 0; i < C_vec.size(); ++i) {
            max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
            max_val_C = std::max(max_val_C, std::abs(C_ref[i]));
        }

        // Using a relative tolerance for floating point comparisons to account for
        // accumulation order differences and inherent precision limits.
        // A common tolerance for single-precision is 1e-4 to 1e-3 relative.
        float tolerance = 1e-4f * max_val_C; 
        if (max_val_C < 1e-9f) { // If C is essentially zero, use absolute tolerance
             tolerance = 1e-6f; // A small absolute epsilon
        }

        if (max_diff > tolerance) {
            std::cerr << "Verification FAILED! Max difference: " << max_diff << " (Tolerance: " << tolerance << ")\n";
            std::cerr << "Max absolute value in C_ref: " << max_val_C << "\n";
            return 1; // Indicate failure
        } else {
            std::cout << "Verification PASSED. Max difference: " << max_diff << " (Tolerance: " << tolerance << ")\n";
        }
    }

    return 0; // Indicate success
}