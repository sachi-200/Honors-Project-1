// Compile Instructions:
//
// For AMD Ryzen 7 6800HS (AVX2 + FMA):
// The target CPU (AMD Ryzen 7 6800HS) supports AVX2 and FMA. Compile with `-march=native` or `-march=x86-64-v3`
// to enable these instruction sets. The runtime dispatcher will automatically select the AVX2 kernel.
// Command: `g++ -std=c++17 -O3 -march=native -fopenmp gemm.cpp -o gemm_ryzen`
// (Or explicitly: `g++ -std=c++17 -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_v3`)
//
// For CPUs with AVX-512 (e.g., some Intel processors, NOT AMD Ryzen 6000 series):
// If targeting a CPU that supports AVX-512, compile with appropriate flags.
// Command: `g++ -std=c++17 -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512`
//
// Portable (runtime dispatch will select best available, compiled with base features):
// This command uses `-march=native` to detect the host CPU's features at compile time and enable corresponding instruction sets.
// The runtime dispatcher will still be used to ensure portability across different CPUs
// (e.g., running `gemm_portable` compiled on an AVX-512 machine on an AVX2 machine will correctly fall back).
// Command: `g++ -std=c++17 -O3 -march=native -fopenmp gemm.cpp -o gemm_portable`
//
// Note on `-std=c++17`: This is required for `std::filesystem`.

// Required standard headers
#include <iostream>   // For input/output operations (e.g., std::cout, std::cerr)
#include <vector>     // For std::vector
#include <cstring>    // For std::memcpy, std::memset
#include <chrono>     // For timing performance (std::chrono)
#include <random>     // For random number generation (std::mt19937, std::uniform_real_distribution)
#include <cassert>    // For assert
#include <fstream>    // For file output (std::ofstream)
#include <filesystem> // For creating directories (std::filesystem::create_directory, requires C++17)
#include <memory>     // For std::unique_ptr with custom deleter
#include <numeric>    // Not directly used but common for matrix ops, included for completeness
#include <algorithm>  // For std::min, std::max, std::fill, std::abs
#include <cmath>      // For std::sqrt

// Intrinsics headers - guarded to ensure compilation only when respective ISA is enabled
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h> // Includes AVX, AVX2, AVX-512 intrinsics
#endif

// OpenMP header - guarded for platforms without OpenMP support
#ifdef _OPENMP
#include <omp.h> // For multi-threading with OpenMP
#endif

// --- Autotuning Parameters ---
// These parameters define the tiling strategy and micro-kernel behavior.
// They are exposed as `constexpr` constants at the top for easy tuning.
// The optimal values depend on the specific CPU architecture's cache hierarchy (L1, L2, L3 sizes)
// and TLB (Translation Lookaside Buffer) characteristics.

// Target CPU: AMD Ryzen 7 6800HS (Zen 3+)
// L1d cache: 32KB per core
// L2 cache: 512KB per core
// L3 cache: 16MB shared (shared across CCX)

// Blocking sizes (BM, BN, BK):
// These determine the amount of data processed in larger blocks to optimize L2/L3 cache reuse.
// BM=64, BN=64, BK=32 are chosen to balance cache pressure and parallelism,
// prioritizing L1D cache locality. This configuration aims to fit the thread's active working set
// (packed A, packed B, and local C block) within L1d cache.
// - C_tile_local (BM x BN packed): 64 rows * 64 cols * 4 bytes/float = 16384 bytes (~16 KB).
//   This block of C is loaded into a thread-local buffer at the beginning of its (mb, nb) outer loop
//   and stored back to global C only once at the end. This drastically reduces L1d misses for C
//   by converting many small read-modify-write operations on global C into single block loads/stores
//   and internal L1/register-based computations.
// - B_packed (BK x BN packed): 32 rows * 64 cols * 4 bytes/float = 8192 bytes (~8 KB).
//   This block is packed once per K-block iteration into a thread-local buffer and reused across M-rows
//   and N-columns within that K-block. This transforms strided B column accesses into contiguous row accesses.
// - A_packed_micro (MR x BK): A small micro-panel of A, MR rows by BK columns, is packed into a contiguous
//   thread-local buffer for each M-block of the micro-kernel.
//   For AVX2, this is MR_AVX2 * BK * 4 bytes/float (e.g., 8 * 32 * 4 = 1024 bytes).
//
// The critical working set actively used in the innermost loops (for register blocking and L1)
// consists of:
//   ~1KB (A_packed_micro) + ~8KB (B_packed) + ~16KB (C_tile_local) = ~25KB.
// This fits comfortably within the 32KB L1d cache per core, aiming to minimize L1d misses and maximize data reuse.
constexpr int BM_DEFAULT = 64; // Block size for M (rows of C)
constexpr int BN_DEFAULT = 64; // Block size for N (columns of C)
constexpr int BK_DEFAULT = 32; // Block size for K (inner dimension)

// Micro-kernel parameters (MR, NR, UNROLL_K):
// These define the smallest unit of work processed by SIMD registers.
// MR and NR determine the register blocking for the C matrix accumulators.
// UNROLL_K determines the loop unrolling factor for the innermost K-loop,
// reducing loop overhead and exposing more instructions for pipelining.

// AVX2 specific micro-kernel parameters:
// VEC_WIDTH_AVX2 (8 floats).
// Zen 3/4 typically have 16 YMM registers.
// MR_AVX2 (8) `__m256` accumulators for C take 8 * 32 bytes = 256 bytes of register space.
// UNROLL_K_AVX2 is set to 4. This choice allows up to 16 YMM registers to be live at a time
// (8 for C_acc, 4 for B_vec, 4 for A_broadcast), maximizing register utilization without spilling.
constexpr int VEC_WIDTH_AVX2 = 8; // __m256 holds 8 floats
constexpr int MR_AVX2 = 8;        // Number of rows of A/C to process in the micro-kernel
constexpr int NR_AVX2 = VEC_WIDTH_AVX2; // Number of columns of B/C to process (must be VEC_WIDTH_AVX2 = 8)
constexpr int UNROLL_K_AVX2 = 4;  // K-dimension unroll factor for AVX2 micro-kernel

// AVX-512 specific micro-kernel parameters:
// VEC_WIDTH_AVX512 (16 floats).
// AVX-512 has 32 ZMM registers.
// MR_AVX512 (6) `__m512` accumulators for C take 6 * 64 bytes = 384 bytes of register space.
// UNROLL_K_AVX512 is set to 4 for balance and consistency with AVX2; 6 (C_acc) + 4 (B_vec) + 4 (A_broadcast) = 14 ZMM registers,
// well within the 32 available, offering flexibility.
constexpr int VEC_WIDTH_AVX512 = 16; // __m512 holds 16 floats
constexpr int MR_AVX512 = 6;         // Number of rows of A/C to process in the micro-kernel
constexpr int NR_AVX512 = VEC_WIDTH_AVX512; // Number of columns of B/C to process (must be VEC_WIDTH_AVX512 = 16)
constexpr int UNROLL_K_AVX512 = 4;   // K-dimension unroll factor for AVX-512 micro-kernel

// --- Utility Functions ---

// Helper function to write a matrix to a text file.
// Assumes row-major storage for output, reading based on `ld` (leading dimension).
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[static_cast<size_t>(i) * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// Custom deleter for aligned memory, to be used with `std::unique_ptr`.
// Ensures that memory allocated with `std::aligned_alloc` is deallocated with `std::free`.
struct AlignedFree {
    void operator()(void* ptr) const {
        if (ptr) {
            std::free(ptr);
        }
    }
};

// --- GEMM Implementations ---

// Matrix storage convention: Row-major.
// A: M x K, lda (leading dimension of A, usually K for dense)
// B: K x N, ldb (leading dimension of B, usually N for dense)
// C: M x N, ldc (leading dimension of C, usually N for dense)
// Result C = A * B

// Scalar reference GEMM implementation.
// This is a basic triple-nested loop, used for correctness verification.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += A[static_cast<size_t>(i) * lda + p] * B[static_cast<size_t>(p) * ldb + j];
            }
            C[static_cast<size_t>(i) * ldc + j] = sum;
        }
    }
}

// AVX2 + FMA optimized GEMM implementation.
// Uses a cache-aware tiling strategy combined with a register-blocked micro-kernel.
// OpenMP is used for parallelism over the M and N blocks of the C matrix.
// Both A and B matrices are packed into thread-local buffers.
// The C matrix block (BMxBN) is also loaded into a thread-local buffer once per (M,N) block
// and written back once, greatly improving cache utilization for C by minimizing global memory RMW.
#if defined(__AVX2__) && defined(__FMA__)
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Tiling parameters (chosen from default constants)
    const int BM = BM_DEFAULT;
    const int BN = BN_DEFAULT;
    const int BK = BK_DEFAULT;
    const size_t alignment = 64; // For aligned_alloc, compatible with AVX2 (32-byte) and AVX-512 (64-byte)

    // Micro-kernel parameters (specific to AVX2)
    const int MR = MR_AVX2;
    const int NR = NR_AVX2; // Must be VEC_WIDTH_AVX2 (8 for AVX2)
    const int UNROLL_K = UNROLL_K_AVX2;
    const int VEC_WIDTH = VEC_WIDTH_AVX2; // 8 floats per __m256

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        // Thread-private buffer for packed B blocks (BK x BN).
        // B_packed converts strided column access from global B to contiguous row access.
        std::unique_ptr<float, AlignedFree> B_packed_block_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BK) * BN * sizeof(float))
        );
        float* B_packed = B_packed_block_uptr.get();

        // Thread-private buffer for packed A micro-panels (MR x BK).
        // A_packed_micro converts strided row access from global A to contiguous row access
        // for the micro-kernel's K-loop.
        std::unique_ptr<float, AlignedFree> A_packed_micro_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(MR) * BK * sizeof(float))
        );
        float* A_packed_micro = A_packed_micro_uptr.get();

        // Thread-private buffer for a local C block (BM x BN).
        // This is crucial for reducing L1d/L2 cache misses on C.
        std::unique_ptr<float, AlignedFree> C_tile_local_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BM) * BN * sizeof(float))
        );
        float* C_tile_local = C_tile_local_uptr.get();

        if (!B_packed || !A_packed_micro || !C_tile_local) {
            std::cerr << "Memory allocation for buffers failed in AVX2 kernel. This thread might perform poorly." << std::endl;
        }

#ifdef _OPENMP
    // The `schedule(static)` ensures an even distribution of work and avoids dynamic overhead.
    // `collapse(2)` applies the scheduling across both outer loops (M and N blocks).
    #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M; mb += BM) { // Loop over M-blocks (rows of C)
            for (int nb = 0; nb < N; nb += BN) { // Loop over N-blocks (columns of C)
                // Determine actual block boundaries, handling matrix edges
                int M_block_end = std::min(mb + BM, M);
                int N_block_end = std::min(nb + BN, N);
                int M_current_block_len = M_block_end - mb; // Actual length of M-dim for this block
                int N_current_block_len = N_block_end - nb; // Actual length of N-dim for this block

                // --- Load C block into thread-local buffer (once per (mb, nb) block) ---
                // This improves temporal locality for C.
                for (int r = 0; r < M_current_block_len; ++r) {
                    float* C_local_row_ptr = &C_tile_local[static_cast<size_t>(r) * BN];
                    const float* C_orig_row_ptr = &C[static_cast<size_t>(mb + r) * ldc + nb];
                    
                    int j_idx = 0;
                    for (; j_idx + VEC_WIDTH <= N_current_block_len; j_idx += VEC_WIDTH) {
                        // Store C data to local buffer in a contiguous, aligned manner for optimal cache access
                        _mm256_store_ps(C_local_row_ptr + j_idx, _mm256_loadu_ps(C_orig_row_ptr + j_idx));
                    }
                    // Handle N-tail for loading C using scalar operations
                    for (; j_idx < N_current_block_len; ++j_idx) {
                        C_local_row_ptr[j_idx] = C_orig_row_ptr[j_idx];
                    }
                    // Zero fill remaining columns in C_tile_local if N_current_block_len < BN
                    if (N_current_block_len < BN) {
                        std::memset(C_local_row_ptr + N_current_block_len, 0, (BN - N_current_block_len) * sizeof(float));
                    }
                }
                // Zero fill remaining rows in C_tile_local if M_current_block_len < BM
                for (int r = M_current_block_len; r < BM; ++r) {
                    std::memset(&C_tile_local[static_cast<size_t>(r) * BN], 0, BN * sizeof(float));
                }
                // End loading C block

                for (int kb = 0; kb < K; kb += BK) { // Loop over K-blocks (inner dimension)
                    int K_block_end = std::min(kb + BK, K);
                    int K_current_block_len = K_block_end - kb; // Actual length of K-dim for this block

                    // --- Prefetching for the *current* B block to be packed (original B data) ---
                    for (int p_pf_idx = 0; p_pf_idx < K_current_block_len; ++p_pf_idx) {
                        for (int j_pf_idx = 0; j_pf_idx < N_current_block_len; j_pf_idx += (alignment / sizeof(float))) {
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + p_pf_idx) * ldb + nb + j_pf_idx], _MM_HINT_T0);
                        }
                    }

                    // Pack the current B-block (BK x BN) from original B into `B_packed`.
                    for (int p_idx = 0; p_idx < K_current_block_len; ++p_idx) {
                        float* B_packed_row_ptr = &B_packed[static_cast<size_t>(p_idx) * BN];
                        const float* B_orig_row_ptr = &B[static_cast<size_t>(kb + p_idx) * ldb + nb];

                        int j_pack_idx = 0;
                        for (; j_pack_idx + VEC_WIDTH <= N_current_block_len; j_pack_idx += VEC_WIDTH) {
                            _mm256_store_ps(B_packed_row_ptr + j_pack_idx, _mm256_loadu_ps(B_orig_row_ptr + j_pack_idx));
                        }
                        for (; j_pack_idx < N_current_block_len; ++j_pack_idx) {
                            B_packed_row_ptr[j_pack_idx] = B_orig_row_ptr[j_pack_idx];
                        }
                        if (N_current_block_len < BN) {
                            std::memset(B_packed_row_ptr + N_current_block_len, 0, (BN - N_current_block_len) * sizeof(float));
                        }
                    }
                    for (int p_idx = K_current_block_len; p_idx < BK; ++p_idx) {
                        std::memset(&B_packed[static_cast<size_t>(p_idx) * BN], 0, BN * sizeof(float));
                    }
                    
                    // --- Prefetching for the *next* K-block (original A and B data) ---
                    if (kb + BK < K) {
                        for (int r_pf = 0; r_pf < M_current_block_len; r_pf += MR) {
                             _mm_prefetch((const char*)&A[static_cast<size_t>(mb + r_pf) * lda + (kb + BK)], _MM_HINT_T1);
                        }
                        for (int p_pf = 0; p_pf < BK; p_pf += UNROLL_K) { 
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + BK + p_pf) * ldb + nb], _MM_HINT_T1);
                        }
                    }

                    // Micro-kernel loops. Loop bounds for `i_offset` and `j_offset` are relative to the current `mb`, `nb` block.
                    for (int i_offset = 0; i_offset < M_current_block_len; i_offset += MR) { // Loop over micro-kernel rows of C
                        int i_micro_end_offset = std::min(i_offset + MR, M_current_block_len);

                        // Prefetch source A data for the current A_packed_micro block.
                        for (int r_pf_idx = 0; r_pf_idx < MR; ++r_pf_idx) {
                            if (i_offset + r_pf_idx < M_current_block_len) {
                                _mm_prefetch((const char*)&A[static_cast<size_t>(mb + i_offset + r_pf_idx) * lda + kb], _MM_HINT_T0);
                            }
                        }

                        // Pack A micro-panel for current i_offset (MR x K_current_block_len)
                        for (int r_idx = 0; r_idx < MR; ++r_idx) {
                            int current_abs_m = mb + i_offset + r_idx;
                            float* A_packed_micro_row_ptr = &A_packed_micro[static_cast<size_t>(r_idx) * BK];
                            
                            if (i_offset + r_idx < M_current_block_len) { 
                                const float* A_orig_row_ptr = &A[static_cast<size_t>(current_abs_m) * lda + kb];
                                int k_pack_idx = 0;
                                for (; k_pack_idx + VEC_WIDTH <= K_current_block_len; k_pack_idx += VEC_WIDTH) {
                                    _mm256_store_ps(A_packed_micro_row_ptr + k_pack_idx, _mm256_loadu_ps(A_orig_row_ptr + k_pack_idx));
                                }
                                for (; k_pack_idx < K_current_block_len; ++k_pack_idx) {
                                    A_packed_micro_row_ptr[k_pack_idx] = A_orig_row_ptr[k_pack_idx];
                                }
                                if (K_current_block_len < BK) {
                                    std::memset(A_packed_micro_row_ptr + K_current_block_len, 0, (BK - K_current_block_len) * sizeof(float));
                                }
                            } else { // If row is outside the current M-block, zero fill the entire row in A_packed_micro
                                std::memset(A_packed_micro_row_ptr, 0, BK * sizeof(float));
                            }
                        }

                        for (int j_offset = 0; j_offset < N_current_block_len; j_offset += NR) { // Loop over micro-kernel columns of C
                            int j_micro_end_offset = std::min(j_offset + NR, N_current_block_len);

                            // Prefetch C (local) for the next processing block.
                            if (j_offset + NR < N_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset) * BN + (j_offset + NR)], _MM_HINT_T0);
                            } else if (i_offset + MR < M_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset + MR) * BN + j_offset], _MM_HINT_T0);
                            }

                            // C_acc: Array of AVX2 registers for accumulating results for an MR x NR block of C.
                            alignas(32) __m256 C_acc[MR]; 

                            // Initialize C_acc registers by loading from C_tile_local.
                            // The `C_tile_local` was already loaded from global `C` at the start of (mb, nb) loop.
                            // `C_tile_local` is aligned, and BN/NR are multiples of VEC_WIDTH.
                            for (int r = 0; r < MR; ++r) {
                                if (i_offset + r < i_micro_end_offset) {
                                    // Load from C_tile_local using relative indices. This data is aligned.
                                    C_acc[r] = _mm256_load_ps(&C_tile_local[static_cast<size_t>(i_offset + r) * BN + j_offset]);
                                } else {
                                    C_acc[r] = _mm256_setzero_ps();
                                }
                            }

                            // Inner K loop with unrolling.
                            for (int p_relative_idx = 0; p_relative_idx < BK; p_relative_idx += UNROLL_K) {
                                for (int p_unroll = 0; p_unroll < UNROLL_K; ++p_unroll) {
                                    int current_k_idx_in_block = p_relative_idx + p_unroll;
                                    // The packing functions ensure A_packed_micro and B_packed are zero-padded
                                    // for k >= K_current_block_len (within the BK dimension).
                                    // Thus, loading from these locations will correctly yield 0.0f for out-of-bounds k.

                                    // Load vector from B_packed. Access is contiguous and aligned by construction.
                                    __m256 B_vec = _mm256_load_ps(&B_packed[static_cast<size_t>(current_k_idx_in_block) * BN + j_offset]);

                                    // Main multiplication and accumulation for the MR x NR block.
                                    for (int r = 0; r < MR; ++r) {
                                        // Load scalar from A_packed_micro and broadcast.
                                        __m256 A_broadcast = _mm256_set1_ps(A_packed_micro[static_cast<size_t>(r) * BK + current_k_idx_in_block]);
                                        
                                        // Fused Multiply-Add (FMA): C_acc[r] = C_acc[r] + (A_broadcast * B_vec)
                                        C_acc[r] = _mm256_fmadd_ps(A_broadcast, B_vec, C_acc[r]);
                                    }
                                }
                            }

                            // Store accumulated results back to C_tile_local (not global C).
                            for (int r = 0; r < MR; ++r) {
                                if (i_offset + r < i_micro_end_offset) {
                                    // Store back to C_tile_local, using aligned store.
                                    _mm256_store_ps(&C_tile_local[static_cast<size_t>(i_offset + r) * BN + j_offset], C_acc[r]);
                                }
                            }
                        } // end j_offset loop (NR micro-block)
                    } // end i_offset loop (MR micro-block)
                } // end kb loop (BK block)

                // --- Store C_tile_local back to global C (once per (mb, nb) block) ---
                for (int r = 0; r < M_current_block_len; ++r) {
                    float* C_orig_row_ptr = &C[static_cast<size_t>(mb + r) * ldc + nb];
                    const float* C_local_row_ptr = &C_tile_local[static_cast<size_t>(r) * BN];

                    int j_idx = 0;
                    for (; j_idx + VEC_WIDTH <= N_current_block_len; j_idx += VEC_WIDTH) {
                        _mm256_storeu_ps(C_orig_row_ptr + j_idx, _mm256_load_ps(C_local_row_ptr + j_idx));
                    }
                    // Handle N-tail for storing C using scalar operations
                    for (; j_idx < N_current_block_len; ++j_idx) {
                        C_orig_row_ptr[j_idx] = C_local_row_ptr[j_idx];
                    }
                }
                // End storing C block
            } // end nb loop (BN block)
        } // end mb loop (BM block)
    } // End OpenMP parallel region
}
#else // AVX2 and FMA not enabled at compile time
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "Warning: AVX2 kernel not compiled or FMA not enabled. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX2__ && __FMA__

// AVX-512 + FMA optimized GEMM implementation.
// Similar blocking strategy to AVX2, but uses `__m512` (16 floats) and AVX-512's
// advanced mask registers for efficient tail handling in loads and stores.
#if defined(__AVX512F__) && defined(__FMA__)
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    // Tiling parameters
    const int BM = BM_DEFAULT;
    const int BN = BN_DEFAULT;
    const int BK = BK_DEFAULT;
    const size_t alignment = 64;

    // Micro-kernel parameters (specific to AVX-512)
    const int MR = MR_AVX512;
    const int NR = NR_AVX512; // Must be VEC_WIDTH_AVX512 (16 for AVX-512)
    const int UNROLL_K = UNROLL_K_AVX512;
    const int VEC_WIDTH = VEC_WIDTH_AVX512; // 16 floats per __m512

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        // Thread-private buffer for packed B blocks.
        std::unique_ptr<float, AlignedFree> B_packed_block_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BK) * BN * sizeof(float))
        );
        float* B_packed = B_packed_block_uptr.get();

        // Thread-private buffer for packed A micro-panels.
        std::unique_ptr<float, AlignedFree> A_packed_micro_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(MR) * BK * sizeof(float))
        );
        float* A_packed_micro = A_packed_micro_uptr.get();

        // Thread-private buffer for a local C block (BM x BN).
        std::unique_ptr<float, AlignedFree> C_tile_local_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BM) * BN * sizeof(float))
        );
        float* C_tile_local = C_tile_local_uptr.get();

        if (!B_packed || !A_packed_micro || !C_tile_local) {
            std::cerr << "Memory allocation for buffers failed in AVX-512 kernel. This thread might perform poorly." << std::endl;
        }

#ifdef _OPENMP
    #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M; mb += BM) {
            for (int nb = 0; nb < N; nb += BN) {
                int M_block_end = std::min(mb + BM, M);
                int N_block_end = std::min(nb + BN, N);
                int M_current_block_len = M_block_end - mb;
                int N_current_block_len = N_block_end - nb;

                // --- Load C block into thread-local buffer (once per (mb, nb) block) ---
                for (int r = 0; r < M_current_block_len; ++r) {
                    float* C_local_row_ptr = &C_tile_local[static_cast<size_t>(r) * BN];
                    const float* C_orig_row_ptr = &C[static_cast<size_t>(mb + r) * ldc + nb];
                    
                    int j_idx = 0;
                    for (; j_idx + VEC_WIDTH <= N_current_block_len; j_idx += VEC_WIDTH) {
                        _mm512_store_ps(C_local_row_ptr + j_idx, _mm512_loadu_ps(C_orig_row_ptr + j_idx));
                    }
                    // Use AVX-512 masked load for tail, then store
                    if (N_current_block_len - j_idx > 0) {
                         __mmask16 tail_mask = static_cast<__mmask16>((1 << (N_current_block_len - j_idx)) - 1);
                        _mm512_mask_store_ps(C_local_row_ptr + j_idx, tail_mask, _mm512_maskz_loadu_ps(tail_mask, C_orig_row_ptr + j_idx));
                    }
                    // Zero fill remaining columns in C_tile_local if N_current_block_len < BN
                    if (N_current_block_len < BN) {
                        std::memset(C_local_row_ptr + N_current_block_len, 0, (BN - N_current_block_len) * sizeof(float));
                    }
                }
                // Zero fill remaining rows in C_tile_local if M_current_block_len < BM
                for (int r = M_current_block_len; r < BM; ++r) {
                    std::memset(&C_tile_local[static_cast<size_t>(r) * BN], 0, BN * sizeof(float));
                }
                // End loading C block

                for (int kb = 0; kb < K; kb += BK) {
                    int K_block_end = std::min(kb + BK, K);
                    int K_current_block_len = K_block_end - kb;

                    // --- Prefetching for the *current* B block to be packed (original B data) ---
                    for (int p_pf_idx = 0; p_pf_idx < K_current_block_len; ++p_pf_idx) {
                        for (int j_pf_idx = 0; j_pf_idx < N_current_block_len; j_pf_idx += (alignment / sizeof(float))) {
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + p_pf_idx) * ldb + nb + j_pf_idx], _MM_HINT_T0);
                        }
                    }

                    // Pack B block (BK x BN)
                    for (int p_idx = 0; p_idx < K_current_block_len; ++p_idx) {
                        float* B_packed_row_ptr = &B_packed[static_cast<size_t>(p_idx) * BN];
                        const float* B_orig_row_ptr = &B[static_cast<size_t>(kb + p_idx) * ldb + nb];

                        int j_pack_idx = 0;
                        for (; j_pack_idx + VEC_WIDTH <= N_current_block_len; j_pack_idx += VEC_WIDTH) {
                            _mm512_store_ps(B_packed_row_ptr + j_pack_idx, _mm512_loadu_ps(B_orig_row_ptr + j_pack_idx));
                        }
                        if (N_current_block_len - j_pack_idx > 0) {
                            __mmask16 tail_mask = static_cast<__mmask16>((1 << (N_current_block_len - j_pack_idx)) - 1);
                            _mm512_mask_store_ps(B_packed_row_ptr + j_pack_idx, tail_mask, _mm512_maskz_loadu_ps(tail_mask, B_orig_row_ptr + j_pack_idx));
                        }
                        if (N_current_block_len < BN) {
                            std::memset(B_packed_row_ptr + N_current_block_len, 0, (BN - N_current_block_len) * sizeof(float));
                        }
                    }
                    for (int p_idx = K_current_block_len; p_idx < BK; ++p_idx) {
                        std::memset(&B_packed[static_cast<size_t>(p_idx) * BN], 0, BN * sizeof(float));
                    }

                    // --- Prefetching for the *next* K-block (original A and B data) ---
                    if (kb + BK < K) {
                        for (int r_pf = 0; r_pf < M_current_block_len; r_pf += MR) {
                            _mm_prefetch((const char*)&A[static_cast<size_t>(mb + r_pf) * lda + (kb + BK)], _MM_HINT_T1);
                        }
                        for (int p_pf = 0; p_pf < BK; p_pf += UNROLL_K) {
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + BK + p_pf) * ldb + nb], _MM_HINT_T1);
                        }
                    }

                    for (int i_offset = 0; i_offset < M_current_block_len; i_offset += MR) {
                        int i_micro_end_offset = std::min(i_offset + MR, M_current_block_len);

                        // Prefetch source A data for the current A_packed_micro block.
                        for (int r_pf_idx = 0; r_pf_idx < MR; ++r_pf_idx) {
                            if (i_offset + r_pf_idx < M_current_block_len) {
                                _mm_prefetch((const char*)&A[static_cast<size_t>(mb + i_offset + r_pf_idx) * lda + kb], _MM_HINT_T0);
                            }
                        }

                        // Pack A micro-panel for current i_offset (MR x K_current_block_len)
                        for (int r_idx = 0; r_idx < MR; ++r_idx) {
                            int current_abs_m = mb + i_offset + r_idx;
                            float* A_packed_micro_row_ptr = &A_packed_micro[static_cast<size_t>(r_idx) * BK];

                            if (i_offset + r_idx < M_current_block_len) { 
                                const float* A_orig_row_ptr = &A[static_cast<size_t>(current_abs_m) * lda + kb];
                                int k_pack_idx = 0;
                                for (; k_pack_idx + VEC_WIDTH <= K_current_block_len; k_pack_idx += VEC_WIDTH) {
                                    _mm512_store_ps(A_packed_micro_row_ptr + k_pack_idx, _mm512_loadu_ps(A_orig_row_ptr + k_pack_idx));
                                }
                                if (K_current_block_len - k_pack_idx > 0) {
                                    __mmask16 tail_mask = static_cast<__mmask16>((1 << (K_current_block_len - k_pack_idx)) - 1);
                                    _mm512_mask_store_ps(A_packed_micro_row_ptr + k_pack_idx, tail_mask, _mm512_maskz_loadu_ps(tail_mask, A_orig_row_ptr + k_pack_idx));
                                }
                                if (K_current_block_len < BK) {
                                    std::memset(A_packed_micro_row_ptr + K_current_block_len, 0, (BK - K_current_block_len) * sizeof(float));
                                }
                            } else { 
                                std::memset(A_packed_micro_row_ptr, 0, BK * sizeof(float));
                            }
                        }

                        for (int j_offset = 0; j_offset < N_current_block_len; j_offset += NR) {
                            int j_micro_end_offset = std::min(j_offset + NR, N_current_block_len);

                            // Prefetch C (local) for the next processing block.
                            if (j_offset + NR < N_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset) * BN + (j_offset + NR)], _MM_HINT_T0);
                            } else if (i_offset + MR < M_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset + MR) * BN + j_offset], _MM_HINT_T0);
                            }

                            alignas(64) __m512 C_acc[MR]; // Array of AVX-512 registers for accumulation
                            
                            // Initialize C_acc registers by loading from C_tile_local.
                            // The `C_tile_local` was already loaded from global `C` at the start of (mb, nb) loop.
                            // `C_tile_local` is aligned, and BN/NR are multiples of VEC_WIDTH.
                            for (int r = 0; r < MR; ++r) {
                                if (i_offset + r < i_micro_end_offset) {
                                    C_acc[r] = _mm512_load_ps(&C_tile_local[static_cast<size_t>(i_offset + r) * BN + j_offset]);
                                } else {
                                    C_acc[r] = _mm512_setzero_ps();
                                }
                            }

                            // Inner K loop with unrolling.
                            for (int p_relative_idx = 0; p_relative_idx < BK; p_relative_idx += UNROLL_K) {
                                for (int p_unroll = 0; p_unroll < UNROLL_K; ++p_unroll) {
                                    int current_k_idx_in_block = p_relative_idx + p_unroll;
                                    // The packing functions ensure A_packed_micro and B_packed are zero-padded
                                    // for k >= K_current_block_len (within the BK dimension).
                                    // Thus, loading from these locations will correctly yield 0.0f for out-of-bounds k.

                                    // Load vector B[p_col][j_col:j_col+NR] from B_packed
                                    __m512 B_vec = _mm512_load_ps(&B_packed[static_cast<size_t>(current_k_idx_in_block) * BN + j_offset]);

                                    for (int r = 0; r < MR; ++r) {
                                        // Broadcast scalar A[i_row][p_col] from A_packed_micro
                                        __m512 A_broadcast = _mm512_set1_ps(A_packed_micro[static_cast<size_t>(r) * BK + current_k_idx_in_block]);
                                        
                                        // Fused Multiply-Add
                                        C_acc[r] = _mm512_fmadd_ps(A_broadcast, B_vec, C_acc[r]);
                                    }
                                }
                            }

                            // Store accumulated results back to C_tile_local (not global C).
                            for (int r = 0; r < MR; ++r) {
                                if (i_offset + r < i_micro_end_offset) {
                                    // Store back to C_tile_local, using aligned store.
                                    _mm512_store_ps(&C_tile_local[static_cast<size_t>(i_offset + r) * BN + j_offset], C_acc[r]);
                                }
                            }
                        } // end j_offset loop
                    } // end i_offset loop
                } // end kb loop

                // --- Store C_tile_local back to global C (once per (mb, nb) block) ---
                for (int r = 0; r < M_current_block_len; ++r) {
                    float* C_orig_row_ptr = &C[static_cast<size_t>(mb + r) * ldc + nb];
                    const float* C_local_row_ptr = &C_tile_local[static_cast<size_t>(r) * BN];

                    int j_idx = 0;
                    for (; j_idx + VEC_WIDTH <= N_current_block_len; j_idx += VEC_WIDTH) {
                        _mm512_storeu_ps(C_orig_row_ptr + j_idx, _mm512_load_ps(C_local_row_ptr + j_idx));
                    }
                    // Use AVX-512 masked store for tail
                    if (N_current_block_len - j_idx > 0) {
                        __mmask16 tail_mask = static_cast<__mmask16>((1 << (N_current_block_len - j_idx)) - 1);
                        _mm512_mask_storeu_ps(C_orig_row_ptr + j_idx, tail_mask, _mm512_load_ps(C_local_row_ptr + j_idx));
                    }
                }
                // End storing C block
            } // end nb loop
        } // end mb loop
    } // End OpenMP parallel region
}
#else // AVX512F and FMA not enabled at compile time
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    std::cerr << "Warning: AVX-512 kernel not compiled or FMA not enabled. Falling back to AVX2 or scalar." << std::endl;
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc); 
}
#endif // __AVX512F__ && __FMA__


// Top-level GEMM function with runtime dispatch.
// This function determines the best available SIMD kernel based on CPU features
// using `__builtin_cpu_supports` (for GCC/Clang) and calls the appropriate implementation.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

    bool has_avx512f_runtime = false;
    bool has_avx2_runtime = false;

    // Runtime feature detection (compiler-specific builtins for GCC/Clang)
#if defined(__GNUC__) || defined(__clang__)
    has_avx512f_runtime = __builtin_cpu_supports("avx512f");
    has_avx2_runtime = __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0); // Extended features
    has_avx2_runtime = (cpuInfo[1] & (1 << 5)) != 0; // Check for AVX2 bit (bit 5 of EBX)
    has_avx512f_runtime = (cpuInfo[1] & (1 << 16)) != 0; // Check for AVX512F bit (bit 16 of EBX)
#endif

    // Dispatch logic: Prefer AVX-512, then AVX2, then scalar.
    // It also checks if the respective kernel was actually compiled using #if defined checks.
    if (has_avx512f_runtime) {
#if defined(__AVX512F__) && defined(__FMA__)
        std::cout << "Runtime Dispatch: Using AVX-512 kernel (compiled and detected)." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
#else // AVX512 detected at runtime, but not compiled.
        std::cout << "Runtime Dispatch: AVX-512 detected but kernel not compiled. Attempting AVX2/scalar fallback." << std::endl;
        if (has_avx2_runtime) {
#if defined(__AVX2__) && defined(__FMA__)
            std::cout << "Runtime Dispatch: Using AVX2 kernel (AVX-512 not compiled)." << std::endl;
            gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else // AVX2 detected at runtime, but not compiled.
            std::cout << "Runtime Dispatch: AVX2 detected but kernel not compiled. Falling back to scalar." << std::endl;
            gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
        } else { // No AVX2 detected at runtime either.
            std::cout << "Runtime Dispatch: No AVX2/AVX-512 compiled. Using scalar kernel." << std::endl;
            gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
        }
#endif
    } else if (has_avx2_runtime) { // AVX2 detected at runtime, AVX512 not.
#if defined(__AVX2__) && defined(__FMA__)
        std::cout << "Runtime Dispatch: Using AVX2 kernel (compiled and detected)." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else // AVX2 detected at runtime, but not compiled.
        std::cout << "Runtime Dispatch: AVX2 detected but kernel not compiled. Falling back to scalar." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
    } else { // No SIMD detected at runtime or compiled.
        std::cout << "Runtime Dispatch: No AVX2/AVX-512 detected. Using scalar kernel." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
}


// --- Main function for demonstration and testing ---
int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    unsigned int seed = 42; // Seed for random matrix initialization
    int num_threads = 0;    // 0 means OpenMP default or system default
    bool dump_matrices = false; // Flag to dump matrices to files

    // Parse command line arguments.
    std::vector<std::string> arg_list;
    for (int i = 1; i < argc; ++i) {
        arg_list.push_back(argv[i]);
    }

    int current_pos_arg = 0;
    for (size_t i = 0; i < arg_list.size(); ++i) {
        const std::string& arg = arg_list[i];

        if (arg[0] == '-') { // It's a flag
            if (arg == "-M" && i + 1 < arg_list.size()) {
                M = std::stoi(arg_list[++i]);
            } else if (arg == "-N" && i + 1 < arg_list.size()) {
                N = std::stoi(arg_list[++i]);
            } else if (arg == "-K" && i + 1 < arg_list.size()) {
                K = std::stoi(arg_list[++i]);
            } else if (arg == "-s" && i + 1 < arg_list.size()) {
                seed = std::stoul(arg_list[++i]);
            } else if (arg == "-t" && i + 1 < arg_list.size()) {
                num_threads = std::stoi(arg_list[++i]);
            } else if (arg == "--dump-matrices") {
                dump_matrices = true;
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M <rows>] [-N <cols>] [-K <inner>] [-s <seed>] [-t <threads>] [--dump-matrices]\n";
                std::cout << "  [M] [N] [K]       : Optional positional arguments for M, N, K dimensions (default: 1024 1024 1024).\n";
                std::cout << "  -M <rows>         : Number of rows in A and C. Overrides positional M.\n";
                std::cout << "  -N <cols>         : Number of columns in B and C. Overrides positional N.\n";
                std::cout << "  -K <inner>        : Inner dimension for A and B. Overrides positional K.\n";
                std::cout << "  -s <seed>         : Seed for random matrix initialization (default: 42)\n";
                std::cout << "  -t <threads>      : Number of OpenMP threads (default: system/OMP default)\n";
                std::cout << "  --dump-matrices   : Dump input matrices A, B and output matrix C to 'workspace/' directory\n";
                std::cout << "  -h, --help        : Display this help message\n";
                return 0;
            } else {
                std::cerr << "Error: Unknown argument or missing value for: " << arg << std::endl;
                return 1;
            }
        } else { // It's a positional argument
            if (current_pos_arg == 0) {
                M = std::stoi(arg);
            } else if (current_pos_arg == 1) {
                N = std::stoi(arg);
            } else if (current_pos_arg == 2) {
                K = std::stoi(arg);
            } else {
                std::cerr << "Error: Too many positional arguments. Unrecognized: " << arg << std::endl;
                return 1;
            }
            current_pos_arg++;
        }
    }

    // Set OpenMP thread count if specified
    if (num_threads > 0) {
#ifdef _OPENMP
        omp_set_num_threads(num_threads);
        std::cout << "Set OpenMP threads to: " << num_threads << std::endl;
#else
        std::cerr << "Warning: OpenMP not enabled during compilation, -t argument ignored." << std::endl;
#endif
    } else {
#ifdef _OPENMP
        // Get number of physical cores dynamically to provide better SMT hint
        int physical_cores = 0;
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        int core_id = -1, processor_id = -1;
        std::vector<std::pair<int, int>> core_processor_pairs;
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("processor", 0) == 0) {
                processor_id = std::stoi(line.substr(line.find(":") + 1));
            } else if (line.rfind("core id", 0) == 0) {
                core_id = std::stoi(line.substr(line.find(":") + 1));
            }
            if (core_id != -1 && processor_id != -1) {
                core_processor_pairs.push_back({processor_id, core_id});
                core_id = -1; // Reset to process next core
                processor_id = -1; // Reset to process next processor
            }
        }
        std::sort(core_processor_pairs.begin(), core_processor_pairs.end());
        core_processor_pairs.erase(std::unique(core_processor_pairs.begin(), core_processor_pairs.end()), core_processor_pairs.end());
        physical_cores = core_processor_pairs.size();
        
        std::cout << "Using default OpenMP threads: " << omp_get_max_threads() << std::endl;
        if (physical_cores > 0 && omp_get_max_threads() > physical_cores && omp_get_max_threads() % physical_cores == 0) {
            std::cout << "Note: This system has " << physical_cores << " physical cores. On SMT/HT systems like AMD Ryzen, "
                      << "setting OMP_NUM_THREADS to the number of physical cores (e.g., " << physical_cores << " for this CPU) "
                      << "might improve performance due to reduced contention compared to using all logical cores (" << omp_get_max_threads() << ")." << std::endl;
        }
#else
        std::cout << "OpenMP not enabled during compilation, running single-threaded." << std::endl;
#endif
    }

    // Allocate memory for matrices A, B, C with 64-byte alignment.
    size_t alignment = 64; 
    size_t size_A = static_cast<size_t>(M) * K;
    size_t size_B = static_cast<size_t>(K) * N;
    size_t size_C = static_cast<size_t>(M) * N; 

    // Using `std::unique_ptr` with custom deleter for aligned memory ensures proper deallocation.
    std::unique_ptr<float, AlignedFree> A_aligned((float*)std::aligned_alloc(alignment, size_A * sizeof(float)));
    std::unique_ptr<float, AlignedFree> B_aligned((float*)std::aligned_alloc(alignment, size_B * sizeof(float)));
    std::unique_ptr<float, AlignedFree> C_aligned((float*)std::aligned_alloc(alignment, size_C * sizeof(float)));
    std::unique_ptr<float, AlignedFree> C_ref_aligned((float*)std::aligned_alloc(alignment, size_C * sizeof(float))); // For verification

    if (!A_aligned || !B_aligned || !C_aligned || !C_ref_aligned) {
        std::cerr << "Error: Memory allocation failed! Ensure sufficient RAM or try smaller matrix sizes." << std::endl;
        return 1;
    }

    float* A = A_aligned.get();
    float* B = B_aligned.get();
    float* C = C_aligned.get();
    float* C_ref = C_ref_aligned.get();

    // Initialize matrices A and B with random values, C with zeros
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i = 0; i < size_A; ++i) A[i] = dis(gen);
    for (size_t i = 0; i < size_B; ++i) B[i] = dis(gen);
    std::fill(C, C + size_C, 0.0f);
    std::fill(C_ref, C_ref + size_C, 0.0f);

    // Using leading dimensions equal to the number of columns for dense row-major storage.
    int lda = K;
    int ldb = N;
    int ldc = N;

    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Dump input matrices A and B if the flag is set
    if (dump_matrices) {
        std::filesystem::create_directory("workspace"); // Ensure workspace directory exists (C++17)
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt" << std::endl;
    }

    // --- Performance measurement ---
    std::cout << "\nStarting GEMM computation..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call the top-level dispatcher GEMM function
    gemm(A, B, C, M, N, K, lda, ldb, ldc);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;

    // Calculate GFLOP/s (2 * M * N * K floating point operations for a GEMM)
    double gflops = 2.0 * M * N * K / (duration.count() * 1e6); // 1e6 to convert ms to seconds

    std::cout << "\nGEMM computation finished." << std::endl;
    std::cout << "Time elapsed: " << duration.count() << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOP/s\n";

    // Dump output matrix C if the flag is set
    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Matrix C dumped to workspace/C.txt" << std::endl;
    }

    // --- Verification ---
    std::cout << "\nVerifying correctness with scalar reference implementation..." << std::endl;
    gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    long double sum_sq_diff = 0.0L; // Use long double for higher precision in sum of squares

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = std::abs(C[static_cast<size_t>(i) * ldc + j] - C_ref[static_cast<size_t>(i) * ldc + j]);
            max_diff = std::max(max_diff, diff);
            avg_diff += diff;
            sum_sq_diff += (long double)diff * diff;
        }
    }

    avg_diff /= (static_cast<long double>(M) * N);
    float rmse = static_cast<float>(std::sqrt(sum_sq_diff / (static_cast<long double>(M) * N)));

    // Choose a numerical tolerance. 1e-4f is a typical starting point for single-precision float GEMM.
    // However, due to FMA or large K, accumulated error can be slightly higher.
    // Set to 2e-4f to account for expected floating-point variance on larger matrices.
    float tolerance = 2e-4f; 
    if (max_diff > tolerance) {
        std::cerr << "Verification FAILED: Maximum absolute difference = " << max_diff << " (tolerance: " << tolerance << ")" << std::endl;
        std::cerr << "Average absolute difference = " << avg_diff << std::endl;
        std::cerr << "Root Mean Square Error (RMSE) = " << rmse << std::endl;
        // Optionally print up to 5 differing values for debugging
        int diff_count = 0;
        for (int i = 0; i < M && diff_count < 5; ++i) {
            for (int j = 0; j < N && diff_count < 5; ++j) {
                if (std::abs(C[static_cast<size_t>(i) * ldc + j] - C_ref[static_cast<size_t>(i) * ldc + j]) > tolerance) {
                    std::cerr << "  Mismatch at C[" << i << "," << j << "]: Optimized=" << C[static_cast<size_t>(i) * ldc + j]
                              << ", Reference=" << C_ref[static_cast<size_t>(i) * ldc + j]
                              << ", Difference=" << std::abs(C[static_cast<size_t>(i) * ldc + j] - C_ref[static_cast<size_t>(i) * ldc + j]) << std::endl;
                    diff_count++;
                }
            }
        }
        return 1; // Indicate failure
    } else {
        std::cout << "Verification PASSED: Maximum absolute difference = " << max_diff << " (tolerance: " << tolerance << ")" << std::endl;
        std::cout << "Average absolute difference = " << avg_diff << std::endl;
        std::cout << "Root Mean Square Error (RMSE) = " << rmse << std::endl;
    }

    return 0;
}