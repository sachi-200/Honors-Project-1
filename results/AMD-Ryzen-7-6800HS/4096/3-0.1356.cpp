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
#include <numeric>    // For std::iota (not currently used, but often helpful)
#include <algorithm>  // For std::min, std::max, std::fill, std::abs
#include <cmath>      // For std::sqrt
#include <map>        // For std::map in cpuinfo parsing (optional, refined)
#include <set>        // For std::set in cpuinfo parsing (optional, refined)
#include <iomanip>    // For std::setprecision in output

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
//
// Note on "Autotune harness": Given the strict requirement for `constexpr` values at the top
// and exact function signatures, true runtime autotuning that dynamically changes these
// core blocking parameters for the optimized `gemm_avx2`/`gemm_avx512` kernels during a single run
// is not feasible without violating the given constraints. These `constexpr` values represent
// the *pre-tuned* parameters for the target CPU (AMD Ryzen 7 6800HS).
// A "harness" might involve a separate benchmarking tool to experiment with different `constexpr` sets
// and then manually updating these values for recompilation.

// Target CPU: AMD Ryzen 7 6800HS (Zen 3+)
// L1d cache: 32KB per core
// L2 cache: 512KB per core
// L3 cache: 16MB shared (shared across CCX)

// Blocking sizes (BM, BN, BK):
// These determine the amount of data processed in larger blocks to optimize L2/L3 cache reuse.
// The values (BM=128, BN=128, BK=64) are chosen to target L2 cache utilization.
// This configuration aims to fit the thread's active working set (packed A_block, packed B_block,
// and local C_tile_local) primarily within L2 cache, minimizing expensive main memory (DRAM) access.
// - C_tile_local (BM x BN packed): 128 rows * 128 cols * 4 bytes/float = 65536 bytes (~64 KB).
// - B_packed (BK x BN packed): 64 rows * 128 cols * 4 bytes/float = 32768 bytes (~32 KB).
// - A_packed_block (BM x BK): 128 rows * 64 cols * 4 bytes/float = 32768 bytes (~32 KB).
//
// The critical working set actively used by a thread in its local buffers:
//   ~32KB (A_packed_block) + ~32KB (B_packed) + ~64KB (C_tile_local) = ~128KB.
// This fits well within the 512KB L2 cache per core. The outermost loops `mb` and `nb` are
// parallelized with OpenMP, ensuring each thread computes a distinct `C` block. The `kb` loop
// is innermost for these block loops, allowing `C_tile_local` to accumulate contributions from
// all `K`-blocks before being written back to global `C`.
//
// `C_tile_local` is initialized to zero once per `(mb, nb)` block, avoiding redundant reads
// from global `C` (which is initially zero as per problem statement). Packing `A` and `B` blocks
// for each `kb` iteration is a trade-off: it reduces memory traffic for the micro-kernel by converting
// strided global memory access into contiguous local buffer access, at the cost of packing overhead.
// The default `omp_set_num_threads` behavior has been adjusted in `main` to favor physical cores for Zen architectures.
constexpr int BM_DEFAULT = 128; // Block size for M (rows of C)
constexpr int BN_DEFAULT = 128; // Block size for N (columns of C)
constexpr int BK_DEFAULT = 64;  // Block size for K (inner dimension)

// Micro-kernel parameters (MR, NR, UNROLL_K):
// These define the smallest unit of work processed by SIMD registers.
// MR and NR determine the register blocking for the C matrix accumulators.
// UNROLL_K determines the loop unrolling factor for the innermost K-loop,
// reducing loop overhead and exposing more instructions for pipelining.

// AVX2 specific micro-kernel parameters:
// VEC_WIDTH_AVX2 (8 floats). Zen 3/4 typically have 16 YMM registers.
// MR_AVX2 (8) `__m256` accumulators for C take 8 * 32 bytes = 256 bytes of register space.
// UNROLL_K_AVX2 is set to 4. This choice allows up to 16 YMM registers to be live at a time
// (8 for C_acc, 4 for B_vec, 4 for A_broadcast), maximizing register utilization without spilling.
constexpr int VEC_WIDTH_AVX2 = 8; // __m256 holds 8 floats
constexpr int MR_AVX2 = 8;        // Number of rows of A/C to process in the micro-kernel
constexpr int NR_AVX2 = VEC_WIDTH_AVX2; // Number of columns of B/C to process (must be VEC_WIDTH_AVX2 = 8)
constexpr int UNROLL_K_AVX2 = 4;  // K-dimension unroll factor for AVX2 micro-kernel

// AVX-512 specific micro-kernel parameters:
// VEC_WIDTH_AVX512 (16 floats). AVX-512 has 32 ZMM registers.
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
    ofs << std::fixed << std::setprecision(6); // Set precision for float output
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
// The C matrix block (BMxBN) is initialized to zero in a thread-local buffer (C_tile_local)
// once per (M,N) block. It accumulates over the K dimension, and then the final
// `C_tile_local` is written back to global C once.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
#if defined(__AVX2__) && defined(__FMA__)
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

        // Thread-private buffer for packed A blocks (BM x BK).
        // A_packed_block converts strided row access from global A to contiguous row access
        // for the micro-kernel's K-loop within a BM-block.
        std::unique_ptr<float, AlignedFree> A_packed_block_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BM) * BK * sizeof(float))
        );
        float* A_packed_block = A_packed_block_uptr.get();

        // Thread-private buffer for a local C block (BM x BN).
        // This is crucial for reducing L1d/L2 cache misses on C.
        std::unique_ptr<float, AlignedFree> C_tile_local_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BM) * BN * sizeof(float))
        );
        float* C_tile_local = C_tile_local_uptr.get();

        if (!B_packed || !A_packed_block || !C_tile_local) {
            std::cerr << "Memory allocation for buffers failed in AVX2 kernel. This thread might perform poorly." << std::endl;
            #ifdef _OPENMP
            #pragma omp critical
            std::exit(1);
            #endif
        }

#ifdef _OPENMP
    // The `schedule(static)` ensures an even distribution of work and avoids dynamic overhead.
    // `collapse(2)` applies the scheduling across both outermost loops (M and N blocks).
    // This effectively tiles the C matrix into BMxBN chunks and assigns them to threads.
    #pragma omp for collapse(2) schedule(static)
#endif
        for (int mb = 0; mb < M; mb += BM) { // Loop over M-blocks (rows of C)
            for (int nb = 0; nb < N; nb += BN) { // Loop over N-blocks (columns of C)
                // Determine actual block boundaries, handling matrix edges
                int M_block_end = std::min(mb + BM, M);
                int N_block_end = std::min(nb + BN, N);
                int M_current_block_len = M_block_end - mb; // Actual length of M-dim for this block
                int N_current_block_len = N_block_end - nb; // Actual length of N-dim for this block

                // --- Initialize C_tile_local block to zeros (once per (mb, nb) block) ---
                // As per problem statement, initial C is zero. Directly zero out the local tile.
                // This avoids unnecessary reads from global C memory.
                std::memset(C_tile_local, 0, static_cast<size_t>(BM) * BN * sizeof(float));

                for (int kb = 0; kb < K; kb += BK) { // Loop over K-blocks (inner dimension for accumulation)
                    int K_block_end = std::min(kb + BK, K);
                    int K_current_block_len = K_block_end - kb;

                    // --- Prefetching for the *current* A block to be packed (original A data) ---
                    // T0 hint for L1/L2 cache, bringing data closer for the packing operation.
                    for (int r_pf_idx = 0; r_pf_idx < M_current_block_len; ++r_pf_idx) {
                        for (int k_pf_idx = 0; k_pf_idx < K_current_block_len; k_pf_idx += (alignment / sizeof(float))) {
                            _mm_prefetch((const char*)&A[static_cast<size_t>(mb + r_pf_idx) * lda + kb + k_pf_idx], _MM_HINT_T0);
                        }
                    }

                    // Pack A block (BM x BK) from original A into `A_packed_block`.
                    // This is done once per (mb, nb, kb) tuple. A_packed_block stores rows of A_block contiguously.
                    for (int r_idx = 0; r_idx < M_current_block_len; ++r_idx) {
                        float* A_packed_row_ptr = &A_packed_block[static_cast<size_t>(r_idx) * BK];
                        const float* A_orig_row_ptr = &A[static_cast<size_t>(mb + r_idx) * lda + kb];
                        int k_pack_idx = 0;
                        for (; k_pack_idx + VEC_WIDTH <= K_current_block_len; k_pack_idx += VEC_WIDTH) {
                            _mm256_store_ps(A_packed_row_ptr + k_pack_idx, _mm256_loadu_ps(A_orig_row_ptr + k_pack_idx));
                        }
                        for (; k_pack_idx < K_current_block_len; ++k_pack_idx) {
                            A_packed_row_ptr[k_pack_idx] = A_orig_row_ptr[k_pack_idx];
                        }
                        // Zero fill remaining columns in A_packed_block for this row if K_current_block_len < BK
                        if (K_current_block_len < BK) {
                            std::memset(A_packed_row_ptr + K_current_block_len, 0, (BK - K_current_block_len) * sizeof(float));
                        }
                    }
                    // Zero fill remaining rows in A_packed_block if M_current_block_len < BM
                    for (int r_idx = M_current_block_len; r_idx < BM; ++r_idx) {
                        std::memset(&A_packed_block[static_cast<size_t>(r_idx) * BK], 0, BK * sizeof(float));
                    }
                    
                    // --- Prefetching for the *current* B block to be packed (original B data) ---
                    // T0 hint for L1/L2 cache, bringing data closer for the packing operation.
                    for (int p_pf_idx = 0; p_pf_idx < K_current_block_len; ++p_pf_idx) {
                        for (int j_pf_idx = 0; j_pf_idx < N_current_block_len; j_pf_idx += (alignment / sizeof(float))) {
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + p_pf_idx) * ldb + nb + j_pf_idx], _MM_HINT_T0);
                        }
                    }

                    // Pack the current B-block (BK x BN) from original B into `B_packed`.
                    // B_packed stores rows of B_block contiguously.
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
                        // Zero fill remaining columns in B_packed for this row if N_current_block_len < BN
                        if (N_current_block_len < BN) {
                            std::memset(B_packed_row_ptr + N_current_block_len, 0, (BN - N_current_block_len) * sizeof(float));
                        }
                    }
                    // Zero fill remaining rows in B_packed if K_current_block_len < BK
                    for (int p_idx = K_current_block_len; p_idx < BK; ++p_idx) {
                        std::memset(&B_packed[static_cast<size_t>(p_idx) * BN], 0, BK * sizeof(float));
                    }
                    
                    // --- Prefetching for the *next* A and B blocks (original A and B data) ---
                    // These are T1 prefetch hints, aiming for L2/L3 cache, for subsequent `kb` iterations.
                    if (kb + BK < K) { // Prefetch next K-block of A and B
                        for (int r_pf = 0; r_pf < M_current_block_len; r_pf += MR) {
                             _mm_prefetch((const char*)&A[static_cast<size_t>(mb + r_pf) * lda + (kb + BK)], _MM_HINT_T1);
                        }
                        for (int p_pf = 0; p_pf < K_current_block_len; p_pf += UNROLL_K) { 
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + BK + p_pf) * ldb + nb], _MM_HINT_T1);
                        }
                    }

                    // Micro-kernel loops. Loop bounds for `i_offset` and `j_offset` are relative to the current `mb`, `nb` block.
                    for (int i_offset = 0; i_offset < M_current_block_len; i_offset += MR) { // Loop over micro-kernel rows of C
                        int i_micro_end_offset = std::min(i_offset + MR, M_current_block_len);

                        for (int j_offset = 0; j_offset < N_current_block_len; j_offset += NR) { // Loop over micro-kernel columns of C
                            // Prefetch C_tile_local for the next processing block. (T0 hint for L1 cache)
                            if (j_offset + NR < N_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset) * BN + (j_offset + NR)], _MM_HINT_T0);
                            } else if (i_offset + MR < M_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset + MR) * BN + j_offset], _MM_HINT_T0);
                            }

                            // C_acc: Array of AVX2 registers for accumulating results for an MR x NR block of C
                            // for the current K-block. These are initialized to zero.
                            alignas(32) __m256 C_acc[MR]; 
                            for (int r = 0; r < MR; ++r) {
                                C_acc[r] = _mm256_setzero_ps(); // Initialize accumulators to zero for this K-block's contribution
                            }

                            // Inner K loop with unrolling.
                            // Iterates over the full BK dimension of the packed B buffer.
                            // Zero-padding of A_packed_block and B_packed buffers handles K-dimension tails automatically.
                            for (int p_relative_idx = 0; p_relative_idx < BK; p_relative_idx += UNROLL_K) {
                                for (int p_unroll = 0; p_unroll < UNROLL_K; ++p_unroll) {
                                    int current_k_idx_in_block = p_relative_idx + p_unroll;
                                    // The packing functions ensure A_packed_block and B_packed are zero-padded
                                    // for k >= K_current_block_len (within the BK dimension).
                                    // Thus, loading from these locations will correctly yield 0.0f for out-of-bounds k.

                                    // Load vector from B_packed. Access is contiguous and aligned by construction.
                                    // B_packed is 64-byte aligned, BN (row stride) is BN_DEFAULT=128 (multiple of VEC_WIDTH=8).
                                    // j_offset increments by NR=8 (multiple of 8). So, &B_packed[...] is always 32-byte aligned.
                                    __m256 B_vec = _mm256_load_ps(&B_packed[static_cast<size_t>(current_k_idx_in_block) * BN + j_offset]);

                                    // Main multiplication and accumulation for the MR x NR block.
                                    for (int r = 0; r < MR; ++r) {
                                        // Load scalar from A_packed_block and broadcast.
                                        // A_packed_block is zero-padded for out-of-bounds rows and columns.
                                        // Access to A_packed_block uses i_offset + r for row index within the BM-block.
                                        __m256 A_broadcast = _mm256_set1_ps(A_packed_block[static_cast<size_t>(i_offset + r) * BK + current_k_idx_in_block]);
                                        
                                        // Fused Multiply-Add (FMA): C_acc[r] = C_acc[r] + (A_broadcast * B_vec)
                                        C_acc[r] = _mm256_fmadd_ps(A_broadcast, B_vec, C_acc[r]);
                                    }
                                }
                            }

                            // Add accumulated results from C_acc to C_tile_local.
                            // C_tile_local is aligned and zero-padded for N-tails.
                            // The `j_offset` is always a multiple of `NR` (and thus `VEC_WIDTH`).
                            // So, `C_local_ptr` is always aligned.
                            // Full vector load/add/store is safe and correct, as values beyond N_current_block_len in C_tile_local are zero due to earlier padding.
                            for (int r = 0; r < MR; ++r) {
                                if (i_offset + r < i_micro_end_offset) { // Ensure row is within M-block
                                    float* C_local_ptr = &C_tile_local[static_cast<size_t>(i_offset + r) * BN + j_offset];
                                    __m256 existing_C_tile_vec = _mm256_load_ps(C_local_ptr); // Load ALIGNED C_tile_local part
                                    __m256 updated_C_tile_vec = _mm256_add_ps(existing_C_tile_vec, C_acc[r]);
                                    _mm256_store_ps(C_local_ptr, updated_C_tile_vec); // Store ALIGNED
                                }
                            }
                        } // end j_offset loop (NR micro-block)
                    } // end i_offset loop (MR micro-block)
                } // end kb loop (all K contributions accumulated for this (mb, nb) block)

                // --- Store C_tile_local back to global C (once per (mb, nb) block) ---
                // This step must handle potential unaligned access and N-dimension tails for the global C matrix.
                for (int r = 0; r < M_current_block_len; ++r) {
                    float* C_orig_row_ptr = &C[static_cast<size_t>(mb + r) * ldc + nb];
                    const float* C_local_row_ptr = &C_tile_local[static_cast<size_t>(r) * BN];

                    int j_idx = 0;
                    for (; j_idx + VEC_WIDTH <= N_current_block_len; j_idx += VEC_WIDTH) {
                        _mm256_storeu_ps(C_orig_row_ptr + j_idx, _mm256_load_ps(C_local_row_ptr + j_idx));
                    }
                    // Handle N-tail for storing C using scalar operations (global C might not be padded)
                    for (; j_idx < N_current_block_len; ++j_idx) {
                        C_orig_row_ptr[j_idx] = C_local_row_ptr[j_idx];
                    }
                }
                // End storing C block
            } // end nb loop (BN block)
        } // end mb loop (BM block)
    } // End OpenMP parallel region
#else // AVX2 and FMA not enabled at compile time
    std::cerr << "Warning: AVX2 kernel not compiled or FMA not enabled. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif // __AVX2__ && __FMA__
}

// AVX-512 + FMA optimized GEMM implementation.
// Similar blocking strategy to AVX2, but uses `__m512` (16 floats) and AVX-512's
// advanced mask registers for efficient tail handling in loads and stores.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__) && defined(__FMA__)
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

        // Thread-private buffer for packed A blocks.
        std::unique_ptr<float, AlignedFree> A_packed_block_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BM) * BK * sizeof(float))
        );
        float* A_packed_block = A_packed_block_uptr.get();

        // Thread-private buffer for a local C block (BM x BN).
        std::unique_ptr<float, AlignedFree> C_tile_local_uptr(
            (float*)std::aligned_alloc(alignment, static_cast<size_t>(BM) * BN * sizeof(float))
        );
        float* C_tile_local = C_tile_local_uptr.get();

        if (!B_packed || !A_packed_block || !C_tile_local) {
            std::cerr << "Memory allocation for buffers failed in AVX-512 kernel. This thread might perform poorly." << std::endl;
            #ifdef _OPENMP
            #pragma omp critical
            std::exit(1);
            #endif
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

                // --- Initialize C_tile_local block to zeros (once per (mb, nb) block) ---
                std::memset(C_tile_local, 0, static_cast<size_t>(BM) * BN * sizeof(float));

                for (int kb = 0; kb < K; kb += BK) {
                    int K_block_end = std::min(kb + BK, K);
                    int K_current_block_len = K_block_end - kb;

                    // --- Prefetching for the *current* A block to be packed (original A data) ---
                    for (int r_pf_idx = 0; r_pf_idx < M_current_block_len; ++r_pf_idx) {
                        for (int k_pf_idx = 0; k_pf_idx < K_current_block_len; k_pf_idx += (alignment / sizeof(float))) {
                            _mm_prefetch((const char*)&A[static_cast<size_t>(mb + r_pf_idx) * lda + kb + k_pf_idx], _MM_HINT_T0);
                        }
                    }

                    // Pack A block (BM x BK)
                    for (int r_idx = 0; r_idx < M_current_block_len; ++r_idx) {
                        float* A_packed_row_ptr = &A_packed_block[static_cast<size_t>(r_idx) * BK];
                        const float* A_orig_row_ptr = &A[static_cast<size_t>(mb + r_idx) * lda + kb];
                        int k_pack_idx = 0;
                        for (; k_pack_idx + VEC_WIDTH <= K_current_block_len; k_pack_idx += VEC_WIDTH) {
                            _mm512_store_ps(A_packed_row_ptr + k_pack_idx, _mm512_loadu_ps(A_orig_row_ptr + k_pack_idx));
                        }
                        if (K_current_block_len - k_pack_idx > 0) {
                            __mmask16 tail_mask = static_cast<__mmask16>((1 << (K_current_block_len - k_pack_idx)) - 1);
                            _mm512_mask_store_ps(A_packed_row_ptr + k_pack_idx, tail_mask, _mm512_maskz_loadu_ps(tail_mask, A_orig_row_ptr + k_pack_idx));
                        }
                        if (K_current_block_len < BK) {
                            std::memset(A_packed_row_ptr + K_current_block_len, 0, (BK - K_current_block_len) * sizeof(float));
                        }
                    }
                    for (int r_idx = M_current_block_len; r_idx < BM; ++r_idx) {
                        std::memset(&A_packed_block[static_cast<size_t>(r_idx) * BK], 0, BK * sizeof(float));
                    }

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
                    
                    // --- Prefetching for the *next* A and B blocks (original A and B data) ---
                    if (kb + BK < K) { // Prefetch next K-block of A and B
                        for (int r_pf = 0; r_pf < M_current_block_len; r_pf += MR) {
                            _mm_prefetch((const char*)&A[static_cast<size_t>(mb + r_pf) * lda + (kb + BK)], _MM_HINT_T1);
                        }
                        for (int p_pf = 0; p_pf < K_current_block_len; p_pf += UNROLL_K) {
                            _mm_prefetch((const char*)&B[static_cast<size_t>(kb + BK + p_pf) * ldb + nb], _MM_HINT_T1);
                        }
                    }

                    for (int i_offset = 0; i_offset < M_current_block_len; i_offset += MR) {
                        int i_micro_end_offset = std::min(i_offset + MR, M_current_block_len);

                        for (int j_offset = 0; j_offset < N_current_block_len; j_offset += NR) {
                            int j_micro_end_offset = std::min(j_offset + NR, N_current_block_len);

                            // Prefetch C_tile_local for the next processing block. (T0 hint for L1 cache)
                            if (j_offset + NR < N_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset) * BN + (j_offset + NR)], _MM_HINT_T0);
                            } else if (i_offset + MR < M_current_block_len) {
                                _mm_prefetch((const char*)&C_tile_local[static_cast<size_t>(i_offset + MR) * BN + j_offset], _MM_HINT_T0);
                            }

                            alignas(64) __m512 C_acc[MR]; // Array of AVX-512 registers for accumulation
                            for (int r = 0; r < MR; ++r) {
                                C_acc[r] = _mm512_setzero_ps(); // Initialize accumulators to zero
                            }

                            // Inner K loop with unrolling.
                            for (int p_relative_idx = 0; p_relative_idx < BK; p_relative_idx += UNROLL_K) {
                                for (int p_unroll = 0; p_unroll < UNROLL_K; ++p_unroll) {
                                    int current_k_idx_in_block = p_relative_idx + p_unroll;
                                    // The packing functions ensure A_packed_block and B_packed are zero-padded
                                    // for k >= K_current_block_len (within the BK dimension).
                                    // Thus, loading from these locations will correctly yield 0.0f for out-of-bounds k.

                                    // Load vector B[p_col][j_col:j_col+NR] from B_packed
                                    // B_packed is 64-byte aligned, BN is BN_DEFAULT=128 (multiple of 64).
                                    // j_offset increments by NR=16 (multiple of 16). So, &B_packed[...] is always 64-byte aligned.
                                    __m512 B_vec = _mm512_load_ps(&B_packed[static_cast<size_t>(current_k_idx_in_block) * BN + j_offset]);

                                    for (int r = 0; r < MR; ++r) {
                                        // Broadcast scalar A[i_row][p_col] from A_packed_block
                                        __m512 A_broadcast = _mm512_set1_ps(A_packed_block[static_cast<size_t>(i_offset + r) * BK + current_k_idx_in_block]);
                                        
                                        // Fused Multiply-Add
                                        C_acc[r] = _mm512_fmadd_ps(A_broadcast, B_vec, C_acc[r]);
                                    }
                                }
                            }

                            // Add accumulated results from C_acc to C_tile_local (masked for N-tails).
                            // The C_tile_local was already zero-initialized, so this is directly adding to it.
                            __mmask16 n_mask_update = static_cast<__mmask16>((1 << (j_micro_end_offset - j_offset)) - 1);
                            if (j_micro_end_offset - j_offset <= 0) n_mask_update = 0; // Ensure mask is zero if no columns to process.

                            for (int r = 0; r < MR; ++r) {
                                if (i_offset + r < i_micro_end_offset) {
                                    float* C_local_ptr = &C_tile_local[static_cast<size_t>(i_offset + r) * BN + j_offset];
                                    // C_local_ptr is aligned and C_tile_local is zero-padded.
                                    // For AVX512, maskz_load and mask_store are optimal for tails.
                                    __m512 existing_C_tile_vec = _mm512_maskz_load_ps(n_mask_update, C_local_ptr); // Masked load from C_tile_local
                                    __m512 updated_C_tile_vec = _mm512_add_ps(existing_C_tile_vec, C_acc[r]);
                                    _mm512_mask_store_ps(C_local_ptr, n_mask_update, updated_C_tile_vec); // Masked store to C_tile_local
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
#else // AVX512F and FMA not enabled at compile time
    std::cerr << "Warning: AVX-512 kernel not compiled or FMA not enabled. Falling back to AVX2 or scalar." << std::endl;
    // Fallback order ensures the next best available kernel is tried.
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc); 
#endif // __AVX512F__ && __FMA__
}


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
    // MSVC specific __cpuidex for runtime feature detection
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
                std::cout << "  -t <threads>      : Number of OpenMP threads (default: auto-detected physical cores or system default)\n";
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
        // Dynamically get number of physical cores to provide better SMT hint for OpenMP.
        // This attempts to count unique (physical_id, core_id) pairs to get physical core count.
        int physical_cores = 0;
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        std::set<std::pair<int, int>> physical_core_ids; // Store unique (physical ID, core ID) pairs
        int current_physical_id = -1;
        int current_core_id = -1;

        while (std::getline(cpuinfo, line)) {
            // Trim leading/trailing whitespace and check for relevant keywords
            size_t colon_pos = line.find(':');
            if (colon_pos != std::string::npos) {
                std::string key = line.substr(0, colon_pos);
                std::string value = line.substr(colon_pos + 1);
                
                // Trim whitespace from key and value
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                value.erase(0, value.find_first_not_of(" \t"));
                value.erase(value.find_last_not_of(" \t") + 1);

                if (key == "physical id") {
                    current_physical_id = std::stoi(value);
                } else if (key == "core id") {
                    current_core_id = std::stoi(value);
                }
            }
            
            if (current_physical_id != -1 && current_core_id != -1) {
                physical_core_ids.insert({current_physical_id, current_core_id});
                current_physical_id = -1; // Reset for next block
                current_core_id = -1;      // Reset for next block
            }
        }
        physical_cores = physical_core_ids.size();
        
        // --- If not explicitly set by user, attempt to set OpenMP threads to physical cores. ---
        if (physical_cores > 0) {
            omp_set_num_threads(physical_cores);
            std::cout << "Detected " << physical_cores << " physical cores. Setting OpenMP threads to " << physical_cores << "." << std::endl;
        } else {
            std::cout << "Could not reliably determine physical core count. Using default OpenMP threads: " << omp_get_max_threads() << std::endl;
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
    std::fill(C, C + size_C, 0.0f); // Explicitly zero C as per problem requirement
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

    // Choose a numerical tolerance. For large matrices (K=1000+), accumulated floating-point
    // errors due to FMA's different summation order compared to strict scalar operations can
    // lead to larger differences. A tolerance of 5.0e-3f is more practical for robustness.
    float tolerance = 5.0e-3f; 
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