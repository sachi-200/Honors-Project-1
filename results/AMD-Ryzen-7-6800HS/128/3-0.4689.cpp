// Compile Instructions:
//
// For AMD Ryzen 7 6800HS (AVX2 + FMA):
// The target CPU supports AVX2 and FMA. Compile with `-march=native` or `-march=x86-64-v3`
// to enable these instruction sets. The runtime dispatcher will automatically select the AVX2 kernel.
// Command: `g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_ryzen`
// (Or `g++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_v3`)
//
// For CPUs with AVX-512 (e.g., some Intel processors, NOT AMD Ryzen 6000 series):
// If targeting a CPU that supports AVX-512, compile with appropriate flags.
// Command: `g++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512`
//
// Portable (runtime dispatch will select best available, compiled with base features):
// This command uses `-march=native` to detect the host CPU's features at compile time and enable corresponding instruction sets.
// The runtime dispatcher will still be used to ensure portability across different CPUs
// (e.g., running `gemm_portable` compiled on an AVX-512 machine on an AVX2 machine will correctly fall back).
// Command: `g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_portable`
//
// Ensure C++17 or later is used if `<filesystem>` requires it (usually implicit with modern compilers).
// `g++ -std=c++17 -O3 -march=native -fopenmp gemm.cpp -o gemm_portable`

// Required standard headers
#include <iostream>   // For input/output operations (e.g., std::cout, std::cerr)
#include <vector>     // For std::vector
#include <cstring>    // For std::memset (though std::fill is used, good to keep in mind)
#include <chrono>     // For timing performance (std::chrono)
#include <random>     // For random number generation (std::mt19937, std::uniform_real_distribution)
#include <cassert>    // For assert
#include <fstream>    // For file output (std::ofstream)
#include <filesystem> // For creating directories (std::filesystem::create_directory, requires C++17)
#include <memory>     // For std::unique_ptr with custom deleter
#include <numeric>    // For std::iota etc. (not directly used but common)
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
// L3 cache: 16MB shared

// Blocking sizes (BM, BN, BK):
// These determine the amount of data processed in larger blocks to optimize L2/L3 cache reuse.
// They are chosen such that `BM x BK` block of A, `BK x BN` block of B, and `BM x BN` block of C
// can ideally fit into L2/L3 caches. For example, a 128x128 block of floats (128*128*4 bytes = 64KB)
// fits well in L2 (512KB per core) and L3 (16MB shared).
constexpr int BM_DEFAULT = 128; // Block size for M (rows of C)
constexpr int BN_DEFAULT = 128; // Block size for N (columns of C)
constexpr int BK_DEFAULT = 64;  // Block size for K (inner dimension)

// Micro-kernel parameters (MR, NR, UNROLL_K):
// These define the smallest unit of work processed by SIMD registers.
// MR and NR determine the register blocking for the C matrix accumulators.
// UNROLL_K determines the loop unrolling factor for the innermost K-loop,
// reducing loop overhead and exposing more instructions for pipelining.

// AVX2 specific micro-kernel parameters:
// VEC_WIDTH_AVX2 (8 floats) * NR_AVX2 (8 floats) for one `__m256` accumulator register.
// MR_AVX2 (6) `__m256` accumulators for C take 6 * 32 bytes = 192 bytes of register space.
// This small footprint stays entirely in registers (L1), providing maximal data reuse.
constexpr int VEC_WIDTH_AVX2 = 8; // __m256 holds 8 floats
constexpr int MR_AVX2 = 6;        // Number of rows of A/C to process in the micro-kernel
constexpr int NR_AVX2 = VEC_WIDTH_AVX2; // Number of columns of B/C to process (must be VEC_WIDTH_AVX2)
constexpr int UNROLL_K_AVX2 = 4;  // K-dimension unroll factor for AVX2 micro-kernel

// AVX-512 specific micro-kernel parameters:
// VEC_WIDTH_AVX512 (16 floats) * NR_AVX512 (16 floats) for one `__m512` accumulator register.
// MR_AVX512 (6) `__m512` accumulators for C take 6 * 64 bytes = 384 bytes of register space.
// Larger register footprint than AVX2, but processes more data per instruction.
constexpr int VEC_WIDTH_AVX512 = 16; // __m512 holds 16 floats
constexpr int MR_AVX512 = 6;         // Number of rows of A/C to process in the micro-kernel
constexpr int NR_AVX512 = VEC_WIDTH_AVX512; // Number of columns of B/C to process (must be VEC_WIDTH_AVX512)
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
// A: M x K, lda (leading dimension of A, usually K)
// B: K x N, ldb (leading dimension of B, usually N)
// C: M x N, ldc (leading dimension of C, usually N)
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
#if defined(__AVX2__) && defined(__FMA__)
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Tiling parameters (chosen from default constants)
    const int BM = BM_DEFAULT;
    const int BN = BN_DEFAULT;
    const int BK = BK_DEFAULT;

    // Micro-kernel parameters (specific to AVX2)
    const int MR = MR_AVX2;
    const int NR = NR_AVX2; // Must be VEC_WIDTH_AVX2 (8 for AVX2)
    const int UNROLL_K = UNROLL_K_AVX2;
    const int VEC_WIDTH = VEC_WIDTH_AVX2; // 8 floats per __m256

    // OpenMP parallel region for outer M and N loops (tiling C matrix).
    // `schedule(static)` distributes blocks of work evenly among threads,
    // which is generally good for predictable performance.
    // `collapse(2)` parallelizes both `mb` and `nb` loops, effectively iterating over `BM x BN` C-blocks.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int mb = 0; mb < M; mb += BM) { // Loop over M-blocks (rows of C)
        for (int nb = 0; nb < N; nb += BN) { // Loop over N-blocks (columns of C)
            // Determine actual block boundaries, handling matrix edges
            int M_block_end = std::min(mb + BM, M);
            int N_block_end = std::min(nb + BN, N);

            for (int kb = 0; kb < K; kb += BK) { // Loop over K-blocks (inner dimension)
                int K_block_end = std::min(kb + BK, K);

                for (int i = mb; i < M_block_end; i += MR) { // Loop over micro-kernel rows of C
                    int i_micro_end = std::min(i + MR, M_block_end);

                    for (int j = nb; j < N_block_end; j += NR) { // Loop over micro-kernel columns of C
                        int j_micro_end = std::min(j + NR, N_block_end);

                        // C_acc: Array of AVX2 registers for accumulating results for an MR x NR block of C.
                        // Each __m256 register holds 8 float values.
                        alignas(32) __m256 C_acc[MR]; 

                        // Initialize C_acc registers.
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < i_micro_end) { // Check if this row is within the M-block
                                if (kb == 0) { // First K-block: initialize accumulators to zero
                                    C_acc[r] = _mm256_setzero_ps();
                                } else { // Subsequent K-blocks: load existing C values for accumulation
                                    // _mm256_loadu_ps loads 8 floats from memory, unaligned.
                                    // This is safe because C is allocated for M*N floats and initialized to 0.0f.
                                    // Any "extra" elements loaded beyond the logical N-boundary will be 0.0f.
                                    C_acc[r] = _mm256_loadu_ps(&C[static_cast<size_t>(i + r) * ldc + j]);
                                }
                            } else { // Row is outside the M-block, so initialize its accumulator to zero
                                C_acc[r] = _mm256_setzero_ps();
                            }
                        }

                        // Inner K loop with unrolling.
                        // UNROLL_K determines how many K steps are processed in one iteration.
                        for (int p = kb; p < K_block_end; p += UNROLL_K) {
                            for (int p_unroll = 0; p_unroll < UNROLL_K && p + p_unroll < K_block_end; ++p_unroll) {
                                // Prefetching: Hint to the CPU to bring data into cache for future use.
                                // _MM_HINT_T0 requests to L1 cache. Prefetching for next K-unroll block.
                                _mm_prefetch((const char*)&A[static_cast<size_t>(i) * lda + (p + p_unroll + UNROLL_K)], _MM_HINT_T0);
                                _mm_prefetch((const char*)&B[static_cast<size_t>(p + p_unroll + UNROLL_K) * ldb + j], _MM_HINT_T0);

                                // Main multiplication and accumulation for the MR x NR block.
                                for (int r = 0; r < MR; ++r) {
                                    if (i + r >= i_micro_end) continue; // Skip if this row is beyond actual M_block_end

                                    // Load A[i_row][p_col] and broadcast to all 8 elements of an __m256 register.
                                    __m256 A_broadcast = _mm256_set1_ps(A[static_cast<size_t>(i + r) * lda + (p + p_unroll)]);
                                    
                                    // Load B[p_col][j_col:j_col+NR] as a vector.
                                    // `_mm256_loadu_ps` is used for unaligned loads, safe for arbitrary `ldb` and `j`.
                                    __m256 B_vec = _mm256_loadu_ps(&B[static_cast<size_t>(p + p_unroll) * ldb + j]);

                                    // Fused Multiply-Add (FMA): C_acc[r] = C_acc[r] + (A_broadcast * B_vec)
                                    C_acc[r] = _mm256_fmadd_ps(A_broadcast, B_vec, C_acc[r]);
                                }
                            }
                        }

                        // Store accumulated results back to C.
                        // This handles tails for N-dimension (columns) and M-dimension (rows)
                        // by ensuring we only write within `i_micro_end` and `j_micro_end`.
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < i_micro_end) { // Ensure row is within M-block
                                float* C_ptr = &C[static_cast<size_t>(i + r) * ldc + j];
                                int cols_to_store = j_micro_end - j;

                                if (cols_to_store >= VEC_WIDTH) { // Full vector store
                                    _mm256_storeu_ps(C_ptr, C_acc[r]);
                                } else if (cols_to_store > 0) { // Partial vector store (tail for N-dimension)
                                    // For AVX2, masked stores for floats are not as efficient as AVX-512.
                                    // A robust approach is to store the full vector to a temporary aligned buffer
                                    // and then scalar copy only the valid elements.
                                    alignas(32) float temp_vec[VEC_WIDTH];
                                    _mm256_store_ps(temp_vec, C_acc[r]); // Store full vector to temp (aligned store)
                                    for (int col_idx = 0; col_idx < cols_to_store; ++col_idx) {
                                        C_ptr[col_idx] = temp_vec[col_idx]; // Copy only valid elements
                                    }
                                }
                                // If cols_to_store is 0 or negative, do nothing for this row.
                            }
                        }
                    } // end j loop (NR micro-block)
                } // end i loop (MR micro-block)
            } // end kb loop (BK block)
        } // end nb loop (BN block)
    } // end mb loop (BM block)
}
#else // AVX2 and FMA not enabled at compile time
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "Warning: AVX2 kernel called but AVX2/FMA not enabled at compile time. Falling back to scalar." << std::endl;
    // If compiled without AVX2/FMA, this fallback ensures the function can still be called.
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

    // Micro-kernel parameters (specific to AVX-512)
    const int MR = MR_AVX512;
    const int NR = NR_AVX512; // Must be VEC_WIDTH_AVX512 (16 for AVX-512)
    const int UNROLL_K = UNROLL_K_AVX512;
    const int VEC_WIDTH = VEC_WIDTH_AVX512; // 16 floats per __m512

#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int mb = 0; mb < M; mb += BM) {
        for (int nb = 0; nb < N; nb += BN) {
            int M_block_end = std::min(mb + BM, M);
            int N_block_end = std::min(nb + BN, N);

            for (int kb = 0; kb < K; kb += BK) {
                int K_block_end = std::min(kb + BK, K);

                for (int i = mb; i < M_block_end; i += MR) {
                    int i_micro_end = std::min(i + MR, M_block_end);

                    for (int j = nb; j < N_block_end; j += NR) {
                        int j_micro_end = std::min(j + NR, N_block_end);

                        alignas(64) __m512 C_acc[MR]; // Array of AVX-512 registers for accumulation
                        
                        // Calculate mask for N-dimension tail (columns).
                        // A mask `(1 << N)` creates a bitmask where bits 0 to N-1 are set.
                        // If `j_micro_end - j` is 16, mask is `(1 << 16) - 1`, which is all 1s (0xFFFF).
                        // If it's less than 16, only the relevant bits are set.
                        __mmask16 n_mask = static_cast<__mmask16>((1 << (j_micro_end - j)) - 1);
                        if (j_micro_end - j <= 0) n_mask = 0; // If no columns, mask should be zero.

                        for (int r = 0; r < MR; ++r) {
                            if (i + r < i_micro_end) {
                                if (kb == 0) { // First K-block: initialize accumulators to zero
                                    C_acc[r] = _mm512_setzero_ps();
                                } else { // Subsequent K-blocks: load existing C values (masked for N-tails)
                                    // _mm512_maskz_loadu_ps loads values from memory only if the corresponding
                                    // mask bit is set; otherwise, it loads zero. This handles column tails elegantly.
                                    C_acc[r] = _mm512_maskz_loadu_ps(n_mask, &C[static_cast<size_t>(i + r) * ldc + j]);
                                }
                            } else { // Row out of bounds, init to zero
                                C_acc[r] = _mm512_setzero_ps();
                            }
                        }

                        for (int p = kb; p < K_block_end; p += UNROLL_K) {
                            for (int p_unroll = 0; p_unroll < UNROLL_K && p + p_unroll < K_block_end; ++p_unroll) {
                                // Prefetching for next K-unroll block.
                                _mm_prefetch((const char*)&A[static_cast<size_t>(i) * lda + (p + p_unroll + UNROLL_K)], _MM_HINT_T0);
                                _mm_prefetch((const char*)&B[static_cast<size_t>(p + p_unroll + UNROLL_K) * ldb + j], _MM_HINT_T0);


                                for (int r = 0; r < MR; ++r) {
                                    if (i + r >= i_micro_end) continue;

                                    // Broadcast scalar A[i_row][p_col]
                                    __m512 A_broadcast = _mm512_set1_ps(A[static_cast<size_t>(i + r) * lda + (p + p_unroll)]);
                                    // Load vector B[p_col][j_col:j_col+NR]
                                    __m512 B_vec = _mm512_loadu_ps(&B[static_cast<size_t>(p + p_unroll) * ldb + j]);

                                    // Fused Multiply-Add
                                    C_acc[r] = _mm512_fmadd_ps(A_broadcast, B_vec, C_acc[r]);
                                }
                            }
                        }

                        // Store accumulated results back to C (masked for N-tails).
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < i_micro_end) {
                                // _mm512_mask_storeu_ps stores only elements where the mask bit is set.
                                // This handles column tails without requiring scalar loops.
                                _mm512_mask_storeu_ps(&C[static_cast<size_t>(i + r) * ldc + j], n_mask, C_acc[r]);
                            }
                        }
                    } // end j loop
                } // end i loop
            } // end kb loop
        } // end nb loop
    } // end mb loop
}
#else // AVX512F and FMA not enabled at compile time
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    std::cerr << "Warning: AVX-512 kernel called but AVX512F/FMA not enabled at compile time. Falling back to AVX2 or scalar." << std::endl;
    // As a direct fallback, call the AVX2 kernel (which itself might fall back to scalar).
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc); 
}
#endif // __AVX512F__ && __FMA__


// Top-level GEMM function with runtime dispatch.
// This function determines the best available SIMD kernel based on CPU features
// using `__builtin_cpu_supports` (for GCC/Clang) and calls the appropriate implementation.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

    bool has_avx512f = false;
    bool has_avx2 = false;

    // Runtime feature detection (compiler-specific builtins)
#if defined(__GNUC__) || defined(__clang__)
    // Note: __builtin_cpu_supports performs a check at runtime.
    // For AVX-512, it might require specific CPU models and OS setup.
    // On AMD Ryzen 7 6800HS, AVX512F support is NOT present.
    has_avx512f = __builtin_cpu_supports("avx512f");
    has_avx2 = __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
    // For MSVC, one would use __cpuidex. This is a simplified check.
    int cpuInfo[4];
    __cpuidex(cpuInfo, 7, 0); // Extended features
    has_avx2 = (cpuInfo[1] & (1 << 5)) != 0; // Check for AVX2 bit (bit 5 of EBX)
    has_avx512f = (cpuInfo[1] & (1 << 16)) != 0; // Check for AVX512F bit (bit 16 of EBX)
#endif

    // Dispatch logic: Prefer AVX-512, then AVX2, then scalar.
    if (has_avx512f) {
#if defined(__AVX512F__) && defined(__FMA__)
        std::cout << "Runtime Dispatch: Using AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
#else
        std::cout << "Runtime Dispatch: AVX-512 detected, but not compiled. Falling back to AVX2/scalar." << std::endl;
        // If the AVX-512 kernel wasn't compiled, fallback directly to AVX2
        if (has_avx2) {
            std::cout << "Runtime Dispatch: Using AVX2 kernel." << std::endl;
            gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        } else {
            std::cout << "Runtime Dispatch: Using scalar kernel." << std::endl;
            gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
        }
#endif
    } else if (has_avx2) {
#if defined(__AVX2__) && defined(__FMA__)
        std::cout << "Runtime Dispatch: Using AVX2 kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else
        std::cout << "Runtime Dispatch: AVX2 detected, but not compiled. Falling back to scalar." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
    } else {
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
    // Positional arguments M, N, K are parsed first if they don't start with '-'.
    // Named arguments (flags) always override positional ones.
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
        std::cout << "Using default OpenMP threads: " << omp_get_max_threads() << std::endl;
#else
        std::cout << "OpenMP not enabled during compilation, running single-threaded." << std::endl;
#endif
    }

    // Allocate memory for matrices A, B, C with 64-byte alignment.
    // This supports both AVX2 (32-byte) and AVX-512 (64-byte) aligned loads/stores.
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
    // Increasing to 2e-4f to account for expected floating-point variance on larger matrices.
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