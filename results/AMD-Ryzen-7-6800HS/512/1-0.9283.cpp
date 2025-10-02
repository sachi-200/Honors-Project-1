// Example compile commands for GCC/Clang:
//
// For AVX-512 path (requires a CPU with AVX-512, e.g., Intel Skylake-SP/X, Ice Lake, Sapphire Rapids):
// g++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx512
// clang++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx512
// (Note: AMD Ryzen 6000 series does NOT support AVX-512, so this path will not be chosen at runtime on the target CPU.)
//
// For AVX2 path (compatible with AMD Ryzen 6000 series, Intel Haswell/Broadwell/Skylake/Kaby Lake/Coffee Lake):
// g++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx2
// clang++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx2
//
// Portable (will fall back to AVX2 or scalar if AVX-512 not available/enabled):
// g++ -O3 -march=native -fopenmp -std=c++17 gemm.cpp -o gemm_portable
// clang++ -O3 -march=native -fopenmp -std=c++17 gemm.cpp -o gemm_portable
// (This is the recommended one for the AMD Ryzen 7 6800HS target as it will enable AVX2/FMA via `native` and use the `gemm_avx2` kernel.)
//
// Note: -march=x86-64-v3 includes AVX, AVX2, FMA, BMI, BMI2.
//       -march=x86-64-v4 includes AVX512F, AVX512BW, AVX512DQ, AVX512VL, VPOPCNTDQ.
// The `__builtin_cpu_supports` check handles runtime dispatch even if compiled with a broad -march or specific -mavx flags.

#include <iostream>
#include <vector>
#include <cstring>      // For memcpy
#include <chrono>       // For timing
#include <random>       // For random matrix initialization
#include <cassert>      // For assertions
#include <cmath>        // For fabs
#include <string>
#include <numeric>      // For std::iota
#include <fstream>      // For file output
#include <filesystem>   // For creating directories (C++17)
#include <cstdlib>      // For aligned_alloc, free
#include <algorithm>    // For std::min, std::max, std::fill
#include <limits>       // For std::numeric_limits

#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>  // For AVX, AVX2, AVX-512 intrinsics
#else
// Fallback for other compilers, though typically GCC/Clang are used for intrinsics
#warning "Non-GCC/Clang compiler detected. Intrinsics might not be available or may require specific headers."
#endif

#ifdef _OPENMP
#include <omp.h>        // For OpenMP multi-threading
#else
#warning "OpenMP not available. Code will run in single-threaded mode."
#endif

// --- Tunable Parameters ---
// These are default values for cache-aware tiling and micro-kernel blocking.
// A full autotuner would dynamically explore and select optimal values.
// Chosen to be reasonable for typical CPU caches on modern x86-64 CPUs.
//
// Cache Level Considerations:
// - L1d cache (data cache): typically 32-64KB per core. We aim for micro-kernel data to fit here.
// - L2 cache: typically 256KB-1MB per core. BM*BK and BK*BN blocks should fit here for reuse.
// - L3 cache: typically 4MB-32MB+ shared across cores. Larger blocks might fit here.

// Core blocking factors for the outer loops (M, N, K dimensions)
// These define the size of sub-matrices processed by individual threads and cache blocks.
// For a Ryzen 7 6800HS (Zen 3/Zen 3+ based), L1=32KB, L2=512KB, L3=16MB.
// BM, BN, BK are chosen to keep blocks in L2/L3 cache.
constexpr int DEFAULT_BM_TILE = 192; // Block size for M (rows of A/C block). A multiple of MR_ACCUM is good.
constexpr int DEFAULT_BN_TILE = 192; // Block size for N (cols of B/C block). A multiple of NR_ACCUM (vector width) is good.
constexpr int DEFAULT_BK_TILE = 256; // Block size for K (inner dimension). This often determines L1/L2 data locality for A and B.

// Micro-kernel unroll factor for K
// Unrolls the innermost loop over K to expose more instruction-level parallelism.
constexpr int UNROLL_K_MICRO = 8; 

// Micro-kernel register blocking for AVX2 (8 floats per __m256 vector)
constexpr int AVX2_VEC_FLOATS = 8;
constexpr int MR_AVX2_ACCUM = 6; // Number of rows of C computed concurrently (each holding NR_AVX2_ACCUM columns). Uses MR_AVX2_ACCUM `__m256` registers.
constexpr int NR_AVX2_ACCUM = AVX2_VEC_FLOATS; // Number of columns of C computed concurrently. Must be AVX2_VEC_FLOATS (8).

// Micro-kernel register blocking for AVX-512 (16 floats per __m512 vector)
constexpr int AVX512_VEC_FLOATS = 16;
constexpr int MR_AVX512_ACCUM = 6; // Number of rows of C computed concurrently. Uses MR_AVX512_ACCUM `__m512` registers.
constexpr int NR_AVX512_ACCUM = AVX512_VEC_FLOATS; // Number of columns of C computed concurrently. Must be AVX512_VEC_FLOATS (16).

// Cache prefetching distances
// `_MM_HINT_T0` prefetches into all levels of cache.
constexpr int PREFETCH_DIST_A = 4; // Prefetch A block a few iterations ahead
constexpr int PREFETCH_DIST_B = 4; // Prefetch B block a few iterations ahead

// Define alignment for matrices (64 bytes for AVX-512, 32 for AVX2, choose max for compatibility)
constexpr size_t MATRIX_ALIGNMENT = 64;

// Matrix storage: Row-major convention is used throughout this implementation.
// A is M x K with leading dimension lda (lda >= K)
// B is K x N with leading dimension ldb (ldb >= N)
// C is M x N with leading dimension ldc (ldc >= N)

// --- Helper Functions ---

// Aligned memory allocation and deallocation
float* aligned_alloc_float(size_t num_elements) {
    void* ptr = nullptr;
    // C++17 std::aligned_alloc guarantees alignment and returns nullptr on failure.
    // Ensure total allocated size is a multiple of alignment for safety.
    size_t size_bytes = num_elements * sizeof(float);
    ptr = std::aligned_alloc(MATRIX_ALIGNMENT, (size_bytes + MATRIX_ALIGNMENT - 1) / MATRIX_ALIGNMENT * MATRIX_ALIGNMENT);
    if (!ptr) {
        throw std::bad_alloc();
    }
    return static_cast<float*>(ptr);
}

void aligned_free_float(float* ptr) {
    std::free(ptr);
}

// Function to write a matrix to a text file
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::filesystem::path dir = "workspace";
    if (!std::filesystem::exists(dir)) {
        std::error_code ec;
        if (!std::filesystem::create_directory(dir, ec)) {
            std::cerr << "Error: Could not create directory " << dir << ": " << ec.message() << std::endl;
            return;
        }
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file.precision(6); // Set precision for floating point output
    file << std::fixed;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[static_cast<size_t>(i) * ld + j] << (j == cols - 1 ? "" : " ");
        }
        file << std::endl;
    }
    file.close();
}

// --- Scalar GEMM Implementation (Reference) ---
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<size_t>(i) * lda + k] * B[static_cast<size_t>(k) * ldb + j];
            }
            C[static_cast<size_t>(i) * ldc + j] += sum; // Accumulate, as C might have initial values (though main initializes to 0)
        }
    }
}

// --- AVX2 GEMM Implementation ---
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
#if defined(__AVX2__) && defined(__FMA__)
    // Blocking parameters
    const int BM = DEFAULT_BM_TILE;
    const int BN = DEFAULT_BN_TILE;
    const int BK = DEFAULT_BK_TILE;

    // Outer loops for M and N dimensions, parallelized with OpenMP.
    // `schedule(static)` is chosen for simplicity and to ensure even load distribution
    // among threads when dimensions are large and tiles are uniform.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            // Inner loop for K dimension (cache blocking)
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                // Determine actual block sizes for current iteration, handling matrix boundaries.
                const int m_block_end = std::min(m_block_start + BM, M);
                const int n_block_end = std::min(n_block_start + BN, N);
                const int k_block_end = std::min(k_block_start + BK, K);

                // Micro-kernel for the current block: Processes MR_AVX2_ACCUM rows by NR_AVX2_ACCUM columns
                for (int m_i = m_block_start; m_i < m_block_end; m_i += MR_AVX2_ACCUM) {
                    const int m_rows_to_compute = std::min(MR_AVX2_ACCUM, m_block_end - m_i);

                    for (int n_j = n_block_start; n_j < n_block_end; n_j += NR_AVX2_ACCUM) {
                        const int n_cols_to_compute = std::min(NR_AVX2_ACCUM, n_block_end - n_j);

                        // Initialize C accumulators (MR_AVX2_ACCUM `__m256` vectors, each for 8 floats)
                        // These accumulators hold the sum of A_block * B_block for the CURRENT BK block.
                        __m256 c_acc[MR_AVX2_ACCUM];
                        for (int r = 0; r < m_rows_to_compute; ++r) {
                            c_acc[r] = _mm256_setzero_ps(); // Set to zero for current K-block accumulation
                        }

                        // K loop for inner micro-kernel: Iterates through the K dimension of the block
                        for (int k_s = k_block_start; k_s < k_block_end; k_s += UNROLL_K_MICRO) {
                            // Prefetching for A and B data into cache
                            // _MM_HINT_T0 loads into L1, L2, L3 cache levels.
                            // Ensure prefetch addresses are within bounds by checking against total matrix size.
                            // Using static_cast<size_t> for matrix bounds comparison to prevent potential int overflow.
                            if (static_cast<size_t>(m_i) * lda + (k_s + PREFETCH_DIST_A * UNROLL_K_MICRO) < static_cast<size_t>(M) * K)
                                _mm_prefetch((char*)(A + static_cast<size_t>(m_i) * lda + (k_s + PREFETCH_DIST_A * UNROLL_K_MICRO)), _MM_HINT_T0);
                            if (static_cast<size_t>(k_s + PREFETCH_DIST_B * UNROLL_K_MICRO) * ldb + n_j < static_cast<size_t>(K) * N)
                                _mm_prefetch((char*)(B + static_cast<size_t>(k_s + PREFETCH_DIST_B * UNROLL_K_MICRO) * ldb + n_j), _MM_HINT_T0);

                            // UNROLL_K_MICRO loop: Unroll to expose more ILP (Instruction-Level Parallelism)
                            for (int k_unroll = 0; k_unroll < UNROLL_K_MICRO; ++k_unroll) {
                                int current_k = k_s + k_unroll;
                                if (current_k >= k_block_end) break; // K boundary check for unrolling

                                // Load B vector: `__m256` loads 8 floats from B.
                                // `_mm256_loadu_ps` is unaligned load. It might read past N if N is not a multiple of 8,
                                // but this is generally safe if the memory is allocated to allow it (which `aligned_alloc_float` does).
                                __m256 b_vec = _mm256_loadu_ps(&B[static_cast<size_t>(current_k) * ldb + n_j]);

                                for (int r = 0; r < m_rows_to_compute; ++r) {
                                    // Load A scalar and broadcast it to all elements of a `__m256` vector
                                    __m256 a_broadcast = _mm256_set1_ps(A[static_cast<size_t>(m_i + r) * lda + current_k]);
                                    // Fused Multiply-Add (FMA): c_acc = A_broadcast * B_vec + c_acc
                                    c_acc[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // Store accumulators back to C. C = C + accumulated_block.
                        for (int r = 0; r < m_rows_to_compute; ++r) {
                            float* c_dest_ptr = &C[static_cast<size_t>(m_i + r) * ldc + n_j];

                            if (n_cols_to_compute == NR_AVX2_ACCUM) { // Full vector store
                                __m256 current_c_val = _mm256_loadu_ps(c_dest_ptr); // Load existing C values
                                __m256 final_c_val = _mm256_add_ps(current_c_val, c_acc[r]); // Add the current K-block's sum
                                _mm256_storeu_ps(c_dest_ptr, final_c_val); // Store back
                            } else {
                                // Scalar tail processing for C stores to avoid out-of-bounds loads/stores.
                                // This extracts the vector into a temporary array and then adds element-wise to C.
                                // It's safer than masked vector loads/stores for AVX2 due to complexities with `_mm256_maskload_ps` potentially
                                // faulting on partial cache line reads beyond actual allocation boundary.
                                alignas(MATRIX_ALIGNMENT) float temp_c_acc_buffer[NR_AVX2_ACCUM];
                                _mm256_storeu_ps(temp_c_acc_buffer, c_acc[r]); // Store vector accumulator to temp array

                                for (int col = 0; col < n_cols_to_compute; ++col) {
                                    c_dest_ptr[col] += temp_c_acc_buffer[col]; // Scalar add to existing C
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else // __AVX2__ not defined or __FMA__ not defined
    std::cerr << "AVX2 kernel called but AVX2/FMA not enabled at compile time. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

// --- AVX-512 GEMM Implementation ---
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__) && defined(__FMA__)
    // Blocking parameters
    const int BM = DEFAULT_BM_TILE;
    const int BN = DEFAULT_BN_TILE;
    const int BK = DEFAULT_BK_TILE;

    // Outer loops for M and N dimensions, parallelized with OpenMP.
    // `schedule(static)` is chosen for simplicity and to ensure even load distribution.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            // Inner loop for K dimension (cache blocking)
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                // Determine actual block sizes for current iteration, handling matrix boundaries.
                const int m_block_end = std::min(m_block_start + BM, M);
                const int n_block_end = std::min(n_block_start + BN, N);
                const int k_block_end = std::min(k_block_start + BK, K);

                // Micro-kernel for the current block: Processes MR_AVX512_ACCUM rows by NR_AVX512_ACCUM columns
                for (int m_i = m_block_start; m_i < m_block_end; m_i += MR_AVX512_ACCUM) {
                    const int m_rows_to_compute = std::min(MR_AVX512_ACCUM, m_block_end - m_i);

                    for (int n_j = n_block_start; n_j < n_block_end; n_j += NR_AVX512_ACCUM) {
                        const int n_cols_to_compute = std::min(NR_AVX512_ACCUM, n_block_end - n_j);

                        // Initialize C accumulators (MR_AVX512_ACCUM `__m512` vectors, each for 16 floats)
                        __m512 c_acc[MR_AVX512_ACCUM];
                        for (int r = 0; r < m_rows_to_compute; ++r) {
                            c_acc[r] = _mm512_setzero_ps(); // Set to zero for current K-block accumulation
                        }

                        // K loop for inner micro-kernel: Iterates through the K dimension of the block
                        for (int k_s = k_block_start; k_s < k_block_end; k_s += UNROLL_K_MICRO) {
                            // Prefetching for A and B data into cache
                            // Using static_cast<size_t> for matrix bounds comparison to prevent potential int overflow.
                            if (static_cast<size_t>(m_i) * lda + (k_s + PREFETCH_DIST_A * UNROLL_K_MICRO) < static_cast<size_t>(M) * K)
                                _mm_prefetch((char*)(A + static_cast<size_t>(m_i) * lda + (k_s + PREFETCH_DIST_A * UNROLL_K_MICRO)), _MM_HINT_T0);
                            if (static_cast<size_t>(k_s + PREFETCH_DIST_B * UNROLL_K_MICRO) * ldb + n_j < static_cast<size_t>(K) * N)
                                _mm_prefetch((char*)(B + static_cast<size_t>(k_s + PREFETCH_DIST_B * UNROLL_K_MICRO) * ldb + n_j), _MM_HINT_T0);

                            // UNROLL_K_MICRO loop: Unroll to expose more ILP
                            for (int k_unroll = 0; k_unroll < UNROLL_K_MICRO; ++k_unroll) {
                                int current_k = k_s + k_unroll;
                                if (current_k >= k_block_end) break; // K boundary check for unrolling

                                // Load B vector: `__m512` loads 16 floats from B.
                                // AVX-512 masks allow loading only valid elements, filling the rest with zeros (maskz_loadu).
                                __mmask16 b_load_mask = (1 << std::min(n_cols_to_compute, NR_AVX512_ACCUM)) - 1;
                                __m512 b_vec = _mm512_maskz_loadu_ps(b_load_mask, &B[static_cast<size_t>(current_k) * ldb + n_j]);

                                for (int r = 0; r < m_rows_to_compute; ++r) {
                                    // Load A scalar and broadcast it to all elements of a `__m512` vector
                                    __m512 a_broadcast = _mm512_set1_ps(A[static_cast<size_t>(m_i + r) * lda + current_k]);
                                    // Fused Multiply-Add (FMA): c_acc = A_broadcast * B_vec + c_acc
                                    c_acc[r] = _mm512_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // Store accumulators back to C. C = C + accumulated_block.
                        for (int r = 0; r < m_rows_to_compute; ++r) {
                            float* c_dest_ptr = &C[static_cast<size_t>(m_i + r) * ldc + n_j];

                            // For AVX-512, perform masked load, add, and store for tails directly with intrinsics.
                            __mmask16 c_mask = (1 << std::min(n_cols_to_compute, NR_AVX512_ACCUM)) - 1;
                            __m512 current_c_val = _mm512_maskz_loadu_ps(c_mask, c_dest_ptr); // Load existing C values, mask out-of-bounds
                            __m512 final_c_val = _mm512_add_ps(current_c_val, c_acc[r]); // Add the current K-block's sum
                            _mm512_mask_storeu_ps(c_dest_ptr, c_mask, final_c_val); // Store back, mask out-of-bounds
                        }
                    }
                }
            }
        }
    }
#else // __AVX512F__ not defined or __FMA__ not defined
    std::cerr << "AVX-512 kernel called but AVX-512F/FMA not enabled at compile time. Falling back to AVX2 if available, else scalar." << std::endl;
    // Fallback if AVX-512 compilation failed, try AVX2 if available.
#if defined(__AVX2__) && defined(__FMA__)
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
#endif
}

// --- Top-level GEMM function with runtime dispatch ---
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    // Runtime dispatch based on CPU features detected using GCC/Clang builtins.
#if defined(__GNUC__) || defined(__clang__)
    // For AMD Ryzen 7 6800HS, `__builtin_cpu_supports("avx512f")` will be false.
    // The AVX2 path will be selected.
    if (__builtin_cpu_supports("avx512f")) {
        // This path is taken if the CPU supports AVX-512F and it's enabled at compile time.
        std::cout << "Dispatching to AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    } else if (__builtin_cpu_supports("avx2")) {
        // This is the expected path for AMD Ryzen 7 6800HS, as it supports AVX2 and FMA.
        std::cout << "Dispatching to AVX2 kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    } else {
        std::cerr << "No AVX2 or AVX-512 support detected at runtime. Falling back to scalar GEMM." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
#else
    // Fallback for compilers without `__builtin_cpu_supports` (e.g., MSVC, or general compilation).
    // It relies on compile-time feature flags like `__AVX512F__` or `__AVX2__`.
#ifdef __AVX512F__ // If compiled with AVX-512 (e.g., -mavx512f)
    std::cout << "Dispatching to AVX-512 kernel (compile-time). " << std::endl;
    gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
#elif defined(__AVX2__) // Else, if compiled with AVX2 (e.g., -mavx2)
    std::cout << "Dispatching to AVX2 kernel (compile-time). " << std::endl;
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else // Otherwise, use scalar
    std::cerr << "No AVX2 or AVX-512 support compiled. Falling back to scalar GEMM." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
#endif
}

// --- Main Function for Demo and Benchmarking ---
int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    unsigned int seed = 42;
    int num_threads = 0; // 0 means OpenMP default (uses OMP_NUM_THREADS env var or system cores)
    bool dump_matrices = false;
    bool check_correctness = true;

    // Track how many positional arguments for M, N, K have been consumed
    int positional_dims_set = 0; 
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Check if argument is a flag (starts with '-')
        if (arg.length() > 0 && arg[0] == '-') {
            if (arg == "-M" && i + 1 < argc) {
                M = std::stoi(argv[++i]);
            } else if (arg == "-N" && i + 1 < argc) {
                N = std::stoi(argv[++i]);
            } else if (arg == "-K" && i + 1 < argc) {
                K = std::stoi(argv[++i]);
            } else if (arg == "-s" && i + 1 < argc) {
                seed = std::stoul(argv[++i]);
            } else if (arg == "-t" && i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            } else if (arg == "--dump-matrices") {
                dump_matrices = true;
            } else if (arg == "--no-check") {
                check_correctness = false;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0] << " [M N K] [-M <rows>] [-N <cols>] [-K <inner>] [-s <seed>] [-t <threads>] [--dump-matrices] [--no-check] [--help]" << std::endl;
                std::cout << "  [M N K]: Positional arguments for matrix dimensions (optional, if first arguments are numbers)." << std::endl;
                std::cout << "           These are overridden by -M, -N, -K flags if also present." << std::endl;
                std::cout << "  -M <rows>: Set M dimension (default: 1024)" << std::endl;
                std::cout << "  -N <cols>: Set N dimension (default: 1024)" << std::endl;
                std::cout << "  -K <inner>: Set K dimension (default: 1024)" << std::endl;
                std::cout << "  -s <seed>: Random seed for matrix initialization (default: 42)" << std::endl;
                std::cout << "  -t <threads>: Number of OpenMP threads (default: use OMP_NUM_THREADS env var or system default)" << std::endl;
                std::cout << "  --dump-matrices: Write A.txt, B.txt, C.txt to 'workspace' directory" << std::endl;
                std::cout << "  --no-check: Skip correctness verification with scalar GEMM" << std::endl;
                return 0;
            } else {
                std::cerr << "Unknown or incomplete argument: " << arg << ". Use --help for usage." << std::endl;
                return 1;
            }
        } else { // Not a flag, try to parse as positional M, N, K
            try {
                if (positional_dims_set == 0) {
                    M = std::stoi(arg);
                    positional_dims_set++;
                } else if (positional_dims_set == 1) {
                    N = std::stoi(arg);
                    positional_dims_set++;
                } else if (positional_dims_set == 2) {
                    K = std::stoi(arg);
                    positional_dims_set++;
                } else {
                    // We've already parsed M, N, K positionally, treat further non-flags as errors
                    std::cerr << "Unknown positional argument (expected M, N, K only): " << arg << ". Use --help for usage." << std::endl;
                    return 1;
                }
            } catch (const std::invalid_argument& e) {
                // If it's not a flag and not a valid number for M, N, K, then it's an error.
                std::cerr << "Invalid argument: " << arg << ". Expected numeric M, N, K or a flag. Use --help for usage." << std::endl;
                return 1;
            } catch (const std::out_of_range& e) {
                std::cerr << "Numeric argument out of range: " << arg << ". Use --help for usage." << std::endl;
                return 1;
            }
        }
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads." << std::endl;
#else
    if (num_threads > 0) {
        std::cerr << "Warning: -t " << num_threads << " specified but OpenMP not enabled. Running single-threaded." << std::endl;
    }
    std::cout << "OpenMP not enabled. Running single-threaded." << std::endl;
#endif

    // Allocate matrices with specified alignment using aligned_alloc_float.
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* C_ref = nullptr; // For correctness check

    try {
        A = aligned_alloc_float(static_cast<size_t>(M) * K);
        B = aligned_alloc_float(static_cast<size_t>(K) * N);
        C = aligned_alloc_float(static_cast<size_t>(M) * N);
        if (check_correctness) {
            C_ref = aligned_alloc_float(static_cast<size_t>(M) * N);
        }
    } catch (const std::bad_alloc& e) {
        std::cerr << "Failed to allocate memory: " << e.what() << std::endl;
        // Clean up any partially allocated memory
        aligned_free_float(A);
        aligned_free_float(B);
        aligned_free_float(C);
        if (C_ref) aligned_free_float(C_ref);
        return 1;
    }

    // Initialize matrices
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Values between 0 and 1

    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i) A[i] = dist(rng);
    for (size_t i = 0; i < static_cast<size_t>(K) * N; ++i) B[i] = dist(rng);
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f); // C must be initialized to 0 for accumulation to work correctly
    if (check_correctness) {
        std::fill(C_ref, C_ref + static_cast<size_t>(M) * N, 0.0f);
    }

    // Leading dimensions (lda, ldb, ldc).
    // For simplicity, we assume tightly packed matrices, so lda=K, ldb=N, ldc=N.
    // In real-world scenarios, these might be larger than the actual dimension.
    int lda = K;
    int ldb = N;
    int ldc = N;

    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    if (dump_matrices) {
        std::cout << "Dumping A and B matrices to workspace/A.txt and workspace/B.txt..." << std::endl;
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
    }

    // Warm-up run (optional but recommended for consistent benchmarks as it fills caches)
    std::cout << "Performing warm-up run..." << std::endl;
    // Call the dispatching gemm, but ensure C is cleared afterwards
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f); // Clear C for actual timed run

    // Timed run
    std::cout << "Starting timed GEMM..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate performance
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    double time_ms = duration.count();
    long long flop_count = 2LL * M * N * K; // 2 operations (mul, add) per element
    double gflops = (double)flop_count / (time_ms * 1e6); // Divide by 1 million for GFLOP/s

    std::cout << "GEMM finished in " << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    if (dump_matrices) {
        std::cout << "Dumping C matrix to workspace/C.txt..." << std::endl;
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
    }

    // Correctness check
    if (check_correctness) {
        std::cout << "Running scalar GEMM for correctness check..." << std::endl;
        // Re-initialize C_ref to 0.0f, as scalar gemm also accumulates
        std::fill(C_ref, C_ref + static_cast<size_t>(M) * N, 0.0f); 
        auto scalar_start_time = std::chrono::high_resolution_clock::now();
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc); // Scalar reference also accumulates
        auto scalar_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> scalar_duration = scalar_end_time - scalar_start_time;
        std::cout << "Scalar GEMM finished in " << scalar_duration.count() << " ms" << std::endl;

        const float epsilon = 1e-4f; // Tolerance for floating point comparison
        bool correct = true;
        float max_diff = 0.0f;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float diff = std::fabs(C[static_cast<size_t>(i) * ldc + j] - C_ref[static_cast<size_t>(i) * ldc + j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (diff > epsilon) {
                    // For brevity, only print the first few mismatches if there are many.
                    // The max_diff will summarize the overall error.
                    if (correct) { // Only print first error detailed
                        std::cerr << "Mismatch at C[" << i << "][" << j << "]: Optimized=" << C[static_cast<size_t>(i) * ldc + j]
                                  << ", Scalar=" << C_ref[static_cast<size_t>(i) * ldc + j] << ". Diff=" << diff << std::endl;
                    }
                    correct = false;
                }
            }
        }

        if (correct) {
            std::cout << "Correctness check PASSED." << std::endl;
        } else {
            std::cerr << "Correctness check FAILED! Max absolute difference: " << max_diff << std::endl;
        }
    }

    // Clean up allocated memory
    aligned_free_float(A);
    aligned_free_float(B);
    aligned_free_float(C);
    if (C_ref) aligned_free_float(C_ref);

    return 0;
}