// Compile instructions for different CPU architectures and capabilities:
//
// 1. For AVX-512 capable CPUs (e.g., Intel Skylake-X, Rocket Lake, AMD Zen 4 and later):
//    It is important to target specific AVX-512 extensions. `x86-64-v4` provides a good baseline,
//    or specify desired features explicitly.
//    g++ -O3 -std=c++17 -march=x86-64-v4 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//    (For older GCC/Clang or specific needs, you might use `-march=skylake-avx512` or explicitly
//     list extensions: `-mavx512f -mavx512bw -mavx512dq -mavx512vl`).
//
// 2. For AVX2+FMA capable CPUs (e.g., AMD Ryzen 7 6800HS, most modern x86-64 CPUs):
//    `x86-64-v2` guarantees SSE4.2, AVX, POPCNT. We explicitly add AVX2 and FMA.
//    g++ -O3 -std=c++17 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// 3. For portable compilation (compiler determines best ISA available on the build machine):
//    `march=native` allows the compiler to use the best features of the *current* CPU.
//    g++ -O3 -std=c++17 -march=native -fopenmp gemm.cpp -o gemm_native
//
// 4. For maximum compatibility (scalar fallback, no specific SIMD instructions targeted):
//    g++ -O3 -std=c++17 -fopenmp gemm.cpp -o gemm_scalar_fallback

#include <iostream>
#include <vector>
#include <cstring>    // For std::memset (though not explicitly used for C initialization here)
#include <chrono>     // For performance timing
#include <random>     // For matrix initialization
#include <cassert>    // For assertions
#include <string>
#include <fstream>    // For file I/O (dumping matrices)
#include <iomanip>    // For std::fixed, std::setprecision
#include <algorithm>  // For std::min
#include <limits>     // For std::numeric_limits

#ifdef _OPENMP
#include <omp.h>      // OpenMP header for multi-threading
#else
// Define dummy OpenMP functions if OpenMP is not available at compile time.
// This allows the code to compile and run in a single-threaded manner without OpenMP errors.
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
void omp_set_num_threads(int) {} // Dummy function to avoid compiler errors if called
#endif

// Intrinsics headers for SIMD operations.
// These typically include headers like <xmmintrin.h>, <emmintrin.h>, <pmmintrin.h>, etc.
#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h> // Central header for AVX, AVX2, AVX-512 intrinsics
#include <xmmintrin.h> // Contains _mm_malloc and _mm_free for aligned memory allocation
#else
// Error for compilers not supporting GCC/Clang extensions needed for runtime dispatch and target attributes.
#error "Unsupported compiler: Only GCC and Clang are supported for __builtin_cpu_supports and __attribute__((target))"
#endif

// C++17 filesystem for creating directories (e.g., for dumping matrices).
#include <filesystem>


// --- Autotuning Parameters ---
// These constants define the blocking strategy for cache-aware tiling.
// They are chosen to promote data reuse in L1/L2 caches and reduce TLB misses.
// Tuning these values can significantly impact performance for different CPUs and matrix sizes.

// BM: Block size for the M dimension (rows of C and A).
// BN: Block size for the N dimension (columns of C and B).
// BK: Block size for the K dimension (inner dimension for A and B).
//
// Example calculation for cache footprint with float (4 bytes):
// For BM=96, BN=128, BK=64:
// - A_block (BMxBK): 96 * 64 * 4 bytes = 24 KB
// - B_block (BKxBN): 64 * 128 * 4 bytes = 32 KB
// - C_block (BMxBN): 96 * 128 * 4 bytes = 48 KB
// Total working set for one C-block calculation (A_sub, B_sub, C_sub): ~104 KB.
// This size typically fits well within a modern CPU's L2 cache (e.g., 256-512KB per core)
// or a shared L3 cache, minimizing costly main memory access.
const int BM = 96;
const int BN = 128;
const int BK = 64;

// UNROLL_K: Inner K loop unroll factor for the micro-kernel.
// This explicitly unrolls the innermost loop (K dimension), exposing more independent
// Fused Multiply-Add (FMA) operations to the CPU's execution units. This helps
// keep the pipelines full, hide FMA latency, and improve instruction-level parallelism.
// A higher value might increase register pressure but can also improve performance.
const int UNROLL_K = 16; 

// Register blocking factors for micro-kernels:
// MR: Micro-kernel row block size (number of rows of A/C processed concurrently by one vector unit).
// NR: Micro-kernel column block size (number of columns of B/C processed concurrently by one vector unit).
//     NR typically corresponds to the vector register width in floats (e.g., 8 for AVX2, 16 for AVX-512).

const int MR_AVX2 = 4;  // Processes 4 rows of A/C simultaneously in the micro-kernel
const int NR_AVX2 = 8;  // AVX2 vector width for float (256-bit = 8 floats)

const int MR_AVX512 = 4; // Processes 4 rows of A/C simultaneously in the micro-kernel
const int NR_AVX512 = 16; // AVX-512 vector width for float (512-bit = 16 floats)


// --- Helper Functions ---

// write_matrix_to_file: Writes a matrix to a specified file.
// - `filename`: The path to the output file.
// - `matrix`: Pointer to the first element of the matrix.
// - `rows`, `cols`: Dimensions of the matrix.
// - `ld`: Leading dimension (stride) of the matrix, which is the number of elements
//   between the start of one row and the start of the next. For dense row-major, ld = cols.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    ofs << std::fixed << std::setprecision(6); // Format output for better readability and precision
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " "); // Space-separated values
        }
        ofs << "\n"; // Newline after each row
    }
    ofs.close();
}


// --- GEMM Implementations ---

// gemm_scalar: Reference scalar GEMM implementation.
// This function computes C = A * B. All matrices are assumed to be row-major.
// `lda`, `ldb`, `ldc` are leading dimensions (strides) of A, B, and C, respectively.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float c_val = 0.0f; // Accumulator for C[i][j]
            for (int k = 0; k < K; ++k) {
                c_val += A[i * lda + k] * B[k * ldb + j]; // C[i][j] += A[i][k] * B[k][j]
            }
            C[i * ldc + j] = c_val; // Store final accumulated value
        }
    }
}

// gemm_avx2: AVX2 + FMA optimized GEMM implementation.
// Uses GCC/Clang's `__attribute__((target("avx2,fma")))` to enable AVX2 and FMA
// instructions specifically for this function, regardless of the global compilation flags.
void __attribute__((target("avx2,fma"))) gemm_avx2(const float* A, const float* B, float* C,
                                                    int M, int N, int K,
                                                    int lda, int ldb, int ldc) {
    // OpenMP parallelization strategy:
    // The outer loops iterate over M and N blocks of the C matrix.
    // `collapse(2)` tells OpenMP to parallelize the combined iteration space of `i_block` and `j_block`.
    // `schedule(static)` distributes loop iterations evenly among threads, providing predictable load balancing.
    // Each thread computes a distinct C_block tile, ensuring thread-safe writes to C without synchronization overhead.
#pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = 0; i_block < M; i_block += BM) {
        for (int j_block = 0; j_block < N; j_block += BN) {
            // Determine current block dimensions, handling tails for M and N.
            // `std::min` ensures that we don't read/write past the actual matrix bounds.
            int M_current_block_size = std::min(BM, M - i_block);
            int N_current_block_size = std::min(BN, N - j_block);

            // K-block loop: Iterates over the K dimension, accumulating partial results for the current C_block.
            // This is the "outer-k" loop, accumulating C[i_block:i_block+BM, j_block:j_block+BN] += A[i_block:i_block+BM, k_block:k_block+BK] * B[k_block:k_block+BK, j_block:j_block+BN].
            for (int k_block = 0; k_block < K; k_block += BK) {
                int K_current_block_size = std::min(BK, K - k_block);

                // Micro-kernel processing for the current block of C (M_current_block_size x N_current_block_size).
                // Loops over rows of A/C, blocked by MR_AVX2 (register block size for M).
                for (int i = 0; i < M_current_block_size; i += MR_AVX2) {
                    int MR_actual = std::min(MR_AVX2, M_current_block_size - i);

                    // Loops over columns of B/C, vectorized by NR_AVX2 (vector length for N).
                    for (int j = 0; j < N_current_block_size; j += NR_AVX2) {
                        int NR_actual = std::min(NR_AVX2, N_current_block_size - j);

                        // Initialize AVX2 accumulator registers for MR_AVX2 rows * NR_AVX2 columns of C.
                        // Each `__m256` register holds 8 floats. We need `MR_AVX2` such registers.
                        __m256 c_acc[MR_AVX2];
                        for (int r = 0; r < MR_AVX2; ++r) {
                            c_acc[r] = _mm256_setzero_ps(); // Set all elements to 0.0f
                        }

                        // Inner K loop: Computes dot products and accumulates results.
                        // UNROLL_K factor explicitly unrolls this loop to maximize instruction-level parallelism.
                        for (int kk = 0; kk < K_current_block_size; kk += UNROLL_K) {
                            for (int k_unroll = 0; k_unroll < UNROLL_K; ++k_unroll) {
                                int current_k = k_block + kk + k_unroll;
                                if (current_k >= K) break; // Global K-tail handling

                                // Prefetch data for A and B into L1/L2 cache.
                                // `_MM_HINT_T0` suggests prefetching to all levels of the cache hierarchy.
                                // Prefetching `UNROLL_K` elements ahead for A and `NR_AVX2` for B.
                                _mm_prefetch((const char*)(A + (i_block + i) * lda + current_k + UNROLL_K), _MM_HINT_T0);
                                _mm_prefetch((const char*)(B + (current_k + 1) * ldb + j_block + j + NR_AVX2), _MM_HINT_T0);

                                // Load B vector (NR_AVX2 floats) from `B[current_k][j_block+j : j_block+j+NR_AVX2]`.
                                // For N-tail (NR_actual < NR_AVX2), AVX2 lacks masked loads for floats.
                                // We use a scalar-to-buffer approach: load individual floats into an aligned buffer,
                                // then load a full vector from that buffer. This ensures correctness and alignment.
                                __m256 b_vec;
                                if (NR_actual == NR_AVX2) {
                                    b_vec = _mm256_loadu_ps(B + current_k * ldb + j_block + j); // Unaligned load
                                } else {
                                    alignas(32) float b_buffer[NR_AVX2] = {0.0f}; // Initialize to zero
                                    for(int col = 0; col < NR_actual; ++col) {
                                        b_buffer[col] = B[current_k * ldb + j_block + j + col];
                                    }
                                    b_vec = _mm256_load_ps(b_buffer); // Aligned load from buffer
                                }

                                // Perform Fused Multiply-Add (FMADD) operations for MR_AVX2 rows.
                                // `c_acc[r] = a_broadcast * b_vec + c_acc[r]`
                                for (int r = 0; r < MR_actual; ++r) { // Only iterate up to MR_actual for tail handling
                                    // Load A value (scalar) from `A[i_block+i+r][current_k]` and broadcast it across the vector.
                                    __m256 a_broadcast = _mm256_set1_ps(A[(i_block + i + r) * lda + current_k]);
                                    c_acc[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // Store accumulated results back to C.
                        for (int r = 0; r < MR_actual; ++r) { // Only iterate up to MR_actual for tail handling
                            float* C_ptr = C + (i_block + i + r) * ldc + j_block + j;
                            if (k_block == 0) {
                                // For the very first K-block iteration (`k_block == 0`), C is effectively C = A*B.
                                // We overwrite the pre-zeroed C matrix.
                                if (NR_actual == NR_AVX2) {
                                    _mm256_storeu_ps(C_ptr, c_acc[r]); // Unaligned store
                                } else {
                                    // Scalar store for C-tail (if N-dimension is not a multiple of NR_AVX2).
                                    alignas(32) float c_buffer[NR_AVX2];
                                    _mm256_store_ps(c_buffer, c_acc[r]); // Store vector to aligned buffer
                                    for (int col = 0; col < NR_actual; ++col) {
                                        C_ptr[col] = c_buffer[col]; // Store scalar elements
                                    }
                                }
                            } else {
                                // For subsequent K-blocks, we accumulate into existing C values (C += A*B).
                                if (NR_actual == NR_AVX2) {
                                    __m256 c_current = _mm256_loadu_ps(C_ptr);
                                    _mm256_storeu_ps(C_ptr, _mm256_add_ps(c_current, c_acc[r]));
                                } else {
                                    // Scalar load/store for C-tail accumulation.
                                    alignas(32) float c_current_buffer[NR_AVX2];
                                    alignas(32) float c_acc_buffer[NR_AVX2];
                                    // Load existing C values from memory
                                    for (int col = 0; col < NR_actual; ++col) {
                                        c_current_buffer[col] = C_ptr[col];
                                    }
                                    _mm256_store_ps(c_acc_buffer, c_acc[r]); // Store accumulator to buffer
                                    // Add and store back to C
                                    for (int col = 0; col < NR_actual; ++col) {
                                        C_ptr[col] = c_current_buffer[col] + c_acc_buffer[col];
                                    }
                                }
                            }
                        }
                    } // End j loop (over N-columns of C/B)
                } // End i loop (over M-rows of C/A)
            } // End k_block loop (over K-dimension of A/B)
        } // End j_block loop (over N-blocks of C)
    } // End i_block loop (over M-blocks of C)
}

// gemm_avx512: AVX-512 + FMA optimized GEMM implementation.
// Uses `__attribute__((target("avx512f,avx512dq,avx512vl")))` to enable AVX-512 F, DQ, and VL extensions.
// This allows for 512-bit vector operations and masked loads/stores.
void __attribute__((target("avx512f,avx512dq,avx512vl"))) gemm_avx512(const float* A, const float* B, float* C,
                                                                      int M, int N, int K,
                                                                      int lda, int ldb, int ldc) {
    // The overall blocking and multi-threading structure is similar to the AVX2 implementation.
    // The main differences are the use of `__m512` registers and masked operations for tail handling.
#pragma omp parallel for collapse(2) schedule(static)
    for (int i_block = 0; i_block < M; i_block += BM) {
        for (int j_block = 0; j_block < N; j_block += BN) {
            int M_current_block_size = std::min(BM, M - i_block);
            int N_current_block_size = std::min(BN, N - j_block);

            for (int k_block = 0; k_block < K; k_block += BK) {
                int K_current_block_size = std::min(BK, K - k_block);

                for (int i = 0; i < M_current_block_size; i += MR_AVX512) {
                    int MR_actual = std::min(MR_AVX512, M_current_block_size - i);

                    for (int j = 0; j < N_current_block_size; j += NR_AVX512) {
                        int NR_actual = std::min(NR_AVX512, N_current_block_size - j);

                        // Initialize AVX-512 accumulator registers (`__m512` holds 16 floats).
                        __m512 c_acc[MR_AVX512];
                        for (int r = 0; r < MR_AVX512; ++r) {
                            c_acc[r] = _mm512_setzero_ps(); // Set all elements to 0.0f
                        }

                        for (int kk = 0; kk < K_current_block_size; kk += UNROLL_K) {
                            for (int k_unroll = 0; k_unroll < UNROLL_K; ++k_unroll) {
                                int current_k = k_block + kk + k_unroll;
                                if (current_k >= K) break;

                                // Prefetch data for A and B.
                                _mm_prefetch((const char*)(A + (i_block + i) * lda + current_k + UNROLL_K), _MM_HINT_T0);
                                _mm_prefetch((const char*)(B + (current_k + 1) * ldb + j_block + j + NR_AVX512), _MM_HINT_T0);

                                // Load B vector using a masked load for N-tail.
                                // `(1U << NR_actual) - 1` creates a bitmask where the first `NR_actual` bits are set to 1.
                                // `_mm512_maskz_loadu_ps` loads elements where the corresponding mask bit is 1,
                                // and sets remaining elements to zero.
                                __m512 b_vec;
                                unsigned int mask = (1U << NR_actual) - 1;
                                b_vec = _mm512_maskz_loadu_ps(mask, B + current_k * ldb + j_block + j);

                                for (int r = 0; r < MR_actual; ++r) {
                                    // Load A value (scalar) and broadcast it.
                                    __m512 a_broadcast = _mm512_set1_ps(A[(i_block + i + r) * lda + current_k]);
                                    c_acc[r] = _mm512_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // Store accumulated results back to C, using masked stores for N-tail.
                        for (int r = 0; r < MR_actual; ++r) {
                            float* C_ptr = C + (i_block + i + r) * ldc + j_block + j;
                            unsigned int mask = (1U << NR_actual) - 1;
                            if (k_block == 0) {
                                // First K-block: C = A*B
                                _mm512_mask_storeu_ps(C_ptr, mask, c_acc[r]); // Masked store
                            } else {
                                // Subsequent K-blocks: C += A*B
                                // Load existing C values with a mask for correct accumulation at tails.
                                __m512 c_current = _mm512_maskz_loadu_ps(mask, C_ptr);
                                _mm512_mask_storeu_ps(C_ptr, mask, _mm512_add_ps(c_current, c_acc[r]));
                            }
                        }
                    } // End j loop
                } // End i loop
            } // End k_block loop
        } // End j_block loop
    } // End i_block loop
}


// gemm: Top-level GEMM function with runtime dispatch.
// This function acts as a dispatcher, selecting the most optimized SIMD kernel
// (AVX-512, AVX2, or scalar) available on the current CPU at runtime.
// It uses GCC/Clang's `__builtin_cpu_supports` for feature detection.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
#if defined(__GNUC__) || defined(__clang__)
    // Runtime dispatch checks for AVX-512 first, then AVX2, falling back to scalar.
    // For an AMD Ryzen 7 6800HS (Zen 3+), `__builtin_cpu_supports("avx512f")` will return false,
    // and `__builtin_cpu_supports("avx2")` will return true, thus the AVX2 kernel will be selected.
    if (__builtin_cpu_supports("avx512f")) {
        std::cout << "INFO: Using AVX-512 kernel.\n";
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    } else if (__builtin_cpu_supports("avx2")) {
        std::cout << "INFO: Using AVX2 kernel.\n";
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    } else {
        std::cout << "INFO: Using scalar fallback kernel (no AVX2/AVX-512 support detected).\n";
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
#else
    // Fallback for compilers that do not support `__builtin_cpu_supports`.
    // Given the `__attribute__((target(...)))` usage, this usually implies compilers other than GCC/Clang.
    std::cout << "INFO: Using scalar fallback kernel (compiler does not support __builtin_cpu_supports).\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}


// --- Main Function for Demo and Testing ---
int main(int argc, char* argv[]) {
    int M = 1024; // Default matrix M dimension
    int N = 1024; // Default matrix N dimension
    int K = 1024; // Default matrix K dimension (inner dimension)
    unsigned int seed = 42; // Default random seed for reproducibility
    int num_threads_set = 0; // 0 means use OpenMP's default number of threads
    bool dump_matrices = false; // Flag to enable/disable matrix dumping

    // Command-line argument parsing:
    // M, N, K are parsed as positional arguments, followed by optional flags.
    int current_arg_idx = 1;
    if (current_arg_idx < argc) {
        M = std::stoi(argv[current_arg_idx++]);
    }
    if (current_arg_idx < argc) {
        N = std::stoi(argv[current_arg_idx++]);
    }
    if (current_arg_idx < argc) {
        K = std::stoi(argv[current_arg_idx++]);
    }

    // Parse optional flags (e.g., -s <seed>, -t <threads>, --dump-matrices)
    for (int i = current_arg_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-s" && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            num_threads_set = std::stoi(argv[++i]);
        } else if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else {
            // Print usage information if an unknown or malformed argument is encountered.
            std::cerr << "Usage: " << argv[0] << " [M] [N] [K] [-s <seed>] [-t <threads>] [--dump-matrices]\n";
            return 1;
        }
    }

    // Set the number of OpenMP threads if specified via CLI.
    if (num_threads_set > 0) {
#ifdef _OPENMP
        omp_set_num_threads(num_threads_set);
#else
        std::cout << "WARNING: OpenMP not enabled during compilation. Thread count ignored.\n";
#endif
    }
    int actual_threads = omp_get_max_threads(); // Get the actual number of threads being used
    std::cout << "Running GEMM with M=" << M << ", N=" << N << ", K=" << K << ", Threads=" << actual_threads << "\n";

    // Allocate memory for matrices A, B, C, and a reference C_ref for correctness checking.
    // `_mm_malloc` is used for 64-byte alignment, which is beneficial for both AVX2 (32-byte)
    // and AVX-512 (64-byte) operations, preventing potential performance penalties from unaligned access.
    // For simplicity, we assume lda=K, ldb=N, ldc=N, representing dense row-major matrices.
    int lda = K; // Leading dimension for A (number of columns in a row)
    int ldb = N; // Leading dimension for B
    int ldc = N; // Leading dimension for C

    float* A = (float*)_mm_malloc(static_cast<size_t>(M) * K * sizeof(float), 64);
    float* B = (float*)_mm_malloc(static_cast<size_t>(K) * N * sizeof(float), 64);
    float* C = (float*)_mm_malloc(static_cast<size_t>(M) * N * sizeof(float), 64);
    float* C_ref = (float*)_mm_malloc(static_cast<size_t>(M) * N * sizeof(float), 64);

    // Check for successful memory allocation.
    if (!A || !B || !C || !C_ref) {
        std::cerr << "Error: Failed to allocate aligned memory.\n";
        _mm_free(A); _mm_free(B); _mm_free(C); _mm_free(C_ref); // Free any successfully allocated memory
        return 1;
    }

    // Initialize matrices A and B with random floating-point values between -1.0f and 1.0f.
    // C and C_ref are initialized to zeros, which is crucial for the first K-block accumulation
    // to correctly represent C = A*B (as opposed to C += A*B).
    std::mt19937 gen(seed); // Mersenne Twister pseudo-random number generator
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f); // Distribution for random floats

    for (int i = 0; i < M * K; ++i) A[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) B[i] = dis(gen);
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f);
    std::fill(C_ref, C_ref + static_cast<size_t>(M) * N, 0.0f);

    // If `--dump-matrices` flag is present, dump initial A and B matrices to files.
    if (dump_matrices) {
        std::filesystem::create_directories("workspace"); // Ensure "workspace" directory exists
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "INFO: Matrices A and B dumped to workspace/A.txt and workspace/B.txt\n";
    }

    // --- Performance Measurement ---
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc); // Call the top-level GEMM, which dispatches to the optimized kernel
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate execution duration and GFLOP/s.
    // GEMM involves 2 floating point operations (1 multiply, 1 add) per element of K.
    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double gflops = (2.0 * static_cast<double>(M) * N * K) / (duration_ms * 1e6);

    std::cout << "Computation finished in " << duration_ms << " ms\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOP/s\n";

    // If `--dump-matrices` flag is present, dump the computed C matrix to a file.
    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "INFO: Matrix C dumped to workspace/C.txt\n";
    }

    // --- Correctness Check ---
    std::cout << "Performing correctness check with scalar reference...\n";
    gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc); // Compute reference result

    float max_diff = 0.0f;
    for (int i = 0; i < static_cast<size_t>(M) * N; ++i) {
        float diff = std::abs(C[i] - C_ref[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    // Floating-point comparison tolerance.
    // A common tolerance for matrix multiplication is `epsilon * K * max_output_value`.
    // Here, `K` represents the number of accumulations, and `10.0f` is a small safety factor.
    // `max_output_value` is implicitly handled by `epsilon`'s relation to typical float ranges.
    float tolerance = std::numeric_limits<float>::epsilon() * static_cast<float>(K) * 10.0f;
    std::cout << "Max difference: " << std::fixed << std::setprecision(8) << max_diff << "\n";
    std::cout << "Tolerance: " << std::fixed << std::setprecision(8) << tolerance << "\n";

    if (max_diff > tolerance) {
        std::cerr << "ERROR: Results differ! Max diff exceeds tolerance.\n";
        // If correctness check fails, dump both optimized and scalar results for debugging,
        // unless they were already dumped by `--dump-matrices`.
        if (!dump_matrices) {
            std::filesystem::create_directories("workspace");
            write_matrix_to_file("workspace/C_optimized.txt", C, M, N, ldc);
            write_matrix_to_file("workspace/C_scalar_ref.txt", C_ref, M, N, ldc);
            std::cout << "INFO: Dumped C_optimized.txt and C_scalar_ref.txt for comparison.\n";
        }
        _mm_free(A); _mm_free(B); _mm_free(C); _mm_free(C_ref);
        return 1; // Indicate failure
    } else {
        std::cout << "Correctness check PASSED.\n";
    }

    // Free all allocated memory.
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_ref);

    return 0; // Indicate success
}