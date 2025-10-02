// Compile instructions:
// For AVX-512 (preferred for Intel 11th Gen i7, `x86-64-v3` implies AVX2/FMA; add `avx512f` for AVX-512):
// g++ -O3 -std=c++17 -march=x86-64-v3 -mavx512f -mfma -fopenmp -Wall -Wextra gemm.cpp -o gemm_avx512
// or using -march=native to automatically pick best architecture with AVX-512 support (if enabled):
// g++ -O3 -std=c++17 -march=native -fopenmp -Wall -Wextra gemm.cpp -o gemm_native

// For AVX2 fallback (if AVX-512 is not desired or available, e.g., older CPUs or specific Intel desktop CPUs):
// g++ -O3 -std=c++17 -march=x86-64-v2 -mavx2 -mfma -fopenmp -Wall -Wextra gemm.cpp -o gemm_avx2

// For a portable version without specific SIMD extensions (will use scalar or OpenMP parallelized scalar):
// g++ -O3 -std=c++17 -fopenmp -Wall -Wextra gemm.cpp -o gemm_portable

#include <immintrin.h> // For SIMD intrinsics (__m256, __m512, _mm256_*, _mm512_*, etc.)
#include <iostream>    // For input/output operations (std::cout, std::cerr)
#include <vector>      // For std::vector (though raw pointers with posix_memalign are used for matrices)
#include <cstring>     // For std::memset (to zero-initialize C)
#include <chrono>      // For timing performance (std::chrono)
#include <random>      // For random number generation (std::mt19937, std::uniform_real_distribution)
#include <cassert>     // For assert (though not strictly used in final code, good for debugging)
#include <string>      // For std::string
#include <fstream>     // For file operations (std::ofstream)
#include <filesystem>  // For directory creation (std::filesystem::create_directory, C++17 feature)
#include <algorithm>   // For std::min, std::max
#include <cmath>       // For std::abs (float version)
#include <stdexcept>   // For std::invalid_argument, std::out_of_range

#ifdef _OPENMP
#include <omp.h>       // For OpenMP parallelization directives
#endif

// --- Autotuning Parameters ---
// These parameters are chosen as a starting point, attempting to balance
// L1/L2 cache usage, register pressure, and OpenMP work distribution.
// Optimal values are highly dependent on the specific CPU model, workload, and compiler.

// Blocking factors for the M, N, K dimensions.
// These define the size of sub-matrices (tiles) processed by the main loops.
// They are chosen to promote data reuse in L2/L3 caches.
// BM: M-dimension block size (rows of A / C). Aim for L3 cache residence.
// BN: N-dimension block size (columns of B / C). Aim for L3 cache residence.
// BK: K-dimension block size (inner dimension for A and B). Aim for L2/L3 cache residence,
// this block of A and B is fully computed before moving to the next K-block.
constexpr int BM = 64;  // M-dimension block size (rows of A / C)
constexpr int BN = 128; // N-dimension block size (columns of B / C)
constexpr int BK = 256; // K-dimension block size (inner dimension for A and B)

// Micro-kernel unroll factors and register blocking.
// These define the innermost computational block, designed to fit into L1 cache and CPU registers.
// MR (M-Register): Number of rows of C computed concurrently in registers.
// UNROLL_K: Number of K-loop iterations unrolled in the micro-kernel.

// Vector sizes (number of float elements) for different SIMD ISAs
constexpr int AVX512_VEC_SIZE = 16; // __m512 holds 16 floats (16 * 4 bytes = 64 bytes)
constexpr int AVX2_VEC_SIZE = 8;    // __m256 holds 8 floats (8 * 4 bytes = 32 bytes)

// Parameters for AVX-512 micro-kernel
// MR_AVX512 rows * AVX512_VEC_SIZE cols of C are accumulated simultaneously in registers.
// Total accumulators: MR_AVX512 * AVX512_VEC_SIZE floats.
// For example, 4 rows * 16 floats/row = 64 floats (256 bytes) in registers for C.
constexpr int MR_AVX512 = 4;        
constexpr int UNROLL_K_AVX512 = 4;  // Unroll the K loop by 4 for better instruction-level parallelism

// Parameters for AVX2 micro-kernel
// MR_AVX2 rows * AVX2_VEC_SIZE cols of C are accumulated simultaneously in registers.
// Total accumulators: MR_AVX2 * AVX2_VEC_SIZE floats.
// For example, 4 rows * 8 floats/row = 32 floats (128 bytes) in registers for C.
constexpr int MR_AVX2 = 4;          
constexpr int UNROLL_K_AVX2 = 4;    

// --- Utility Function for Matrix Dumping ---
// Writes a matrix to a specified file in a space-separated format.
// Handles matrices stored in row-major order with a given leading dimension (stride).
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::filesystem::path dir("workspace");
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directory(dir);
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    // Write matrix elements, respecting the leading dimension
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}


// --- Scalar Reference GEMM Implementation ---
// This is a basic, unoptimized triple-loop implementation of C_new = C_old + A * B.
// It serves as a correctness reference and a fallback for systems without SIMD support.
// Matrix storage: A (M x K), B (K x N), C (M x N) are all assumed to be row-major.
// `lda`, `ldb`, `ldc` are the leading dimensions (strides) of A, B, C respectively.
// In the `main` function, C is initialized to zeros, so this effectively computes C = A*B.
// This implementation performs accumulation into C, suitable for C = alpha*A*B + beta*C.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float c_val = 0.0f; // Local accumulator for A*B product for this C[i][j]
            for (int k = 0; k < K; ++k) {
                c_val += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] += c_val; // Add the computed A*B product to the existing C value
        }
    }
}


// --- AVX2 GEMM Implementation ---
// This kernel uses AVX2 and FMA (Fused Multiply-Add) intrinsics for CPU-optimized GEMM.
// It employs cache-aware blocking (BM, BN, BK) and register blocking (MR_AVX2, UNROLL_K_AVX2)
// to maximize data reuse and instruction-level parallelism.
// It uses unaligned loads/stores (`_mm256_loadu_ps`, `_mm256_storeu_ps`) for flexibility
// with leading dimensions. Aligned memory allocation (64-byte) is used for base pointers.
// Row-major storage is assumed for A, B, and C. This implementation computes C_new = C_old + A*B.
// It relies on C being initialized to zero in `main()` for C = A*B semantics.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
#if defined(__AVX2__) && defined(__FMA__)
    // Outer loops for tiling over M and N dimensions.
    // OpenMP parallelization is applied here to distribute C-blocks among threads.
    // `collapse(2)` parallelizes both loops, `schedule(guided)` helps with load balancing
    // for varying block sizes and tails. Number of threads can be set by OMP_NUM_THREADS.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(guided)
#endif
    for (int i_block = 0; i_block < M; i_block += BM) { // Iterate over M-blocks of A and C
        for (int j_block = 0; j_block < N; j_block += BN) { // Iterate over N-blocks of B and C
            int current_M = std::min(BM, M - i_block); // Actual height of current C block
            int current_N = std::min(BN, N - j_block); // Actual width of current C block

            // K-block loop: This loop processes one K-block of A and B to update the current C-block.
            // This is crucial for L3 cache reuse: loading a block of A (current_M x current_K)
            // and B (current_K x current_N) into L3/L2, then computing the full C-block contribution.
            for (int k_block = 0; k_block < K; k_block += BK) {
                int current_K = std::min(BK, K - k_block); // Actual depth of current K block

                // Micro-kernel loops: These operate on smaller MR_AVX2 x AVX2_VEC_SIZE blocks of C.
                // Designed for L1 cache and register usage.
                for (int i = 0; i < current_M; i += MR_AVX2) { // Iterate over rows of A / C, step by MR_AVX2
                    for (int j = 0; j < current_N; j += AVX2_VEC_SIZE) { // Iterate over columns of B / C, step by AVX2_VEC_SIZE
                        // Initialize MR_AVX2 accumulators, each holding AVX2_VEC_SIZE (8) floats for C.
                        // c_acc[r] will hold C[i_block+i+r][j_block+j ... j_block+j+AVX2_VEC_SIZE-1]
                        __m256 c_acc[MR_AVX2];
                        for (int r = 0; r < MR_AVX2; ++r) {
                            c_acc[r] = _mm256_setzero_ps(); // Set accumulators to zero before accumulation
                        }

                        // Innermost K loop (micro-kernel K-dimension)
                        // Computes the contribution of a K_block from A and B to the current C micro-panel.
                        for (int k = 0; k < current_K; k += UNROLL_K_AVX2) {
                            // Loop for K-unrolling. This helps hide memory latency and utilize execution units.
                            for (int uk = 0; uk < UNROLL_K_AVX2; ++uk) {
                                int k_idx = k_block + k + uk; // Current K index in global K-dimension
                                if (k_idx >= K) break; // Handle K-tail if K is not a multiple of BK or UNROLL_K_AVX2

                                // Load a vector from B: B[k_idx][j_block+j ... j_block+j+AVX2_VEC_SIZE-1]
                                // _mm256_loadu_ps allows unaligned access.
                                const float* B_ptr = &B[k_idx * ldb + (j_block + j)];
                                // Prefetch B_ptr for next iteration or later. _MM_HINT_T0 to L1 cache.
                                _mm_prefetch((char*)(B_ptr + AVX2_VEC_SIZE), _MM_HINT_T0); 
                                __m256 b_vec = _mm256_loadu_ps(B_ptr);

                                // For each row in the C micro-panel (MR_AVX2 rows)
                                for (int r = 0; r < MR_AVX2; ++r) {
                                    int a_row_idx = i_block + i + r; // Current A/C row index in global M-dimension
                                    if (a_row_idx >= M) { // Handle M-tail if M is not a multiple of BM or MR_AVX2
                                        break; // If this row is beyond actual M, break from this inner loop.
                                    }
                                    // Load a scalar from A: A[a_row_idx][k_idx]
                                    // _mm256_set1_ps broadcasts this scalar to all elements of a vector.
                                    const float a_scalar_val = A[a_row_idx * lda + k_idx];
                                    // Prefetch A[a_row_idx * lda + k_idx + UNROLL_K_AVX2] if within K boundary.
                                    _mm_prefetch((char*)&A[a_row_idx * lda + k_idx + UNROLL_K_AVX2], _MM_HINT_T0);
                                    __m256 a_scalar_vec = _mm256_set1_ps(a_scalar_val);

                                    // Fused Multiply-Add: c_acc[r] = a_scalar_vec * b_vec + c_acc[r]
                                    c_acc[r] = _mm256_fmadd_ps(a_scalar_vec, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // Store accumulators back to C.
                        // This part performs C_new = C_old + A_block * B_block_partial.
                        // As 'c_acc' holds the sum over the current K-block's contribution for a micro-panel,
                        // and 'existing_c' loads the current C values for that micro-panel,
                        // this correctly sums contributions from different K-blocks into C.
                        // Since C is initialized to zero in main, the overall effect is C = A*B.
                        int cols_to_store = std::min(AVX2_VEC_SIZE, current_N - j);
                        for (int r = 0; r < MR_AVX2; ++r) {
                            int c_row_idx = i_block + i + r;
                            if (c_row_idx >= M) break; // M-tail handling
                            
                            float* c_ptr = &C[c_row_idx * ldc + (j_block + j)];
                            
                            if (cols_to_store == AVX2_VEC_SIZE) { // Full vector operation
                                __m256 existing_c = _mm256_loadu_ps(c_ptr); // Load existing C values (C_old)
                                c_acc[r] = _mm256_add_ps(c_acc[r], existing_c); // Add new contribution (c_acc[r] is A*B_partial, existing_c is C_old)
                                _mm256_storeu_ps(c_ptr, c_acc[r]); // Store back C_new
                            } else { // N-tail: scalar copy and add
                                // Store the accumulated vector to a temporary aligned buffer.
                                alignas(32) float temp_acc_buf[AVX2_VEC_SIZE];
                                _mm256_store_ps(temp_acc_buf, c_acc[r]); 
                                
                                // Perform scalar add for only the valid elements.
                                for (int col = 0; col < cols_to_store; ++col) {
                                    c_ptr[col] += temp_acc_buf[col]; 
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else // __AVX2__ not defined at compile time
    // Fallback to scalar GEMM if AVX2 intrinsics are not enabled by the compiler.
    // This provides robustness for compilers/platforms not supporting AVX2.
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}


// --- AVX-512 GEMM Implementation ---
// This kernel uses AVX-512 and FMA intrinsics for maximum performance on compatible CPUs.
// The blocking strategy is similar to AVX2, but it utilizes wider 512-bit vectors
// (AVX512_VEC_SIZE = 16 floats) and AVX-512 specific masked operations for efficient tail handling.
// Row-major storage is assumed for A, B, and C. This implementation computes C_new = C_old + A*B.
// It relies on C being initialized to zero in `main()` for C = A*B semantics.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__) && defined(__FMA__)
    // Outer loops for tiling, parallelized with OpenMP.
#ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(guided)
#endif
    for (int i_block = 0; i_block < M; i_block += BM) {
        for (int j_block = 0; j_block < N; j_block += BN) {
            int current_M = std::min(BM, M - i_block);
            int current_N = std::min(BN, N - j_block);

            for (int k_block = 0; k_block < K; k_block += BK) {
                int current_K = std::min(BK, K - k_block);

                for (int i = 0; i < current_M; i += MR_AVX512) {
                    for (int j = 0; j < current_N; j += AVX512_VEC_SIZE) {
                        // Calculate mask for the current vector load/store at C[i][j...j+AVX512_VEC_SIZE-1]
                        // This mask needs to be recalculated for each 'j' loop iteration.
                        // `(current_N - j)` gives the number of valid columns remaining in the current N-block.
                        __mmask16 n_mask_current_vec = (j + AVX512_VEC_SIZE <= current_N) ? (__mmask16)0xFFFF : (__mmask16)((1 << (current_N - j)) - 1);

                        // Initialize MR_AVX512 accumulators, each holding AVX512_VEC_SIZE (16) floats.
                        __m512 c_acc[MR_AVX512];
                        for (int r = 0; r < MR_AVX512; ++r) {
                            c_acc[r] = _mm512_setzero_ps(); // Set accumulators to zero
                        }

                        // Innermost K loop
                        for (int k = 0; k < current_K; k += UNROLL_K_AVX512) {
                            for (int uk = 0; uk < UNROLL_K_AVX512; ++uk) {
                                int k_idx = k_block + k + uk;
                                if (k_idx >= K) break; // K-tail handling

                                // Load a vector from B, using the mask for partial loads at the N-tail.
                                // _mm512_maskz_loadu_ps loads zeros for masked-out elements, ensuring correctness.
                                const float* B_ptr = &B[k_idx * ldb + (j_block + j)];
                                _mm_prefetch((char*)(B_ptr + AVX512_VEC_SIZE), _MM_HINT_T0); 
                                __m512 b_vec = _mm512_maskz_loadu_ps(n_mask_current_vec, B_ptr);

                                // For each row in the C micro-panel
                                for (int r = 0; r < MR_AVX512; ++r) {
                                    int a_row_idx = i_block + i + r;
                                    if (a_row_idx >= M) { // M-tail handling
                                        break; 
                                    }
                                    // Load a scalar from A and broadcast to a 512-bit vector.
                                    const float a_scalar_val = A[a_row_idx * lda + k_idx];
                                    _mm_prefetch((char*)&A[a_row_idx * lda + k_idx + UNROLL_K_AVX512], _MM_HINT_T0);
                                    __m512 a_scalar_vec = _mm512_set1_ps(a_scalar_val);
                                    
                                    // Fused Multiply-Add
                                    c_acc[r] = _mm512_fmadd_ps(a_scalar_vec, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // Store accumulators back to C.
                        // This part performs C_new = C_old + A_block * B_block_partial.
                        // As 'c_acc' holds the sum over the current K-block's contribution for a micro-panel,
                        // and 'existing_c' loads the current C values for that micro-panel,
                        // this correctly sums contributions from different K-blocks into C.
                        // Since C is initialized to zero in main, the overall effect is C = A*B.
                        for (int r = 0; r < MR_AVX512; ++r) {
                            int c_row_idx = i_block + i + r;
                            if (c_row_idx >= M) break; // M-tail

                            float* c_ptr = &C[c_row_idx * ldc + (j_block + j)];
                            
                            // Load existing C values for the current micro-panel, applying the mask.
                            // `_mm512_maskz_loadu_ps` loads existing C and zeroes out elements beyond `current_N`.
                            // This ensures that unmasked C elements (padding) are loaded as 0.0f and not garbage.
                            __m512 existing_c = _mm512_maskz_loadu_ps(n_mask_current_vec, c_ptr);
                            
                            // Add the accumulated contribution from c_acc to existing_c.
                            __m512 final_c_val = _mm512_add_ps(c_acc[r], existing_c);
                            
                            // Store the result back to C, using the mask to write only valid elements.
                            _mm512_mask_storeu_ps(c_ptr, n_mask_current_vec, final_c_val);
                        }
                    }
                }
            }
        }
    }
#else // __AVX512F__ not defined at compile time
    // Fallback to scalar GEMM if AVX-512 intrinsics are not enabled.
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}


// --- Top-level GEMM Dispatcher ---
// This function acts as the public API for GEMM. It performs runtime detection
// of available CPU features (AVX-512, AVX2) using `__builtin_cpu_supports` (GCC/Clang extension)
// and dispatches the call to the most optimized kernel available.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

    // Use a static function pointer to cache the dispatch decision.
    // This avoids redundant CPU feature checks on subsequent calls.
    static void (*gemm_kernel_ptr)(const float*, const float*, float*, int, int, int, int, int, int) = nullptr;
    static std::string kernel_name; // To store the name of the chosen kernel

    // If the kernel pointer hasn't been set, determine the best kernel.
    if (gemm_kernel_ptr == nullptr) {
#if defined(__GNUC__) || defined(__clang__)
        // GCC/Clang specific: check CPU features at runtime
        // The target CPU (i7-1195G7) supports AVX-512 (though sometimes disabled by BIOS on consumer boards).
        // It definitely supports AVX2 and FMA.
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
            gemm_kernel_ptr = &gemm_avx512;
            kernel_name = "AVX-512 + FMA (Runtime)";
        } else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
            gemm_kernel_ptr = &gemm_avx2;
            kernel_name = "AVX2 + FMA (Runtime)";
        } else {
            gemm_kernel_ptr = &gemm_scalar;
            kernel_name = "Scalar (Runtime)";
        }
#else
        // For other compilers or if __builtin_cpu_supports is not available,
        // rely on compile-time definitions (e.g., -mavx512f, -mavx2 flags).
#if defined(__AVX512F__) && defined(__FMA__)
        gemm_kernel_ptr = &gemm_avx512;
        kernel_name = "AVX-512 + FMA (Compile-time)";
#elif defined(__AVX2__) && defined(__FMA__)
        gemm_kernel_ptr = &gemm_avx2;
        kernel_name = "AVX2 + FMA (Compile-time)";
#else
        gemm_kernel_ptr = &gemm_scalar;
        kernel_name = "Scalar (Compile-time)";
#endif
#endif
        std::cout << "Dispatched GEMM kernel: " << kernel_name << std::endl;
    }
    
    // Call the selected (and cached) kernel
    gemm_kernel_ptr(A, B, C, M, N, K, lda, ldb, ldc);
}


// --- Main Function for Demonstration and Testing ---
// This `main` function serves as a command-line interface for running,
// timing, and optionally verifying the GEMM implementation.
int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int seed = 42; // Seed for random number generation
    int num_threads = 0; // 0 means OpenMP default or system default
    bool dump_matrices = false; // Flag to dump matrices to files
    bool run_scalar_check = false; // Flag to run a correctness check against scalar GEMM

    // Positional argument counters for M, N, K
    int positional_arg_count = 0;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        try {
            if (arg[0] == '-') { // It's a flag argument
                if (arg == "-M" && i + 1 < argc) {
                    M = std::stoi(argv[++i]);
                } else if (arg == "-N" && i + 1 < argc) {
                    N = std::stoi(argv[++i]);
                } else if (arg == "-K" && i + 1 < argc) {
                    K = std::stoi(argv[++i]);
                } else if (arg == "--seed" && i + 1 < argc) {
                    seed = std::stoi(argv[++i]);
                } else if (arg == "--threads" && i + 1 < argc) {
                    num_threads = std::stoi(argv[++i]);
                } else if (arg == "--dump-matrices") {
                    dump_matrices = true;
                } else if (arg == "--check") {
                    run_scalar_check = true;
                } else if (arg == "-h" || arg == "--help") {
                    std::cout << "Usage: " << argv[0] << " [M N K] [-M <rows>] [-N <cols>] [-K <inner>] [--seed <val>] [--threads <num>] [--dump-matrices] [--check]\n";
                    std::cout << "  [M N K]           : Positional arguments for M, N, K. These can be overridden by flags.\n";
                    std::cout << "  -M <rows>         : Number of rows in A and C (default: 1024)\n";
                    std::cout << "  -N <cols>         : Number of columns in B and C (default: 1024)\n";
                    std::cout << "  -K <inner>        : Inner dimension for A and B (default: 1024)\n";
                    std::cout << "  --seed <val>      : Seed for random matrix initialization (default: 42)\n";
                    std::cout << "  --threads <num>   : Number of OpenMP threads to use (default: 0, uses OMP_NUM_THREADS or system default)\n";
                    std::cout << "  --dump-matrices   : Write A.txt, B.txt, C.txt to 'workspace/' directory.\n";
                    std::cout << "  --check           : Run scalar GEMM for correctness verification.\n";
                    return 0;
                } else { // Unknown flag or missing value for flag
                    std::cerr << "Error: Unknown flag or missing value for flag: " << arg << std::endl;
                    return 1;
                }
            } else { // It's a non-flag (positional) argument
                if (positional_arg_count == 0) {
                    M = std::stoi(arg);
                } else if (positional_arg_count == 1) {
                    N = std::stoi(arg);
                } else if (positional_arg_count == 2) {
                    K = std::stoi(arg);
                } else {
                    std::cerr << "Error: Too many positional arguments. Unexpected: " << arg << std::endl;
                    return 1;
                }
                positional_arg_count++;
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid numeric argument for '" << arg << "': " << e.what() << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cerr << "Error: Numeric argument out of range for '" << arg << "': " << e.what() << std::endl;
            return 1;
        }
    }

    // Set OpenMP threads if specified via CLI
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << std::endl;
#else
    if (num_threads > 0) {
        std::cerr << "Warning: OpenMP not enabled, --threads argument ignored." << std::endl;
    }
    std::cout << "OpenMP not enabled. Running on a single thread or system default." << std::endl;
#endif

    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Define leading dimensions. For row-major, this is simply the number of columns.
    int lda = K; // A is M x K
    int ldb = N; // B is K x N
    int ldc = N; // C is M x N

    float* A_data;
    float* B_data;
    float* C_data;
    float* C_ref_data = nullptr; // Used only if correctness check is enabled

    // Allocate matrices using posix_memalign for 64-byte alignment.
    // This is beneficial for AVX-512 even with unaligned loads/stores as it
    // guarantees the base address is cache-line aligned.
    // Alignment of 64 bytes is suitable for both AVX2 (32-byte vectors) and AVX-512 (64-byte vectors).
    if (posix_memalign((void**)&A_data, 64, (size_t)M * lda * sizeof(float)) != 0 ||
        posix_memalign((void**)&B_data, 64, (size_t)K * ldb * sizeof(float)) != 0 ||
        posix_memalign((void**)&C_data, 64, (size_t)M * ldc * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory. Exiting." << std::endl;
        return 1;
    }

    // Initialize matrices A and B with random float values, C with zeros.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            A_data[i * lda + k] = dis(gen);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            B_data[k * ldb + j] = dis(gen);
        }
    }
    std::memset(C_data, 0, (size_t)M * ldc * sizeof(float)); // Initialize C to zero

    // Dump initial matrices A and B if requested
    if (dump_matrices) {
        write_matrix_to_file("workspace/A.txt", A_data, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B_data, K, N, ldb);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt\n";
    }

    // Measure performance of the optimized GEMM function
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A_data, B_data, C_data, M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end_time - start_time;
    double time_ms = diff.count() * 1000.0;
    
    // Calculate GFLOP/s (2 * M * N * K operations for one GEMM: M*N*K multiplications + M*N*K additions)
    long long flops = 2LL * M * N * K;
    double gflops = static_cast<double>(flops) / (time_ms * 1e6);

    std::cout << "Optimized GEMM time: " << time_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    // Dump final matrix C if requested
    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C_data, M, N, ldc);
        std::cout << "Matrix C dumped to workspace/C.txt\n";
    }

    // Run correctness check against scalar implementation if requested
    if (run_scalar_check) {
        std::cout << "Running scalar GEMM for correctness check...\n";
        if (posix_memalign((void**)&C_ref_data, 64, (size_t)M * ldc * sizeof(float)) != 0) {
            std::cerr << "Failed to allocate memory for C_ref. Skipping check.\n";
            // Proceed to free and exit, without checking
        } else {
            // Initialize reference C to zero, just like the optimized version,
            // so `gemm_scalar` also computes C=A*B from a zero C.
            std::memset(C_ref_data, 0, (size_t)M * ldc * sizeof(float)); 
            
            auto start_scalar_time = std::chrono::high_resolution_clock::now();
            gemm_scalar(A_data, B_data, C_ref_data, M, N, K, lda, ldb, ldc);
            auto end_scalar_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff_scalar = end_scalar_time - start_scalar_time;
            std::cout << "Scalar GEMM time: " << diff_scalar.count() * 1000.0 << " ms\n";

            // Compare results with a tolerance
            float max_abs_diff = 0.0f;
            float max_rel_diff = 0.0f;
            // The problem uses floats, so numerical stability can be an issue.
            // These tolerances are typical for floating-point comparisons.
            // Due to different summation order in SIMD vs scalar, results may differ slightly.
            // A higher tolerance might be needed for very large K.
            float epsilon = 1e-3f;     // Absolute tolerance for small values (e.g., if true value is near zero)
            float rel_epsilon = 1e-2f;  // Relative tolerance for larger values (percentage of magnitude)
            int error_count = 0;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float val_opt = C_data[i * ldc + j];
                    float val_ref = C_ref_data[i * ldc + j];
                    float abs_diff = std::abs(val_opt - val_ref);
                    
                    max_abs_diff = std::max(max_abs_diff, abs_diff);

                    // Calculate relative difference, avoiding division by zero
                    float denominator = std::max({std::abs(val_opt), std::abs(val_ref), 1e-9f}); 
                    float rel_diff = abs_diff / denominator;
                    max_rel_diff = std::max(max_rel_diff, rel_diff);

                    if (abs_diff > epsilon && rel_diff > rel_epsilon) {
                        error_count++;
                        // Optionally print first few errors:
                        // if (error_count <= 5) {
                        //     std::cerr << "Mismatch at C[" << i << "][" << j << "]: Opt=" << val_opt
                        //               << ", Ref=" << val_ref << ", AbsDiff=" << abs_diff << ", RelDiff=" << rel_diff << std::endl;
                        // }
                    }
                }
            }
            std::cout << "Maximum absolute difference: " << max_abs_diff << std::endl;
            std::cout << "Maximum relative difference: " << max_rel_diff << std::endl;
            if (error_count == 0) {
                std::cout << "Correctness check PASSED (within abs tolerance " << epsilon << " and rel tolerance " << rel_epsilon << ")\n";
            } else {
                std::cout << "Correctness check FAILED: " << error_count << " discrepancies found.\n";
            }
        }
    }

    // Free all allocated memory
    free(A_data);
    free(B_data);
    free(C_data);
    if (C_ref_data) {
        free(C_ref_data);
    }

    return 0;
}