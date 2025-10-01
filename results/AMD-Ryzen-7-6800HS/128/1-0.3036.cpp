// Compile commands for GCC/Clang:
//
// 1. For AVX-512 (if available, e.g., on newer Intel CPUs. Note: AMD Ryzen 7 6800HS does NOT support AVX-512):
//    g++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//
// 2. For AVX2 + FMA (RECOMMENDED for AMD Ryzen 7 6800HS):
//    g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// 3. Portable (let compiler detect and use best available ISA. Will likely pick AVX2 on Ryzen 7 6800HS):
//    g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native
//
// Notes:
// - The AMD Ryzen 7 6800HS supports AVX, AVX2, and FMA but NOT AVX-512.
// - The code includes an AVX-512 kernel as requested. Runtime dispatch will correctly
//   fall back to the AVX2 kernel on the target platform if AVX-512 is not detected.
// - C++17 or later is assumed.

#include <iostream>    // For input/output (e.g., std::cout, std::cerr)
#include <vector>      // For std::vector
#include <cstring>     // For std::memset
#include <chrono>      // For high-resolution timing
#include <random>      // For random number generation (std::mt19937, std::uniform_real_distribution)
#include <cassert>     // For assert()
#include <string>      // For std::string
#include <fstream>     // For file operations (std::ofstream)
#include <algorithm>   // For std::min, std::max
#include <cmath>       // For std::abs

// Intrinsics specific headers for x86-64 SIMD
#if defined(__GNUC__) || defined(__clang__)
#include <immintrin.h> // Provides _mm256, _mm512, and associated intrinsics
#else
// Fallback for other compilers, e.g., MSVC requires specific headers.
// For this problem, GCC/Clang on Linux is assumed.
#endif

// OpenMP for multi-threading
#ifdef _OPENMP
#include <omp.h>
#else
// Define dummy OpenMP functions/macros if OpenMP is not available
// This allows the code to compile and run sequentially without OpenMP.
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {} // Dummy function
#endif

// --- Autotuning Parameters ---
// These parameters define the cache-aware tiling strategy.
// They are chosen as reasonable defaults for modern x86-64 CPUs like AMD Ryzen.
// The goal is to keep data blocks (A, B, C) resident in L1/L2 caches during computation.
// L1d cache (32KB/core): Smaller blocks (e.g., micro-kernel data).
// L2 cache (512KB/core): Larger blocks, allowing more data reuse for A, B.
// L3 cache (16MB shared): Even larger blocks if needed.

// BM: Block size for M dimension (rows of C). Affects C and A block size.
// BN: Block size for N dimension (columns of C). Affects C and B block size.
// BK: Block size for K dimension (inner product dimension). Affects A and B block size.
// These values are often good starting points, balancing register, L1, and L2 cache usage.
constexpr int DEFAULT_BM = 96;  // M-block size. Chosen as a multiple of MR.
constexpr int DEFAULT_BN = 64;  // N-block size. Chosen as a multiple of NR.
constexpr int DEFAULT_BK = 128; // K-block size.
constexpr int DEFAULT_UNROLL_K = 4; // Inner K-loop unroll factor (compiler-dependent without explicit packing).
                                    // For this implementation, the innermost k_micro loop relies on compiler unrolling.

// Micro-kernel dimensions (register blocking factors)
// These define how many rows (MR) and columns (NR) of C are accumulated concurrently
// using SIMD registers for a single k-loop iteration. NR is typically the vector width.
#if defined(__AVX512F__)
constexpr int FLOAT_VEC_WIDTH_AVX512 = 16; // Number of floats in __m512 (512 bits / 32 bits per float)
constexpr int MR_AVX512 = 4;               // Rows of C computed per micro-kernel iteration (e.g., 4 `__m512` accumulators)
constexpr int NR_AVX512 = FLOAT_VEC_WIDTH_AVX512; // Columns of C computed per micro-kernel iteration (vector length)
#else
// Fallback definitions if AVX-512 is not enabled at compile time
constexpr int FLOAT_VEC_WIDTH_AVX512 = 1;
constexpr int MR_AVX512 = 1;
constexpr int NR_AVX512 = 1;
#endif

#if defined(__AVX2__)
constexpr int FLOAT_VEC_WIDTH_AVX2 = 8;    // Number of floats in __m256 (256 bits / 32 bits per float)
constexpr int MR_AVX2 = 4;                 // Rows of C computed per micro-kernel iteration (e.g., 4 `__m256` accumulators)
constexpr int NR_AVX2 = FLOAT_VEC_WIDTH_AVX2; // Columns of C computed per micro-kernel iteration (vector length)
#else
// Fallback definitions if AVX2 is not enabled at compile time
constexpr int FLOAT_VEC_WIDTH_AVX2 = 1;
constexpr int MR_AVX2 = 1;
constexpr int NR_AVX2 = 1;
#endif


// --- Memory Management Helper Functions ---

// Function to allocate aligned memory. Required for optimal SIMD performance.
// SIMD instructions perform best with aligned data, minimizing cache misses and unaligned penalties.
void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#elif defined(__GNUC__) || defined(__clang__)
    void* ptr;
    // posix_memalign requires alignment to be a power of 2
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#else
    // Fallback for other compilers, might not be aligned as requested
    return std::malloc(size);
#endif
}

// Function to free aligned memory.
void aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// --- File I/O Helper Functions ---

// Helper function to create a directory. Used for dumping matrices.
// This is a platform-dependent call.
void create_directory(const std::string& path) {
    // For simplicity, using system command. A more robust solution would use OS-specific APIs (e.g., <filesystem> in C++17, but requires linking).
    // The -p flag ensures parent directories are created and no error if directory exists.
#ifdef _MSC_VER
    _mkdir(path.c_str()); // Requires <direct.h>
#else
    system(("mkdir -p " + path).c_str());
#endif
}

// Helper function to write a matrix to a text file.
// Assumes row-major storage for the conceptual matrix, using 'ld' as the leading dimension (stride).
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// --- GEMM Kernel Implementations ---

// All GEMM kernels assume row-major storage for A, B, and C matrices.
// 'lda', 'ldb', 'ldc' specify the leading dimension (stride) for each matrix.
// For a dense row-major matrix with dimensions R x C, the leading dimension
// is typically C (number of columns).

// 1. Scalar Reference GEMM Implementation
// This is a basic triple-nested loop implementation, primarily used for correctness checking
// and as a fallback if no SIMD support is available.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) { // Loop over rows of C (M)
        for (int j = 0; j < N; ++j) { // Loop over columns of C (N)
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) { // Inner loop for dot product (K)
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// 2. AVX2 + FMA Optimized GEMM Implementation
// This kernel utilizes AVX2 intrinsics and FMA instructions for efficient computation.
// It employs cache-aware tiling and OpenMP for parallelization.
#if defined(__AVX2__) && defined(__FMA__)
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    // Blocking parameters (tuned for cache usage and thread concurrency)
    // These define the sizes of blocks processed by a thread or within cache levels.
    const int BM = DEFAULT_BM; // M-block size: for L2/L3 cache tiling
    const int BN = DEFAULT_BN; // N-block size: for L2/L3 cache tiling
    const int BK = DEFAULT_BK; // K-block size: for L2/L3 cache tiling

    // Micro-kernel dimensions (register blocking)
    // MR rows x NR columns of C are computed simultaneously in registers.
    const int MR = MR_AVX2; // Number of rows of C computed concurrently (e.g., 4)
    const int NR = NR_AVX2; // Number of columns of C computed concurrently (vector width, 8 for AVX2)

    // OpenMP parallel region for outer blocks (M x N).
    // `schedule(static)` is chosen for its simplicity and good performance on uniform workloads.
    // `collapse(2)` allows OpenMP to parallelize both `m_start` and `n_start` loops, providing more work items.
    // Each thread processes distinct blocks of the C matrix, ensuring thread-safe writes.
    // The `std::min` calls handle tail cases where matrix dimensions are not multiples of block sizes.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_start = 0; m_start < M; m_start += BM) {
        for (int n_start = 0; n_start < N; n_start += BN) {
            // Iterate over K dimension with BK granularity
            for (int k_start = 0; k_start < K; k_start += BK) {
                // Determine actual block sizes for current iteration, handling tail cases
                int M_block = std::min(BM, M - m_start);
                int N_block = std::min(BN, N - n_start);
                int K_block = std::min(BK, K - k_start);

                // Iterate over M dimension with MR (micro-kernel row count) granularity
                for (int i = 0; i < M_block; i += MR) {
                    // Iterate over N dimension with NR (vector width) granularity
                    for (int j = 0; j < N_block; j += NR) {
                        // Current output C sub-block: C[m_start+i .. m_start+i+MR-1][n_start+j .. n_start+j+NR-1]

                        // Accumulators for C sub-block (MR rows x NR columns).
                        // Each __m256 register holds NR floats (NR=8 for AVX2).
                        __m256 c_acc[MR];

                        // Initialize accumulators: load existing C values into accumulators.
                        // For C = A*B, the main function should pre-zero C.
                        // For C = A*B + C_initial, C contains initial values.
                        // The loop over K blocks means C needs to accumulate, so we load.
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < M_block) { // M-tail handling: Only initialize if the row is within bounds.
                                float* C_ptr = &C[(m_start + i + r) * ldc + (n_start + j)];
                                if (N_block - j < NR) { // N-tail handling for initial load using mask.
                                    // Create a bitmask for the remaining N elements (N_block - j)
                                    // For AVX2, _mm256_maskload_ps expects a __m256i mask where
                                    // each int element is -1 for active float element, 0 otherwise.
                                    unsigned int mask_val_bits = (1U << (N_block - j)) - 1;
                                    __m256i load_mask = _mm256_set_epi32(
                                        (mask_val_bits & (1<<7)) ? -1 : 0, (mask_val_bits & (1<<6)) ? -1 : 0,
                                        (mask_val_bits & (1<<5)) ? -1 : 0, (mask_val_bits & (1<<4)) ? -1 : 0,
                                        (mask_val_bits & (1<<3)) ? -1 : 0, (mask_val_bits & (1<<2)) ? -1 : 0,
                                        (mask_val_bits & (1<<1)) ? -1 : 0, (mask_val_bits & (1<<0)) ? -1 : 0
                                    );
                                    c_acc[r] = _mm256_maskload_ps(C_ptr, load_mask);
                                } else {
                                    c_acc[r] = _mm256_loadu_ps(C_ptr); // Load existing C values (unaligned load)
                                }
                            } else {
                                c_acc[r] = _mm256_setzero_ps(); // If row is out of M_block, ensure accumulator is zero
                            }
                        }

                        // Inner K loop: Performs the actual matrix multiplication for the sub-block
                        // This loop iterates over the K_block, performing vector-scalar products.
                        for (int k_micro = 0; k_micro < K_block; ++k_micro) {
                            // Prefetch hints for the next A and B elements/vectors can be added here.
                            // This might not always be needed for well-tuned blocking, but can help reduce memory latency.
                            // _MM_HINT_T0: Fetch to all levels of the cache hierarchy.
                            // _mm_prefetch((const char*)&A[(m_start + i) * lda + (k_start + k_micro + 1)], _MM_HINT_T0);
                            // _mm_prefetch((const char*)&B[(k_start + k_micro + 1) * ldb + (n_start + j)], _MM_HINT_T0);

                            // Load B values: B[k_start+k_micro][n_start+j .. n_start+j+NR-1]
                            // These are vector loads of 8 floats.
                            const float* B_ptr = &B[(k_start + k_micro) * ldb + (n_start + j)];
                            
                            __m256 b_vec;
                            // N-tail handling for B: use masked load if the vector extends past N_block boundary
                            if (N_block - j < NR) {
                                unsigned int mask_val_bits = (1U << (N_block - j)) - 1;
                                __m256i load_mask = _mm256_set_epi32(
                                    (mask_val_bits & (1<<7)) ? -1 : 0, (mask_val_bits & (1<<6)) ? -1 : 0,
                                    (mask_val_bits & (1<<5)) ? -1 : 0, (mask_val_bits & (1<<4)) ? -1 : 0,
                                    (mask_val_bits & (1<<3)) ? -1 : 0, (mask_val_bits & (1<<2)) ? -1 : 0,
                                    (mask_val_bits & (1<<1)) ? -1 : 0, (mask_val_bits & (1<<0)) ? -1 : 0
                                );
                                b_vec = _mm256_maskload_ps(B_ptr, load_mask);
                            } else {
                                b_vec = _mm256_loadu_ps(B_ptr); // Unaligned load is typically efficient with AVX2
                            }

                            // Perform MR FMA operations for each row of the C sub-block
                            for (int r = 0; r < MR; ++r) {
                                if (i + r < M_block) { // M-tail handling: Only process rows within the M_block
                                    // Load A scalar and broadcast to all elements of a __m256 register
                                    // A value: A[m_start+i+r][k_start+k_micro]
                                    __m256 a_scalar_vec = _mm256_broadcast_ss(&A[(m_start + i + r) * lda + (k_start + k_micro)]);
                                    
                                    // Fused Multiply-Add (FMA): c_acc[r] = a_scalar_vec * b_vec + c_acc[r]
                                    // This is the core computation, accumulating results. FMA combines multiplication
                                    // and addition into a single instruction, improving throughput and precision.
                                    c_acc[r] = _mm256_fmadd_ps(a_scalar_vec, b_vec, c_acc[r]);
                                }
                            }
                        } // end k_micro loop

                        // Store accumulated results back to C matrix
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < M_block) { // M-tail handling
                                float* C_ptr = &C[(m_start + i + r) * ldc + (n_start + j)];
                                if (N_block - j < NR) { // N-tail handling for final store using mask
                                    unsigned int mask_val_bits = (1U << (N_block - j)) - 1;
                                    __m256i store_mask = _mm256_set_epi32(
                                        (mask_val_bits & (1<<7)) ? -1 : 0, (mask_val_bits & (1<<6)) ? -1 : 0,
                                        (mask_val_bits & (1<<5)) ? -1 : 0, (mask_val_bits & (1<<4)) ? -1 : 0,
                                        (mask_val_bits & (1<<3)) ? -1 : 0, (mask_val_bits & (1<<2)) ? -1 : 0,
                                        (mask_val_bits & (1<<1)) ? -1 : 0, (mask_val_bits & (1<<0)) ? -1 : 0
                                    );
                                    _mm256_maskstore_ps(C_ptr, store_mask, c_acc[r]);
                                } else {
                                    _mm256_storeu_ps(C_ptr, c_acc[r]); // Unaligned store
                                }
                            }
                        }
                    } // end j loop
                } // end i loop
            } // end k_start loop
        } // end n_start loop
    } // end m_start loop (OpenMP parallel for)
}
#else // If AVX2 and FMA are not available at compile time, define a dummy function
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    // This warning indicates that the AVX2 kernel was called, but the compiler was not
    // instructed to enable AVX2 intrinsics (e.g., missing -mavx2 flag).
    // The runtime dispatcher should prevent this from being called on unsupported CPUs.
    std::cerr << "Warning: gemm_avx2 called but AVX2/FMA not enabled at compile time. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX2__ && __FMA__


// 3. AVX-512 Optimized GEMM Implementation
// This kernel uses AVX-512 intrinsics. While included as requested, it's important to note that
// the target CPU (AMD Ryzen 7 6800HS) does *not* support AVX-512. The runtime
// dispatcher (`gemm` function) will detect this and fall back to AVX2 or scalar.
#if defined(__AVX512F__) && defined(__FMA__)
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    // Blocking parameters
    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN;
    const int BK = DEFAULT_BK;

    // Micro-kernel dimensions for AVX-512
    const int MR = MR_AVX512; // 4 rows
    const int NR = NR_AVX512; // 16 floats per __m512 vector

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_start = 0; m_start < M; m_start += BM) {
        for (int n_start = 0; n_start < N; n_start += BN) {
            for (int k_start = 0; k_start < K; k_start += BK) {
                int M_block = std::min(BM, M - m_start);
                int N_block = std::min(BN, N - n_start);
                int K_block = std::min(BK, K - k_start);

                for (int i = 0; i < M_block; i += MR) {
                    for (int j = 0; j < N_block; j += NR) {
                        __m512 c_acc[MR]; // MR accumulators, each holding NR floats

                        // Load existing C values into accumulators, handling M and N tails.
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < M_block) { // M-tail handling
                                float* C_ptr = &C[(m_start + i + r) * ldc + (n_start + j)];
                                if (N_block - j < NR) { // N-tail handling for initial load using k-mask
                                    __mmask16 k_mask = (__mmask16)((1U << (N_block - j)) - 1);
                                    // _mm512_maskz_loadu_ps loads and zeros out elements not covered by the mask.
                                    c_acc[r] = _mm512_maskz_loadu_ps(k_mask, C_ptr);
                                } else {
                                    c_acc[r] = _mm512_loadu_ps(C_ptr); // Unaligned load for 16 floats
                                }
                            } else {
                                c_acc[r] = _mm512_setzero_ps(); // If row is out of M_block, ensure accumulator is zero
                            }
                        }

                        // Inner K loop for accumulation
                        for (int k_micro = 0; k_micro < K_block; ++k_micro) {
                            const float* B_ptr = &B[(k_start + k_micro) * ldb + (n_start + j)];
                            __m512 b_vec;
                            
                            // N-tail handling for B using AVX-512 k-mask registers
                            if (N_block - j < NR) {
                                __mmask16 k_mask = (__mmask16)((1U << (N_block - j)) - 1);
                                b_vec = _mm512_maskz_loadu_ps(k_mask, B_ptr);
                            } else {
                                b_vec = _mm512_loadu_ps(B_ptr); // Unaligned load for 16 floats
                            }

                            for (int r = 0; r < MR; ++r) {
                                if (i + r < M_block) { // M-tail handling
                                    // Load A scalar and broadcast to all 16 elements of a __m512 register
                                    __m512 a_scalar_vec = _mm512_set1_ps(A[(m_start + i + r) * lda + (k_start + k_micro)]);
                                    // Fused Multiply-Add
                                    c_acc[r] = _mm512_fmadd_ps(a_scalar_vec, b_vec, c_acc[r]);
                                }
                            }
                        } // end k_micro loop

                        // Store accumulated results back to C, handling M and N tails.
                        for (int r = 0; r < MR; ++r) {
                            if (i + r < M_block) { // M-tail handling
                                float* C_ptr = &C[(m_start + i + r) * ldc + (n_start + j)];
                                if (N_block - j < NR) { // N-tail handling for final store using k-mask
                                    __mmask16 k_mask = (__mmask16)((1U << (N_block - j)) - 1);
                                    _mm512_mask_storeu_ps(C_ptr, k_mask, c_acc[r]);
                                } else {
                                    _mm512_storeu_ps(C_ptr, c_acc[r]); // Unaligned store
                                }
                            }
                        }
                    } // end j loop
                } // end i loop
            } // end k_start loop
        } // end n_start loop
    } // end m_start loop (OpenMP parallel for)
}
#else // If AVX512F is not available at compile time, define a dummy function
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    // This warning indicates that the AVX-512 kernel was called, but the compiler was not
    // instructed to enable AVX-512 intrinsics (e.g., missing -mavx512f flag).
    // The runtime dispatcher should prevent this from being called on unsupported CPUs.
    std::cerr << "Warning: gemm_avx512 called but AVX512F/FMA not enabled at compile time. Falling back to scalar." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX512F__ && __FMA__


// 4. Top-level GEMM function with runtime dispatch
// This function determines the best available SIMD instruction set at runtime
// and dispatches the call to the appropriate kernel (AVX-512, AVX2, or scalar).
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    // Runtime CPU feature detection using __builtin_cpu_supports (GCC/Clang specific).
    // The checks are ordered from most advanced (AVX-512) to least advanced (scalar).
    
    // Check for AVX-512 support
    // On AMD Ryzen 7 6800HS, this condition will be false.
#if defined(__AVX512F__) && defined(__FMA__)
    if (__builtin_cpu_supports("avx512f")) {
        std::cout << "Detected AVX-512F. Dispatching to AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif

    // Check for AVX2 support
    // This condition will be true on AMD Ryzen 7 6800HS.
#if defined(__AVX2__) && defined(__FMA__)
    if (__builtin_cpu_supports("avx2")) {
        std::cout << "Detected AVX2. Dispatching to AVX2 kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif

    // Fallback to scalar if no advanced SIMD support is detected or compiled without support
    std::cout << "No AVX2/AVX-512 support detected or compiled without. Dispatching to scalar kernel." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}

// --- Main Function for Demo and Testing ---
int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 512;
    int N = 512;
    int K = 512;
    unsigned int seed = 12345;      // Seed for random number generation
    bool dump_matrices = false;     // Flag to dump matrices to files
    int num_threads_set = 0;        // 0 means use OpenMP default (or system default if not set)

    std::vector<std::string> positional_values;

    // Parse command line arguments. Flagged arguments take precedence over positional.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        try {
            if (arg == "-M" && i + 1 < argc) {
                M = std::stoi(argv[++i]);
            } else if (arg == "-N" && i + 1 < argc) {
                N = std::stoi(argv[++i]);
            } else if (arg == "-K" && i + 1 < argc) {
                K = std::stoi(argv[++i]);
            } else if (arg == "-s" && i + 1 < argc) {
                seed = std::stoul(argv[++i]);
            } else if (arg == "-t" && i + 1 < argc) {
                num_threads_set = std::stoi(argv[++i]);
            } else if (arg == "--dump-matrices") {
                dump_matrices = true;
            } else if (arg == "-h" || arg == "--help") {
                std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M rows] [-N cols] [-K inner] [-s seed] [-t threads] [--dump-matrices]" << std::endl;
                std::cout << "  Positional M, N, K: Optional matrix dimensions. If provided, must be the first 1, 2, or 3 non-flag arguments." << std::endl;
                std::cout << "  -M, -N, -K: Flagged matrix dimensions. These override any positional M, N, K." << std::endl;
                std::cout << "  -s: Random seed (default: 12345)." << std::endl;
                std::cout << "  -t: Number of OpenMP threads (default: system/OMP_NUM_THREADS, or max logical CPUs)." << std::endl;
                std::cout << "  --dump-matrices: Write A.txt, B.txt, C.txt to 'workspace' directory." << std::endl;
                return 0;
            } else {
                // If it's not a known flag, consider it a potential positional argument
                positional_values.push_back(arg);
            }
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: Invalid argument for " << arg << ": " << argv[i] << ". " << e.what() << std::endl;
            return 1;
        } catch (const std::out_of_range& e) {
            std::cerr << "Error: Value out of range for " << arg << ": " << argv[i] << ". " << e.what() << std::endl;
            return 1;
        }
    }

    // Apply positional arguments if they were provided and not overridden by flags.
    // This assumes M, N, K would be the first 1, 2, or 3 non-flag arguments.
    // The previous flag parsing already updated M, N, K if flags were explicitly used.
    if (positional_values.size() >= 1) {
        M = std::stoi(positional_values[0]);
    }
    if (positional_values.size() >= 2) {
        N = std::stoi(positional_values[1]);
    }
    if (positional_values.size() >= 3) {
        K = std::stoi(positional_values[2]);
    }
    // Any further positional_values are ignored for this demo.

    // Set OpenMP thread count if specified
    if (num_threads_set > 0) {
#ifdef _OPENMP
        omp_set_num_threads(num_threads_set);
        std::cout << "OpenMP threads explicitly set to " << num_threads_set << "." << std::endl;
#else
        std::cout << "Warning: OpenMP not enabled, cannot set thread count. Ignoring -t flag." << std::endl;
#endif
    }
    std::cout << "Using " << omp_get_max_threads() << " logical threads for computation." << std::endl;
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    // Use 64-byte aligned memory for matrices, suitable for AVX-512 and AVX2.
    // Optimal alignment can vary, but 64 bytes (cache line size) is a good general choice.
    const size_t alignment = 64; 
    const size_t size_A = (size_t)M * K * sizeof(float);
    const size_t size_B = (size_t)K * N * sizeof(float);
    const size_t size_C = (size_t)M * N * sizeof(float);

    float* A = static_cast<float*>(aligned_malloc(size_A, alignment));
    float* B = static_cast<float*>(aligned_malloc(size_B, alignment)); // Corrected typo: aligned_B -> aligned_malloc
    float* C = static_cast<float*>(aligned_malloc(size_C, alignment));
    float* C_ref = static_cast<float*>(aligned_malloc(size_C, alignment)); // For correctness check

    if (!A || !B || !C || !C_ref) {
        std::cerr << "Error: Failed to allocate aligned memory. Check system memory or matrix dimensions." << std::endl;
        aligned_free(A); aligned_free(B); aligned_free(C); aligned_free(C_ref);
        return 1;
    }

    // Initialize matrices A and B with random values, C with zeros
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Values between 0.0 and 1.0

    auto init_matrix_random = [&](float* matrix, int rows, int cols, int ld) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix[i * ld + j] = dis(gen);
            }
        }
    };

    // Initialize A and B with random values. Using K and N as leading dimensions assumes dense row-major.
    init_matrix_random(A, M, K, K); // A is M x K, lda = K
    init_matrix_random(B, K, N, N); // B is K x N, ldb = N
    std::memset(C, 0, size_C);      // Initialize C to all zeros for C = A*B computation
    std::memset(C_ref, 0, size_C);  // Initialize C_ref to all zeros for scalar reference

    // Create 'workspace' directory and dump A, B if requested
    if (dump_matrices) {
        create_directory("workspace");
        write_matrix_to_file("workspace/A.txt", A, M, K, K);
        write_matrix_to_file("workspace/B.txt", B, K, N, N);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt" << std::endl;
    }

    // --- Run optimized GEMM ---
    std::cout << "\nStarting optimized GEMM computation..." << std::endl;
    auto start_opt = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, K, N, N); // Pass K, N, N as leading dimensions for dense row-major matrices
    auto end_opt = std::chrono::high_resolution_clock::now();
    double time_opt_ms = std::chrono::duration<double, std::milli>(end_opt - start_opt).count();
    
    // --- Run scalar GEMM for correctness verification ---
    std::cout << "Starting scalar GEMM for verification..." << std::endl;
    auto start_scalar = std::chrono::high_resolution_clock::now();
    gemm_scalar(A, B, C_ref, M, N, K, K, N, N);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double time_scalar_ms = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();

    // Verification step
    double max_diff = 0.0;
    const float tolerance = 1e-3f; // Absolute tolerance for floating point comparisons
    bool correct = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = std::abs(C[i * N + j] - C_ref[i * N + j]);
            if (diff > tolerance) {
                if (correct) { // Print details of the first detected error only
                    std::cerr << "Verification FAILED: Mismatch at C[" << i << "][" << j << "].\n";
                    std::cerr << "  Optimized result: " << C[i * N + j] << "\n";
                    std::cerr << "  Scalar result:    " << C_ref[i * N + j] << "\n";
                    std::cerr << "  Absolute difference: " << diff << "\n";
                }
                correct = false;
            }
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    if (correct) {
        std::cout << "Verification PASSED. Maximum absolute difference: " << max_diff << std::endl;
    } else {
        std::cout << "Verification FAILED. Maximum absolute difference: " << max_diff << std::endl;
    }

    // Performance Report
    long long flop_count = 2LL * M * N * K; // Each element in C requires K multiplications and K-1 additions, approx 2K FLOPS
    double gflops_opt = (double)flop_count / (time_opt_ms * 1e6); // GFLOP/s = FLOPs / (ms * 10^6)

    std::cout << "\n--- Performance Report ---" << std::endl;
    std::cout << "Optimized GEMM time: " << time_opt_ms << " ms" << std::endl;
    std::cout << "Optimized GEMM GFLOP/s: " << gflops_opt << std::endl;
    std::cout << "Scalar GEMM time: " << time_scalar_ms << " ms" << std::endl;
    
    // Dump C matrix if requested
    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C, M, N, N);
        std::cout << "Matrix C dumped to workspace/C.txt" << std::endl;
    }

    // Clean up allocated memory
    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    aligned_free(C_ref);

    return correct ? 0 : 1; // Return 0 for success, 1 for failure
}