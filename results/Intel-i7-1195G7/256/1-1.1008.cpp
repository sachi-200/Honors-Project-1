// Compile instructions (GCC/Clang):
//
// These flags enable specific instruction sets and aggressive optimizations.
// -O3: Enables a high level of optimization, including loop unrolling, function inlining, etc.
// -march=<target>: Optimizes for a specific CPU architecture or instruction set baseline.
//   - x86-64-v4: Targets CPUs supporting AVX512F, AVX512DQ, AVX512CD, AVX512BW, AVX512VL.
//                This is appropriate for Intel 11th Gen Core i7-1195G7 (Tiger Lake) and newer.
//   - x86-64-v3: Targets CPUs supporting AVX2, FMA, BMI1, BMI2, LZCNT, POPCNT.
//                This is suitable for older CPUs (e.g., Intel Haswell and newer) supporting AVX2+FMA.
//   - native: Detects the CPU architecture of the machine where the compilation is performed
//             and enables all its supported features for maximum performance on that specific machine.
// -mavx512f: Explicitly enables AVX-512 Foundation instructions. While often implied by -march=x86-64-v4,
//            it's included here for clarity, especially if a less specific -march is used.
// -mavx2: Explicitly enables AVX2 instructions. Similar to -mavx512f, often implied by -march=x86-64-v3.
// -mfma: Explicitly enables Fused Multiply-Add instructions. FMA is crucial for high GEMM performance.
// -fopenmp: Enables OpenMP support for multi-threading.
// -std=c++17: Specifies the C++ standard to C++17.
//
// Example Compile Commands:
//
// 1. For AVX-512 (targeting Intel 11th Gen Core i7-1195G7 or similar Tiger Lake/Rocket Lake CPUs):
//    g++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx512
//    (Alternatively, using specific microarchitecture for Tiger Lake:
//    g++ -O3 -march=tigerlake -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx512)
//
// 2. For AVX2 (fallback for older CPUs supporting AVX2+FMA, e.g., Intel Haswell and newer):
//    g++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp -std=c++17 gemm.cpp -o gemm_avx2
//
// 3. Portable (runtime detection, compiles for the native CPU architecture, best available ISA):
//    g++ -O3 -march=native -fopenmp -std=c++17 gemm.cpp -o gemm_native

#include <iostream>
#include <vector>     // Included as per requirements, not strictly used for matrix storage.
#include <random>
#include <chrono>
#include <cstring>    // For memcpy (not explicitly used in core GEMM, but generally useful).
#include <cassert>
#include <numeric>    // For std::iota (included as per requirements, not explicitly used).
#include <string>
#include <fstream>    // For file output (write_matrix_to_file).
#include <filesystem> // For creating directories (C++17 feature).
#include <algorithm>  // For std::min, std::max.
#include <cmath>      // For std::abs.
#include <cctype>     // For std::isdigit used in argument parsing.

// Intrinsics headers for SIMD operations.
// These are guarded to only include if the respective instruction sets are enabled during compilation.
#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

// OpenMP header for multi-threading.
#if defined(_OPENMP)
#include <omp.h>
#endif

// --- Autotuning Parameters ---
// These parameters control the blocking/tiling strategy, crucial for cache efficiency
// and exploiting instruction-level parallelism. Values are chosen based on common
// practices for x86-64 CPUs with large L2/L3 caches and wide SIMD units.
//
// - BM, BN, BK: Sizes of the outer blocks for M, N, K dimensions. These aim to keep
//   the working set of a block within L2/L3 cache to minimize main memory access.
// - MR, NR: Sizes of the micro-kernel (register block) for M and N dimensions.
//   The C (MR x NR) sub-block is ideally held entirely in CPU registers.
// - UNROLL_K: K-loop unroll factor within the micro-kernel. This helps hide
//   memory latency and improve instruction throughput.
//
// Target Platform (Host CPU): Intel 11th Gen Core i7-1195G7 (Tiger Lake)
// - L1 Data Cache (L1D): 48KB (per core)
// - L2 Cache: 1.25MB (per core)
// - L3 Cache: 12MB (shared)
// - Our implementation uses row-major storage for A, B, C.
//   - C[m][n] is the output matrix.
//   - A[m][k] is the left-hand side matrix.
//   - B[k][n] is the right-hand side matrix.

// Parameters for AVX-512 (VEC_SIZE = 16 floats, 512 bits / 64 bytes per vector)
constexpr int BM_AVX512 = 96;  // M-block size: Should be a multiple of MR_AVX512 for full blocks.
constexpr int BN_AVX512 = 128; // N-block size: Should be a multiple of NR_AVX512 (VEC_SIZE).
constexpr int BK_AVX512 = 128; // K-block size: Impacts L2/L3 cache reuse of A and B.
constexpr int MR_AVX512 = 8;   // M-dimension register block size: Number of C rows simultaneously accumulated in registers.
constexpr int NR_AVX512 = 16;  // N-dimension register block size: Matches AVX-512 vector width (16 floats).
constexpr int UNROLL_K_AVX512 = 4; // K-loop unroll factor within the micro-kernel.

// Parameters for AVX2 (VEC_SIZE = 8 floats, 256 bits / 32 bytes per vector)
constexpr int BM_AVX2 = 96;   // M-block size.
constexpr int BN_AVX2 = 128;  // N-block size: Should be a multiple of NR_AVX2 (VEC_SIZE).
constexpr int BK_AVX2 = 128;  // K-block size.
constexpr int MR_AVX2 = 6;    // M-dimension register block size.
constexpr int NR_AVX2 = 8;    // N-dimension register block size: Matches AVX2 vector width (8 floats).
constexpr int UNROLL_K_AVX2 = 4; // K-loop unroll factor.


// --- Helper Functions ---

// Aligned memory allocation/deallocation
// Ensures memory is aligned to `alignment` bytes. This is critical for SIMD performance,
// as aligned loads/stores can be faster. While `_mm_loadu_ps` and `_mm_storeu_ps` handle
// unaligned access, proper alignment can reduce overheads and enable aligned variants.
// We use 64-byte alignment, which is optimal for AVX-512 (64 bytes) and sufficient for AVX2 (32 bytes).
inline void* aligned_alloc(size_t size, size_t alignment) {
#if defined(_MSC_VER)
    // Microsoft Visual C++ specific aligned allocation
    return _aligned_malloc(size, alignment);
#else
    // POSIX compliant aligned allocation (GCC/Clang)
    void* ptr = nullptr;
    // posix_memalign requires alignment to be a power of 2 and a multiple of sizeof(void*).
    // Our chosen alignment (64 bytes) satisfies this for most systems.
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr; // Allocation failed
    }
    return ptr;
#endif
}

inline void aligned_free(void* ptr) {
#if defined(_MSC_VER)
    // Microsoft Visual C++ specific aligned deallocation
    _aligned_free(ptr);
#else
    // POSIX compliant aligned deallocation
    free(ptr);
#endif
}

// Function to write a matrix to a text file
// Assumes row-major storage with a given leading dimension `ld`.
// This function creates the parent directory if it does not exist.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path()); // Create parent directories

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    ofs.precision(6); // Set precision for floating-point output.
    ofs << std::fixed; // Use fixed-point notation for consistent output format.

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[static_cast<size_t>(i) * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}


// --- Scalar Reference GEMM Implementation ---
// This is a basic, unoptimized triple-nested loop implementation (ijk order).
// It serves as a correctness reference against which optimized versions are compared.
// Assumes row-major storage for matrices A, B, and C.
// The computation is C[m][n] = sum(A[m][k] * B[k][n]) for k from 0 to K-1.
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


// --- AVX2 Optimized GEMM Implementation ---
// This implementation uses AVX2 intrinsics and Fused Multiply-Add (FMA) instructions
// for single-precision floating-point arithmetic. It employs cache-aware blocking
// and register blocking (a micro-kernel) to maximize data reuse and instruction-level parallelism.
// OpenMP is used for parallelizing outer loops (M and N blocks).
// Assumes row-major storage for A, B, C.
#if defined(__AVX2__)
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    
    // VEC_SIZE for AVX2 is 8 floats (256-bit vector).
    const int VEC_SIZE = 8;
    assert(NR_AVX2 == VEC_SIZE && "NR_AVX2 must be equal to VEC_SIZE for full vector loads/stores.");

    // Matrix element access for row-major storage:
    // A(m, k) is at address A + m * lda + k
    // B(k, n) is at address B + k * ldb + n
    // C(m, n) is at address C + m * ldc + n

    // Outer loops for M and N blocks. These loops are parallelized using OpenMP.
    // `collapse(2)` parallelizes both loops as a single logical loop, distributing
    // the M x N block space among available threads.
    // `schedule(static)` ensures a simple and generally effective work distribution
    // by assigning contiguous chunks of iterations to each thread.
#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM_AVX2) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN_AVX2) {
            // Calculate actual block sizes for M and N, handling "tails" (remainder elements)
            // if M or N are not perfect multiples of BM_AVX2/BN_AVX2.
            int current_M_block = std::min(BM_AVX2, M - m_block_start);
            int current_N_block = std::min(BN_AVX2, N - n_block_start);

            // K-block loop (L2 cache blocking layer).
            // This loop iterates over blocks of K to promote reuse of A and B data in L1/L2 caches.
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK_AVX2) {
                // Calculate actual K-block size, handling tails.
                int current_K_block = std::min(BK_AVX2, K - k_block_start);

                // Micro-kernel loops (register blocking layer).
                // These loops iterate over MR_AVX2 x NR_AVX2 sub-blocks within the current M x N block.
                // MR_AVX2 dictates how many rows of A/C are processed simultaneously.
                // NR_AVX2 corresponds to the AVX2 vector width (8 floats).
                for (int m_sub_start = 0; m_sub_start < current_M_block; m_sub_start += MR_AVX2) {
                    int current_MR = std::min(MR_AVX2, current_M_block - m_sub_start);

                    for (int n_sub_start = 0; n_sub_start < current_N_block; n_sub_start += NR_AVX2) {
                        int current_NR = std::min(NR_AVX2, current_N_block - n_sub_start);
                        
                        // --- Micro-kernel: Compute a (current_MR x current_NR) sub-block of C ---
                        // Accumulator registers for a C sub-block. This entire block of C values
                        // is held in vector registers to minimize memory access latency.
                        __m256 c_accum[MR_AVX2];

                        // Load initial values of C into accumulators.
                        // Standard GEMM computes C = alpha * A * B + beta * C.
                        // Here, we implement C += A * B (effectively alpha=1, beta=1, and C initialized to 0).
                        for (int r = 0; r < current_MR; ++r) {
                            c_accum[r] = _mm256_loadu_ps(C + static_cast<size_t>(m_block_start + m_sub_start + r) * ldc + (n_block_start + n_sub_start));
                        }
                        // For rows where current_MR < MR_AVX2 (M-dimension tail), initialize remaining accumulators to zero.
                        // This handles cases where M is not a multiple of MR_AVX2, ensuring valid operations.
                        for (int r = current_MR; r < MR_AVX2; ++r) {
                            c_accum[r] = _mm256_setzero_ps();
                        }

                        // Process K dimension within the micro-kernel, utilizing UNROLL_K_AVX2 for unrolling.
                        // This loop performs the vector dot product accumulation for the C sub-block.
                        for (int k_inner = 0; k_inner < current_K_block; k_inner += UNROLL_K_AVX2) {
                            // Loop for K-unrolling. UNROLL_K_AVX2 iterations are executed per `k_inner` step.
                            for (int uk = 0; uk < UNROLL_K_AVX2; ++uk) {
                                int k_val = k_inner + uk;
                                if (k_val >= current_K_block) break; // Handle K-tail within the unroll loop.

                                // Optional Prefetching for B:
                                // _mm_prefetch((const char*)(B + (static_cast<size_t>(k_block_start + k_val + 1) * ldb + (n_block_start + n_sub_start))), _MM_HINT_T0);
                                // Prefetching can help hide memory latencies, but its effectiveness is highly
                                // architecture-dependent and may require careful tuning.

                                // Perform MR_AVX2 row-vector multiplications using FMA.
                                // For each row 'r' in the C sub-block, compute:
                                // c_accum[r] += A[m_row][k_val] * B[k_val][n_vec_block]
                                for (int r = 0; r < current_MR; ++r) {
                                    // Load a single float from A and broadcast it to all elements of an AVX2 vector.
                                    // This loads A[m_block_start + m_sub_start + r][k_block_start + k_val].
                                    __m256 a_broadcast = _mm256_broadcast_ss(A + static_cast<size_t>(m_block_start + m_sub_start + r) * lda + (k_block_start + k_val));
                                    
                                    // Load a vector of floats from B (B[k_block_start + k_val][n_block_start + n_sub_start ... + VEC_SIZE-1]).
                                    // `_mm256_loadu_ps` is used for unaligned loads, which is common for row-major B.
                                    __m256 b_vec = _mm256_loadu_ps(B + static_cast<size_t>(k_block_start + k_val) * ldb + (n_block_start + n_sub_start));
                                    
                                    // Fused Multiply-Add (FMA): c_accum[r] = a_broadcast * b_vec + c_accum[r].
                                    // FMA performs (a*b)+c in a single operation, improving throughput and precision.
                                    c_accum[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_accum[r]);
                                }
                            }
                        }

                        // Store accumulated results back to C, handling M and N tails.
                        // N-tail handling uses an integer mask for partial vector stores.
                        for (int r = 0; r < current_MR; ++r) {
                            if (current_NR < VEC_SIZE) { // Handle N-tail for partial vector store.
                                // Create an integer mask where the first `current_NR` elements are set (-1, all bits)
                                // and the rest are zero. This masks out writing beyond `current_NR`.
                                alignas(32) int mask_array[VEC_SIZE]; // Aligned for _mm256_load_si256.
                                for (int i = 0; i < VEC_SIZE; ++i) {
                                    mask_array[i] = (i < current_NR) ? -1 : 0; // -1 means all bits set (true for mask).
                                }
                                __m256i write_mask = _mm256_load_si256((__m256i*)mask_array);
                                _mm256_maskstore_ps(C + static_cast<size_t>(m_block_start + m_sub_start + r) * ldc + (n_block_start + n_sub_start), write_mask, c_accum[r]);
                            } else { // Full vector store.
                                _mm256_storeu_ps(C + static_cast<size_t>(m_block_start + m_sub_start + r) * ldc + (n_block_start + n_sub_start), c_accum[r]);
                            }
                        }
                    }
                }
            }
        }
    }
}
#else // __AVX2__ not defined
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    // Fallback: If compiled without AVX2 intrinsics support, this function still needs
    // to exist to satisfy the exact signature requirement. However, the runtime dispatch
    // in the top-level `gemm` function should prevent it from being called if AVX2 is not
    // available on the CPU or not enabled during compilation.
    // If it is explicitly called or if dispatch logic fails, it will perform a scalar computation.
    std::cerr << "Warning: AVX2 kernel called but AVX2 intrinsics are not enabled during compilation. Falling back to scalar.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX2__


// --- AVX-512 Optimized GEMM Implementation ---
// This implementation uses AVX-512 intrinsics and FMA for single-precision floating-point arithmetic.
// It employs cache-aware blocking and register blocking (a micro-kernel) to maximize data reuse
// and instruction-level parallelism. OpenMP is used for parallelizing outer loops.
// Assumes row-major storage for A, B, C.
#if defined(__AVX512F__)
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    
    // VEC_SIZE for AVX-512 is 16 floats (512-bit vector).
    const int VEC_SIZE = 16;
    assert(NR_AVX512 == VEC_SIZE && "NR_AVX512 must be equal to VEC_SIZE for full vector loads/stores.");

    // Row-major storage: A[m][k], B[k][n], C[m][n]

#if defined(_OPENMP)
    #pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM_AVX512) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN_AVX512) {
            int current_M_block = std::min(BM_AVX512, M - m_block_start);
            int current_N_block = std::min(BN_AVX512, N - n_block_start);

            for (int k_block_start = 0; k_block_start < K; k_block_start += BK_AVX512) {
                int current_K_block = std::min(BK_AVX512, K - k_block_start);

                for (int m_sub_start = 0; m_sub_start < current_M_block; m_sub_start += MR_AVX512) {
                    int current_MR = std::min(MR_AVX512, current_M_block - m_sub_start);

                    for (int n_sub_start = 0; n_sub_start < current_N_block; n_sub_start += NR_AVX512) {
                        int current_NR = std::min(NR_AVX512, current_N_block - n_sub_start);
                        
                        // --- Micro-kernel (MR_AVX512 x NR_AVX512 block computation) ---
                        // Accumulator registers for C sub-block (current_MR rows x VEC_SIZE columns).
                        __m512 c_accum[MR_AVX512];

                        // Load initial values of C into accumulators.
                        for (int r = 0; r < current_MR; ++r) {
                            c_accum[r] = _mm512_loadu_ps(C + static_cast<size_t>(m_block_start + m_sub_start + r) * ldc + (n_block_start + n_sub_start));
                        }
                        // For rows where current_MR < MR_AVX512 (M-dimension tail), initialize remaining accumulators to zero.
                        for (int r = current_MR; r < MR_AVX512; ++r) {
                            c_accum[r] = _mm512_setzero_ps();
                        }

                        // Process K dimension within the micro-kernel (UNROLL_K_AVX512 unroll factor).
                        for (int k_inner = 0; k_inner < current_K_block; k_inner += UNROLL_K_AVX512) {
                            for (int uk = 0; uk < UNROLL_K_AVX512; ++uk) {
                                int k_val = k_inner + uk;
                                if (k_val >= current_K_block) break; // Handle K-tail within unroll loop.

                                // Optional Prefetching for B:
                                // _mm_prefetch((const char*)(B + (static_cast<size_t>(k_block_start + k_val + 1) * ldb + (n_block_start + n_sub_start))), _MM_HINT_T0);

                                // Load A scalar value for each row in the MR_AVX512 block, then broadcast.
                                for (int r = 0; r < current_MR; ++r) {
                                    // _mm512_set1_ps loads a scalar float from a memory location and broadcasts
                                    // it to all 16 elements of an AVX-512 vector.
                                    __m512 a_broadcast = _mm512_set1_ps(A[static_cast<size_t>(m_block_start + m_sub_start + r) * lda + (k_block_start + k_val)]);
                                    
                                    // Load a vector of floats from B (B[k_val][n_sub_start ... n_sub_start + VEC_SIZE-1]).
                                    // `_mm512_loadu_ps` is used for unaligned loads.
                                    __m512 b_vec = _mm512_loadu_ps(B + static_cast<size_t>(k_block_start + k_val) * ldb + (n_block_start + n_sub_start));
                                    
                                    // Fused Multiply-Add (FMA).
                                    c_accum[r] = _mm512_fmadd_ps(a_broadcast, b_vec, c_accum[r]);
                                }
                            }
                        }

                        // Store accumulated results back to C, handling M and N tails.
                        // N-tail handling uses a __mmask16 for partial vector stores.
                        for (int r = 0; r < current_MR; ++r) {
                            if (current_NR < VEC_SIZE) { // Handle N-tail for partial vector store using a mask.
                                // Create a bitmask where the first `current_NR` bits are set (1s) and the rest are zeros.
                                __mmask16 write_mask = (1 << current_NR) - 1; 
                                _mm512_mask_storeu_ps(C + static_cast<size_t>(m_block_start + m_sub_start + r) * ldc + (n_block_start + n_sub_start), write_mask, c_accum[r]);
                            } else { // Full vector store.
                                _mm512_storeu_ps(C + static_cast<size_t>(m_block_start + m_sub_start + r) * ldc + (n_block_start + n_sub_start), c_accum[r]);
                            }
                        }
                    }
                }
            }
        }
    }
}
#else // __AVX512F__ not defined
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    // Fallback: If compiled without AVX-512 intrinsics support, this function still needs
    // to exist to satisfy the exact signature requirement. However, the runtime dispatch
    // in the top-level `gemm` function should prevent it from being called if AVX-512 is not
    // available on the CPU or not enabled during compilation.
    // If it is explicitly called or if dispatch logic fails, it will perform a scalar computation.
    std::cerr << "Warning: AVX-512 kernel called but AVX-512 intrinsics are not enabled during compilation. Falling back to scalar.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX512F__


// --- Top-level GEMM Function with Runtime Dispatch ---
// This function dynamically selects the best available GEMM kernel
// based on CPU features detected at runtime using `__builtin_cpu_supports`.
// This ensures that the most optimized kernel runs on the host CPU.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    
#if defined(__GNUC__) || defined(__clang__)
    // Runtime dispatch is preferred for flexibility, allowing a single binary
    // to run optimally on different x86-64 CPUs (e.g., AVX-512 on Tiger Lake, AVX2 on Haswell).
    // Prioritize AVX-512 if supported by the CPU.
    // Intel 11th Gen Core i7-1195G7 (Tiger Lake) supports AVX-512.
    if (__builtin_cpu_supports("avx512f")) {
        // std::cout << "Using AVX-512 kernel.\n"; // Uncomment for verbose dispatch info
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    } 
    // Otherwise, check for AVX2.
    else if (__builtin_cpu_supports("avx2")) {
        // std::cout << "Using AVX2 kernel.\n"; // Uncomment for verbose dispatch info
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    } 
    // Fallback to scalar if no SIMD instruction set is available or supported by compilation flags.
    else {
        // std::cout << "Using scalar kernel.\n"; // Uncomment for verbose dispatch info
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
#else
    // For compilers without `__builtin_cpu_supports` (e.g., MSVC), rely on compile-time flags.
    // This approach binds the kernel choice at compile-time based on the build flags.
#if defined(__AVX512F__)
    // std::cout << "Using AVX-512 kernel (compiled with __AVX512F__).\n";
    gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
#elif defined(__AVX2__)
    // std::cout << "Using AVX2 kernel (compiled with __AVX2__).\n";
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
#else
    // std::cout << "Using scalar kernel (no SIMD support detected at compile time).\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
#endif
}


// --- Main function for testing and demonstration ---
int main(int argc, char* argv[]) {
    int M = 512; // Default M dimension
    int N = 512; // Default N dimension
    int K = 512; // Default K dimension
    unsigned int seed = std::chrono::high_resolution_clock::now().time_since_epoch().count(); // Random seed
    // num_threads = 0 means use OpenMP default (typically system's logical core count).
    // This can also be controlled by the OMP_NUM_THREADS environment variable.
    int num_threads = 0; 
    bool dump_matrices = false; // Flag to dump matrices to files

    // Flexible command line argument parsing:
    // First, try to read up to 3 numeric arguments as M, N, K if they appear at the beginning
    // and are not preceded by a flag ('-'). This allows for simpler `gemm 1024 1024 1024` calls.
    int current_arg_idx = 1;
    
    // Parse M, N, K as positional arguments
    if (argc > current_arg_idx && argv[current_arg_idx][0] != '-' && std::isdigit(argv[current_arg_idx][0])) {
        M = std::stoi(argv[current_arg_idx++]);
    }
    if (argc > current_arg_idx && argv[current_arg_idx][0] != '-' && std::isdigit(argv[current_arg_idx][0])) {
        N = std::stoi(argv[current_arg_idx++]);
    }
    if (argc > current_arg_idx && argv[current_arg_idx][0] != '-' && std::isdigit(argv[current_arg_idx][0])) {
        K = std::stoi(argv[current_arg_idx++]);
    }

    // Process remaining arguments as named flags.
    for (int i = current_arg_idx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "-M" && i + 1 < argc) {
            M = std::stoi(argv[++i]);
        } else if (arg == "-N" && i + 1 < argc) {
            N = std::stoi(argv[++i]);
        } else if (arg == "-K" && i + 1 < argc) {
            K = std::stoi(argv[++i]);
        } else if (arg == "-s" && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [M_dim] [N_dim] [K_dim] [-M <rows>] [-N <cols>] [-K <inner_dim>] [-s <seed>] [-t <threads>] [--dump-matrices]\n";
            std::cout << "  [M_dim] [N_dim] [K_dim]: Optional positional arguments for M, N, K (defaults 512, 512, 512). Must appear before any flags.\n";
            std::cout << "  -M, -N, -K: Matrix dimensions (can override positional arguments if given later).\n";
            std::cout << "  -s: Random seed for matrix initialization.\n";
            std::cout << "  -t: Number of OpenMP threads (0 for system default, e.g., 8 for 4-core HT CPU).\n";
            std::cout << "  --dump-matrices: Write matrices A, B, and C to 'workspace/A.txt', 'workspace/B.txt', 'workspace/C.txt'.\n";
            return 0;
        } else {
            std::cerr << "Error: Unknown argument or missing value for " << arg << "\n";
            return 1;
        }
    }

#if defined(_OPENMP)
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << "\n";
#else
    if (num_threads > 0) {
        std::cerr << "Warning: OpenMP not enabled, -t argument ignored.\n";
    }
    std::cout << "OpenMP not enabled.\n";
#endif

    // Define leading dimensions for row-major layout
    // For A (M x K matrix), leading dimension `lda` is K.
    // For B (K x N matrix), leading dimension `ldb` is N.
    // For C (M x N matrix), leading dimension `ldc` is N.
    int lda = K; 
    int ldb = N; 
    int ldc = N; 

    // Allocate matrices with 64-byte alignment.
    // This alignment is optimal for AVX-512 (64 bytes / 512 bits) and also sufficient for AVX2 (32 bytes / 256 bits).
    size_t alignment = 64; 

    float* A = static_cast<float*>(aligned_alloc(static_cast<size_t>(M) * K * sizeof(float), alignment));
    float* B = static_cast<float*>(aligned_alloc(static_cast<size_t>(K) * N * sizeof(float), alignment));
    float* C = static_cast<float*>(aligned_alloc(static_cast<size_t>(M) * N * sizeof(float), alignment));
    float* C_ref = static_cast<float*>(aligned_alloc(static_cast<size_t>(M) * N * sizeof(float), alignment));

    if (!A || !B || !C || !C_ref) {
        std::cerr << "Error: Failed to allocate memory for matrices. Check dimensions or available RAM.\n";
        aligned_free(A); aligned_free(B); aligned_free(C); aligned_free(C_ref);
        return 1;
    }

    // Initialize matrices: A and B with random float values, C and C_ref with zeros.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (long long i = 0; i < static_cast<long long>(M) * K; ++i) A[i] = dis(gen);
    for (long long i = 0; i < static_cast<long long>(K) * N; ++i) B[i] = dis(gen);
    for (long long i = 0; i < static_cast<long long>(M) * N; ++i) C[i] = 0.0f;
    for (long long i = 0; i < static_cast<long long>(M) * N; ++i) C_ref[i] = 0.0f; // C_ref will be computed by scalar GEMM

    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Matrix storage: Row-major\n";
    std::cout << "Random seed: " << seed << "\n";

    if (dump_matrices) {
        std::cout << "Dumping matrices A and B to workspace/A.txt, workspace/B.txt...\n";
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
    }

    // Run scalar reference for correctness checking.
    std::cout << "Running scalar reference GEMM...\n";
    auto start_ref = std::chrono::high_resolution_clock::now();
    gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
    auto end_ref = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_ref = end_ref - start_ref;
    std::cout << "Scalar GEMM time: " << elapsed_ref.count() << " ms\n";

    // Run optimized GEMM (dispatches to AVX-512/AVX2/scalar based on CPU features).
    std::cout << "Running optimized GEMM...\n";
    auto start_opt = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc); 
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_opt = end_opt - start_opt;
    std::cout << "Optimized GEMM time: " << elapsed_opt.count() << " ms\n";

    // Performance metrics: Calculate GFLOP/s (Giga Floating-point Operations per Second).
    // GEMM involves 2 operations (1 multiply, 1 add) for each of M*N*K elements.
    long long num_flops = 2LL * M * N * K; 
    double gflops_ref = static_cast<double>(num_flops) / (elapsed_ref.count() * 1e6);
    double gflops_opt = static_cast<double>(num_flops) / (elapsed_opt.count() * 1e6);
    std::cout << "Scalar GFLOP/s: " << gflops_ref << "\n";
    std::cout << "Optimized GFLOP/s: " << gflops_opt << "\n";

    // Correctness check: compare optimized results with scalar reference.
    float max_diff = 0.0f;
    float max_relative_diff = 0.0f;
    float tolerance = 1e-4f; // Standard tolerance for floating-point comparisons.
    int diff_count = 0;

    for (long long i = 0; i < static_cast<long long>(M) * N; ++i) {
        float diff = std::abs(C[i] - C_ref[i]);
        max_diff = std::max(max_diff, diff);

        // Calculate relative difference only if the reference value is not too close to zero.
        // Adding a small epsilon to the denominator prevents division by zero for very small C_ref values.
        if (std::abs(C_ref[i]) > 1e-6f) { 
            max_relative_diff = std::max(max_relative_diff, diff / std::abs(C_ref[i]));
        }

        // Flag a mismatch if either absolute difference exceeds tolerance OR
        // (if C_ref is not near zero) relative difference exceeds tolerance.
        if (diff > tolerance && (std::abs(C_ref[i]) < 1e-6f || diff / std::abs(C_ref[i]) > tolerance)) {
            diff_count++;
            // Uncomment the block below to print the first few mismatches for debugging purposes.
            /*
            if (diff_count < 10) { 
                long long r_idx = i / N;
                long long c_idx = i % N;
                std::cerr << "Mismatch at C[" << r_idx << "][" << c_idx << "]: C_opt=" << C[i] << ", C_ref=" << C_ref[i] << ", Diff=" << diff << "\n";
            }
            */
        }
    }

    std::cout << "Max absolute difference: " << max_diff << "\n";
    std::cout << "Max relative difference: " << max_relative_diff << "\n";

    if (diff_count == 0 || max_relative_diff < tolerance) {
        std::cout << "Result: PASSED (within tolerance " << tolerance << ")\n";
    } else {
        std::cout << "Result: FAILED (" << diff_count << " mismatches found, max relative diff " << max_relative_diff << " > tolerance " << tolerance << ")\n";
    }

    if (dump_matrices) {
        std::cout << "Dumping matrix C to workspace/C.txt...\n";
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
    }

    // Free all allocated memory to prevent leaks.
    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    aligned_free(C_ref);

    return 0;
}