// Compile instructions for GCC/Clang:
//
// The target CPU (AMD Ryzen 7 6800HS) supports AVX, AVX2, and FMA, but NOT AVX-512.
// Therefore, the primary optimized kernel used on this system will be AVX2.
// The AVX-512 code path will only be compiled and available for runtime dispatch if
// specific AVX-512 flags are provided, targeting a different CPU architecture.
//
// 1. Compile with AVX-512 support (e.g., for Intel Skylake-X or Ice Lake):
//    This command will build AVX-512 enabled code. If run on a non-AVX-512 CPU,
//    the runtime dispatch will correctly fall back to AVX2 or scalar.
//    g++ -O3 -std=c++17 -march=skylake-avx512 -mfma -fopenmp gemm.cpp -o gemm_avx512
//    (Note: '-march=skylake-avx512' implicitly enables AVX512F, AVX512DQ, AVX512BW, AVX512VL)
//
// 2. Compile with AVX2 support (appropriate for AMD Ryzen 7 6800HS and most modern Intel CPUs):
//    This command enables AVX2 and FMA intrinsics. The runtime dispatch will select the AVX2 kernel.
//    g++ -O3 -std=c++17 -march=native -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//    (Note: '-march=native' enables all instruction sets supported by the compiling CPU,
//     which for Ryzen 7 6800HS includes AVX2/FMA. If AVX-512 code is desired for *runtime dispatch*
//     on a separate system, you need to compile with AVX-512 flags as in option 1.)
//
// 3. Portable compilation (no specific SIMD instruction set assumed, relies on runtime dispatch fallbacks):
//    This variant will compile without specific SIMD architecture flags. The runtime dispatch
//    will then check available CPU features and use the most advanced kernel it finds among those
//    that were actually compiled (e.g., if compiled on a CPU with AVX2, it might still use AVX2 if
//    the `-mavx2` flag was *not* explicitly given, but the compiler still recognized the intrinsic calls).
//    However, for strict fallback, it's best to explicitly enable support if you want it.
//    Without `-mavx2` or `-mavx512f`, the SIMD kernels might be optimized out or cause compilation errors
//    for intrinsics. The safer portable option is often `g++ -O3 -std=c++17 -march=x86-64-v2 -mfma -fopenmp gemm.cpp -o gemm_portable`
//    to guarantee AVX2 is available. Or, for a truly minimal version, remove SIMD flags:
//    g++ -O3 -std=c++17 -fopenmp gemm.cpp -o gemm_portable (will likely only use scalar or auto-vectorized code)
//
// Example usage:
// ./gemm_avx2 -M 2048 -N 2048 -K 2048 -t 16
// ./gemm_avx2 512 512 512 --dump-matrices
//

#include <immintrin.h> // For SIMD intrinsics (AVX2, AVX512)
#include <iostream>    // For console I/O (std::cout, std::cerr)
#include <vector>      // For std::vector
#include <cstring>     // For memset (not explicitly used but good for memory ops)
#include <chrono>      // For timing (std::chrono)
#include <random>      // For random matrix initialization (std::mt19937, std::uniform_real_distribution)
#include <cassert>     // For assert (runtime checks)
#include <numeric>     // For std::iota (not used in this version, but generally useful)
#include <algorithm>   // For std::min, std::max
#include <memory>      // For std::unique_ptr (not explicitly used here)
#include <fstream>     // For file I/O (std::ofstream)
#include <filesystem>  // For creating directories (C++17 std::filesystem)
#include <limits>      // For std::numeric_limits
#include <string>      // For std::string and std::stoi/stoul
#include <cmath>       // For std::abs (float/double overloads)

#ifdef _OPENMP
#include <omp.h>       // For OpenMP multi-threading
#else
// Define dummy OpenMP functions if OpenMP is not available at compile time
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {} // Dummy for setting thread count
#endif

// --- Autotuning Parameters ---
// Micro-kernel dimensions (rows of C, columns of C)
// Chosen to fit registers well and maximize data reuse in the inner loops.
// The inner-most loop (micro-kernel) computes a C[MR_X_NR] block.

// AVX2 specific parameters:
// VEC_F_AVX2: Number of floats per __m256 vector (8 floats).
// MR_AVX2: Number of rows of C computed concurrently in the micro-kernel.
//          With 8 rows, we use 8 __m256 accumulators (256 bytes total registers).
//          This value maximizes use of available YMM registers (16 total) while leaving
//          some for A broadcast and B vector.
// NR_AVX2: Number of columns of C computed concurrently (1 vector width).
//          This means we load 1 __m256 vector from B.
constexpr int VEC_F_AVX2 = 8;
constexpr int MR_AVX2 = 8; // Increased from 6 to 8 for better register blocking
constexpr int NR_AVX2 = VEC_F_AVX2; // NR is typically the vector width

// AVX512 specific parameters:
// VEC_F_AVX512: Number of floats per __m512 vector (16 floats).
// MR_AVX512: Number of rows of C computed concurrently.
//            With 4 rows, we use 4 __m512 accumulators (256 bytes total registers).
// NR_AVX512: Number of columns of C computed concurrently (1 vector width).
//            This means we load 1 __m512 vector from B.
constexpr int VEC_F_AVX512 = 16;
constexpr int MR_AVX512 = 4;
constexpr int NR_AVX512 = VEC_F_AVX512; // NR is typically the vector width

// Blocking parameters (M-block, N-block, K-block)
// These define the tile sizes for processing A, B, C to optimize cache reuse.
// The goal is for `BM*BK` (A block) and `BK*BN` (B block) and `BM*BN` (C block)
// to fit within a specific cache level (L1/L2) for active threads.
//
// Target CPU: AMD Ryzen 7 6800HS
// L1 Data cache: 64KB per core (8 cores)
// L2 cache: 512KB per core
// L3 cache: 16MB shared
//
// With BM=96, BN=128, BK=256:
// A_block (BM x BK): 96 * 256 * 4 bytes = 98304 bytes (~96KB)
// B_block (BK x BN): 256 * 128 * 4 bytes = 131072 bytes (~128KB)
// C_block (BM x BN): 96 * 128 * 4 bytes = 49152 bytes (~48KB)
// Total working set per thread for one K-block iteration: ~272KB.
// This should comfortably fit within the 512KB L2 cache per core.
// The chosen values are a good starting point, aiming for L2 cache residency.
constexpr int BM_DEFAULT = 96;
constexpr int BN_DEFAULT = 128;
constexpr int BK_DEFAULT = 256;

// K-loop unroll factor for the micro-kernel.
// More unrolling reduces loop overhead and exposes more instruction-level parallelism,
// but increases instruction cache pressure and register pressure if not managed carefully.
// Value of 8 is a common balance point for AVX2.
constexpr int UNROLL_K_DEFAULT = 8;

// Row-major storage convention:
// Matrix A (M x K) elements are A[i][k] = A + i * lda + k
// Matrix B (K x N) elements are B[k][j] = B + k * ldb + j
// Matrix C (M x N) elements are C + i * ldc + j


// AlignedAllocator: Custom allocator for std::vector to ensure desired alignment.
// This is critical for SIMD performance, especially for aligned memory access intrinsics.
// A 64-byte alignment supports both AVX2 (32-byte) and AVX512 (64-byte) optimally.
// This allocator correctly implements the C++11 (and later) allocator requirements.
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // The rebind struct is crucial for `std::vector` to be able to allocate other types,
    // like its internal node structures, while maintaining the alignment requirement.
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        void* ptr = nullptr;
#if __cplusplus >= 201703L // C++17 introduces std::aligned_alloc
        ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        if (ptr == nullptr) throw std::bad_alloc();
#else // For older C++ standards or systems without std::aligned_alloc, use posix_memalign
        // posix_memalign requires alignment to be a power of 2
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc();
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t n) { // 'n' is required by the allocator concept, but not used by free/std::free
#if __cplusplus >= 201703L
        std::free(p); // Use std::free with std::aligned_alloc
#else
        free(p);      // Use free with posix_memalign
#endif
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U, Alignment>&) const { return true; }
    template <typename U>
    bool operator!=(const AlignedAllocator<U, Alignment>&) const { return false; }
};

// write_matrix_to_file: Helper function to dump matrix content to a text file.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    ofs << std::fixed << std::scientific; // Use scientific notation for floats
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// create_directory_if_not_exists: Helper to ensure the output directory exists for matrix dumps.
void create_directory_if_not_exists(const std::string& path) {
    std::filesystem::path dir_path(path);
    if (!std::filesystem::exists(dir_path)) {
        try {
            std::filesystem::create_directories(dir_path);
            std::cout << "Created directory: " << path << "\n";
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory " << path << ": " << e.what() << "\n";
        }
    }
}


// --- gemm_scalar: Reference (scalar) GEMM implementation ---
// This provides a baseline for correctness verification.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}

// --- gemm_avx2: AVX2 Optimized GEMM Kernel ---
// This kernel uses AVX2 and FMA intrinsics for float operations.
#if defined(__AVX2__) && defined(__FMA__)

// AVX2 micro-kernel: Computes a (MR_AVX2 x NR_AVX2) block of C.
// This micro-kernel handles the innermost loops, accumulating into registers.
// It also handles N-dimension tails using masked loads/stores.
// Parameters:
//   m_start, n_start: Top-left corner of the C block being computed within the current block.
//   k_start, K_end: Range of K-dimension for the current computation (start and end of current K-block).
//   A, B, C, lda, ldb, ldc: Matrix data and leading dimensions.
//   mr_actual, nr_actual: Actual dimensions of the C block, handling M/N tails.
//   current_k_block_size: Total K-block size, used for loop iteration count.
void gemm_avx2_micro_kernel(int m_start, int n_start, int k_start, int K_end,
                            const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                            int mr_actual, int nr_actual, int current_k_block_size) {
    
    // C accumulators: MR_AVX2 rows, each storing one __m256 vector (NR_AVX2 floats).
    __m256 c_acc[MR_AVX2];

    // Create a mask for N-tail handling. This mask is used for both loads and stores.
    // Elements are -1 (all bits set) for active lanes, 0 for inactive lanes.
    alignas(32) int mask_array_avx2[VEC_F_AVX2];
    for (int i = 0; i < VEC_F_AVX2; ++i) {
        mask_array_avx2[i] = (i < nr_actual) ? -1 : 0;
    }
    __m256i load_store_mask_avx2 = _mm256_load_si256((const __m256i*)mask_array_avx2);

    // Initialize C accumulators:
    for (int r = 0; r < MR_AVX2; ++r) {
        if (r < mr_actual) { // Only operate for valid rows in current C micro-block
            if (k_start == 0) {
                // For the first K-block, initialize accumulators to zero.
                c_acc[r] = _mm256_setzero_ps();
            } else {
                // For subsequent K-blocks, load existing partial sums from C.
                // _mm256_maskload_ps sets elements to zero if the corresponding mask bit is not set,
                // which correctly handles N-dimension tails by not loading garbage.
                c_acc[r] = _mm256_maskload_ps(&C[(m_start + r) * ldc + n_start], load_store_mask_avx2);
            }
        } else {
            // Rows beyond mr_actual should also be zeroed to prevent undefined behavior
            // if an intrinsic operation accidentally targets them (though unlikely with proper masking).
            c_acc[r] = _mm256_setzero_ps();
        }
    }

    // K-loop: Iterates through the K-dimension of the current block.
    // It's unrolled by UNROLL_K_DEFAULT to reduce loop overhead and improve ILP.
    for (int k_offset = 0; k_offset < current_k_block_size; k_offset += UNROLL_K_DEFAULT) {
        // Inner unrolled K-loop: processes UNROLL_K_DEFAULT steps.
        for (int uk = 0; uk < UNROLL_K_DEFAULT; ++uk) {
            int k_idx = k_start + k_offset + uk;
            if (k_idx >= K_end) break; // Check K-block boundary for tails

            // Load B vector: B[k_idx][n_start...n_start+NR_AVX2-1]
            // _mm256_maskload_ps performs a masked load, safe for unaligned addresses and N-tails.
            __m256 b_vec = _mm256_maskload_ps(&B[k_idx * ldb + n_start], load_store_mask_avx2);

            // Perform MR_AVX2 Fused Multiply-Add (FMA) operations.
            // For each of the MR_AVX2 rows of C:
            for (int r = 0; r < MR_AVX2; ++r) {
                // Load A scalar: A[m_start+r][k_idx]
                // This loads a single float from matrix A.
                float a_val = A[(m_start + r) * lda + k_idx];
                // Broadcast the A scalar value to all 8 lanes of an AVX2 vector.
                __m256 a_broadcast = _mm256_set1_ps(a_val);

                // Fused Multiply-Add: c_acc[r] = a_broadcast * b_vec + c_acc[r]
                // This computes (A_scalar * B_vector) and adds it to the C accumulator.
                c_acc[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
            }
        }
    }

    // Store results back to C, handling M and N tails.
    // Loop only up to mr_actual rows (handles M tails).
    for (int r = 0; r < mr_actual; ++r) {
        // _mm256_maskstore_ps writes elements to memory based on the mask, leaving unmasked elements untouched.
        _mm256_maskstore_ps(&C[(m_start + r) * ldc + n_start], load_store_mask_avx2, c_acc[r]);
    }
}

// gemm_avx2: Top-level AVX2 GEMM function with tiling and OpenMP.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Use default block sizes (can be tuned via constants).
    int BM = BM_DEFAULT;
    int BN = BN_DEFAULT;
    int BK = BK_DEFAULT;
    
    // Outer loops for M and N blocks, parallelized with OpenMP.
    // 'collapse(2)' allows OpenMP to parallelize both loops over distinct C tiles.
    // 'schedule(static)' provides a balanced workload distribution across threads,
    // which is generally good for predictable performance when block sizes are uniform.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            
            // Determine actual block sizes for M and N to handle matrix dimension tails.
            int current_BM = std::min(BM, M - m_block_start);
            int current_BN = std::min(BN, N - n_block_start);

            // K-block loop: Iterates through the K-dimension.
            // This loop ensures that blocks of A (current_BM x current_BK) and B (current_BK x current_BN)
            // are brought into L2/L3 cache and reused.
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                int current_BK = std::min(BK, K - k_block_start);

                // Loops for micro-kernels within the current M, N, K blocks.
                // Iterate M-dimension in steps of MR_AVX2.
                for (int m_local = 0; m_local < current_BM; m_local += MR_AVX2) {
                    int mr_actual = std::min(MR_AVX2, current_BM - m_local);

                    // Iterate N-dimension in steps of NR_AVX2 (vector width).
                    for (int n_local = 0; n_local < current_BN; n_local += NR_AVX2) {
                        int nr_actual = std::min(NR_AVX2, current_BN - n_local);

                        // Call the AVX2 micro-kernel to compute the C sub-block.
                        gemm_avx2_micro_kernel(m_block_start + m_local,
                                               n_block_start + n_local,
                                               k_block_start,
                                               k_block_start + current_BK, // K_end
                                               A, lda, B, ldb, C, ldc,
                                               mr_actual, nr_actual, current_BK); // current_k_block_size
                    }
                }
            }
        }
    }
}
#else // If AVX2 and FMA are not defined at compile time, provide a dummy function.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "Warning: gemm_avx2 called but compiled without AVX2/FMA support. Falling back to scalar.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX2__ && __FMA__


// --- gemm_avx512: AVX-512 Optimized GEMM Kernel ---
// This kernel uses AVX-512 and FMA intrinsics for float operations.
#if defined(__AVX512F__) && defined(__FMA__)

// AVX-512 micro-kernel: Computes a (MR_AVX512 x NR_AVX512) block of C.
// NR_AVX512 is VEC_F_AVX512 (one __m512 vector).
void gemm_avx512_micro_kernel(int m_start, int n_start, int k_start, int K_end,
                              const float* A, int lda, const float* B, int ldb, float* C, int ldc,
                              int mr_actual, int nr_actual, int current_k_block_size) {
    
    // C accumulators: MR_AVX512 rows, each storing one __m512 vector (NR_AVX512 floats).
    __m512 c_acc[MR_AVX512];

    // Create a k-mask for N-tail handling. This mask is used for both loads and stores.
    // The mask has `nr_actual` lowest bits set to 1.
    __mmask16 load_store_mask_avx512 = (1U << nr_actual) - 1;

    // Initialize C accumulators (load from C if k_start > 0, else zero)
    for (int r = 0; r < MR_AVX512; ++r) {
        if (r < mr_actual) { // Only operate for valid rows in current C micro-block
            if (k_start == 0) {
                // First K-block, initialize accumulators to zero.
                c_acc[r] = _mm512_setzero_ps();
            } else {
                // Subsequent K-blocks, load existing partial sums from C.
                // _mm512_mask_loadu_ps takes an explicit 'src' for masked-out elements,
                // ensuring they are zero (or a specified value).
                c_acc[r] = _mm512_mask_loadu_ps(_mm512_setzero_ps(), load_store_mask_avx512, &C[(m_start + r) * ldc + n_start]);
            }
        } else {
            c_acc[r] = _mm512_setzero_ps(); // Zero out unused accumulators
        }
    }

    // K-loop (main loop unrolled).
    for (int k_offset = 0; k_offset < current_k_block_size; k_offset += UNROLL_K_DEFAULT) {
        for (int uk = 0; uk < UNROLL_K_DEFAULT; ++uk) {
            int k_idx = k_start + k_offset + uk;
            if (k_idx >= K_end) break; // K-block tail

            // Load B vector: B[k_idx][n_start...n_start+NR_AVX512-1] (16 floats).
            // Use _mm512_mask_loadu_ps for unaligned masked load, handling N tails.
            __m512 b_vec = _mm512_mask_loadu_ps(_mm512_setzero_ps(), load_store_mask_avx512, &B[k_idx * ldb + n_start]);

            // Perform MR_AVX512 FMA operations.
            for (int r = 0; r < MR_AVX512; ++r) {
                // Load A scalar: A[m_start+r][k_idx].
                float a_val = A[(m_start + r) * lda + k_idx];
                // Broadcast A scalar to all 16 lanes of an AVX-512 vector.
                __m512 a_broadcast = _mm512_set1_ps(a_val);

                // Fused Multiply-Add.
                c_acc[r] = _mm512_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
            }
        }
    }

    // Store results to C, handling M and N tails.
    for (int r = 0; r < mr_actual; ++r) { // Loop up to mr_actual rows (M tail)
        // _mm512_mask_storeu_ps stores val to mem_addr where k-bit is set, leaves others unchanged.
        _mm512_mask_storeu_ps(&C[(m_start + r) * ldc + n_start], load_store_mask_avx512, c_acc[r]);
    }
}

// gemm_avx512: Top-level AVX-512 GEMM function with tiling and OpenMP.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    
    // Use default block sizes.
    int BM = BM_DEFAULT;
    int BN = BN_DEFAULT;
    int BK = BK_DEFAULT;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            
            int current_BM = std::min(BM, M - m_block_start);
            int current_BN = std::min(BN, N - n_block_start);

            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                int current_BK = std::min(BK, K - k_block_start);

                for (int m_local = 0; m_local < current_BM; m_local += MR_AVX512) {
                    int mr_actual = std::min(MR_AVX512, current_BM - m_local);

                    for (int n_local = 0; n_local < current_BN; n_local += NR_AVX512) {
                        int nr_actual = std::min(NR_AVX512, current_BN - n_local);

                        gemm_avx512_micro_kernel(m_block_start + m_local,
                                                 n_block_start + n_local,
                                                 k_block_start,
                                                 k_block_start + current_BK,
                                                 A, lda, B, ldb, C, ldc,
                                                 mr_actual, nr_actual, current_BK);
                    }
                }
            }
        }
    }
}
#else // If AVX512F and FMA are not defined at compile time, provide a dummy function.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    std::cerr << "Warning: gemm_avx512 called but compiled without AVX512F/FMA support. Falling back to scalar.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX512F__ && __FMA__

// --- gemm: Main GEMM Dispatcher Function ---
// This function dynamically selects the best available SIMD kernel at runtime
// based on CPU feature detection.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

    // Runtime CPU feature detection using __builtin_cpu_supports (GCC/Clang specific).
    // Checks for AVX512F first, then AVX2, then falls back to scalar.

    // Note for target platform (Ryzen 7 6800HS): This CPU does NOT support AVX-512.
    // The AVX-512 code path will only be taken if compiled for an AVX-512 capable CPU
    // (e.g., specific Intel CPUs) AND running on one where "avx512f" is supported.
    // Otherwise, it will correctly fall through to AVX2.
#if defined(__AVX512F__) && defined(__FMA__)
    if (__builtin_cpu_supports("avx512f")) {
        std::cout << "Using AVX-512 kernel (runtime dispatch).\n";
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif

#if defined(__AVX2__) && defined(__FMA__)
    // This path is expected to be taken for the target AMD Ryzen 7 6800HS.
    if (__builtin_cpu_supports("avx2")) {
        std::cout << "Using AVX2 kernel (runtime dispatch).\n";
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif

    // Fallback to scalar if no SIMD support detected or compiled out.
    std::cout << "Using scalar kernel (fallback).\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}

// --- Main Function for Demo and Benchmarking ---
int main(int argc, char* argv[]) {
    // Default matrix dimensions for demonstration
    int M = 1024;
    int N = 1024;
    int K = 1024;
    unsigned int seed = 42;             // Random seed for matrix initialization
    int num_threads = 0;                // Default to 0, meaning OpenMP's internal default or OMP_NUM_THREADS env var
    bool dump_matrices = false;         // Flag to dump matrices to files

    std::vector<int> positional_dims; // To store M, N, K if provided without flags

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M") { 
            if (i + 1 < argc) { M = std::stoi(argv[++i]); } 
            else { std::cerr << "Error: -M requires an argument.\n"; return 1; }
        } else if (arg == "-N") { 
            if (i + 1 < argc) { N = std::stoi(argv[++i]); } 
            else { std::cerr << "Error: -N requires an argument.\n"; return 1; }
        } else if (arg == "-K") { 
            if (i + 1 < argc) { K = std::stoi(argv[++i]); } 
            else { std::cerr << "Error: -K requires an argument.\n"; return 1; }
        } else if (arg == "-s") { 
            if (i + 1 < argc) { seed = std::stoul(argv[++i]); }
            else { std::cerr << "Error: -s requires an argument.\n"; return 1; }
        } else if (arg == "-t") { 
            if (i + 1 < argc) { num_threads = std::stoi(argv[++i]); }
            else { std::cerr << "Error: -t requires an argument.\n"; return 1; }
        } else if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M <rows>] [-N <cols>] [-K <depth>] [-s <seed>] [-t <threads>] [--dump-matrices]\n";
            std::cout << "  M N K: Positional arguments for matrix dimensions (optional, take precedence if flags not used).\n";
            std::cout << "  -M: Number of rows in A and C (default: 1024, or from positional arg)\n";
            std::cout << "  -N: Number of columns in B and C (default: 1024, or from positional arg)\n";
            std::cout << "  -K: Number of columns in A and rows in B (default: 1024, or from positional arg)\n";
            std::cout << "  -s: Seed for random matrix initialization (default: 42)\n";
            std::cout << "  -t: Number of OpenMP threads to use (default: OMP_NUM_THREADS or system default)\n";
            std::cout << "  --dump-matrices: Write matrices A, B, C to 'workspace/' directory\n";
            std::cout << "  -h, --help: Display this help message\n";
            return 0;
        } else {
            // Attempt to parse as positional argument
            try {
                int val = std::stoi(arg);
                positional_dims.push_back(val);
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Unknown argument or invalid number format: " << arg << ". Use -h or --help for usage.\n";
                return 1;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Number out of range: " << arg << ". Use -h or --help for usage.\n";
                return 1;
            }
        }
    }

    // Apply positional dimensions if provided AND flags for M, N, K were not used
    bool M_flag_set = false; // Need to track if M/N/K were set by flags explicitly
    // Re-scan arguments to check for explicit -M, -N, -K flags
    for (int i = 1; i < argc; ++i) { 
        std::string arg = argv[i];
        if (arg == "-M" || arg == "-N" || arg == "-K") { M_flag_set = true; break; }
    }

    if (!M_flag_set && positional_dims.size() == 3) {
        M = positional_dims[0];
        N = positional_dims[1];
        K = positional_dims[2];
    } else if (positional_dims.size() > 0 && positional_dims.size() != 3) {
        std::cerr << "Error: If providing positional arguments, you must provide exactly 3 (M N K). Provided " << positional_dims.size() << ". Use -h or --help for usage.\n";
        return 1;
    } else if (M_flag_set && positional_dims.size() > 0) {
        std::cerr << "Warning: Both flags (-M -N -K) and positional arguments provided. Flags take precedence. Positional arguments ignored.\n";
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads.\n";
#else
    if (num_threads > 1) {
        std::cout << "Warning: OpenMP not enabled during compilation, running with 1 thread.\n";
    }
    std::cout << "OpenMP not available. Running with 1 thread.\n";
    num_threads = 1; // Ensure consistency if OpenMP is not active
#endif

    std::cout << "GEMM: M=" << M << ", N=" << N << ", K=" << K << ", Seed=" << seed << "\n";

    // Define matrix dimensions (leading dimensions) for row-major storage.
    // For a matrix (rows x cols), ld is typically cols.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Use AlignedFloatVector for all matrices to ensure 64-byte alignment.
    using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;

    // Allocate matrices
    AlignedFloatVector A_vec(static_cast<std::size_t>(M) * K);
    AlignedFloatVector B_vec(static_cast<std::size_t>(K) * N);
    AlignedFloatVector C_vec(static_cast<std::size_t>(M) * N); // Optimized result
    AlignedFloatVector C_ref_vec(static_cast<std::size_t>(M) * N); // Scalar reference result for verification

    // Get raw pointers for function calls
    const float* A = A_vec.data();
    const float* B = B_vec.data();
    float* C = C_vec.data();
    float* C_ref = C_ref_vec.data();

    // Initialize matrices A and B with random values, C with zeros.
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (std::size_t i = 0; i < static_cast<std::size_t>(M) * K; ++i) A_vec[i] = dis(gen);
    for (std::size_t i = 0; i < static_cast<std::size_t>(K) * N; ++i) B_vec[i] = dis(gen);
    // CRITICAL: C must be initialized to zeros once before the optimized GEMM call.
    // The micro-kernels handle accumulation over K-blocks correctly if C is initially zero.
    std::fill(C_vec.begin(), C_vec.end(), 0.0f); 
    std::fill(C_ref_vec.begin(), C_ref_vec.end(), 0.0f); // Zero out C_ref for scalar reference

    // Create workspace directory and dump matrices if requested
    if (dump_matrices) {
        create_directory_if_not_exists("workspace");
        std::cout << "Dumping A to workspace/A.txt...\n";
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        std::cout << "Dumping B to workspace/B.txt...\n";
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
    }

    // --- Optimized GEMM Call and Timing ---
    std::cout << "Starting optimized GEMM computation...\n";
    auto start_opt = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_opt = end_opt - start_opt;

    // --- Scalar GEMM Call for Verification ---
    std::cout << "Starting scalar reference GEMM for verification...\n";
    gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
    
    // --- Verification of Results ---
    double tolerance = 1e-4; // Tolerance for float comparison
    bool correct = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
            // Use a relative tolerance for non-zero values, absolute for near-zero
            // A common way to check floats is abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
            if (diff > tolerance * std::max(std::abs(C_ref[i * ldc + j]), std::abs(C[i*ldc + j])) && diff > tolerance) {
                std::cerr << "Mismatch at C[" << i << "][" << j << "]: Optimized="
                          << C[i * ldc + j] << ", Reference=" << C_ref[i * ldc + j]
                          << ", Diff=" << diff << std::endl;
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    if (correct) {
        std::cout << "Verification PASSED.\n";
    } else {
        std::cerr << "Verification FAILED.\n";
    }

    // --- Performance Report ---
    double num_flops = 2.0 * static_cast<double>(M) * N * K; // Each C[i][j] requires K multiplications and K-1 additions.
    double gflops = num_flops / (elapsed_opt.count() * 1e-3) / 1e9; // Convert ms to seconds, then to GFLOP/s
    std::cout << "Optimized GEMM time: " << elapsed_opt.count() << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOP/s\n";

    // Dump C matrix if requested
    if (dump_matrices) {
        std::cout << "Dumping C to workspace/C.txt...\n";
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
    }

    return correct ? 0 : 1;
}