// Compile instructions for GCC/Clang:
//
// The target CPU (AMD Ryzen 7 6800HS) supports AVX, AVX2, and FMA, but NOT AVX-512.
// Therefore, the primary optimized kernel used on this system will be AVX2.
// The AVX-512 code path will only be compiled and available for runtime dispatch if
// specific AVX-512 flags are provided, targeting a different CPU architecture.
//
// Example compile commands:
//
// 1. Compile with AVX-512 support (e.g., for Intel Skylake-X or Ice Lake CPUs):
//    This command enables AVX-512 intrinsics. If run on a non-AVX-512 CPU,
//    the runtime dispatch will correctly fall back to AVX2 or scalar.
//    g++ -O3 -std=c++17 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//
// 2. Compile with AVX2 support (appropriate for AMD Ryzen 7 6800HS and most modern Intel CPUs):
//    This command enables AVX2 and FMA intrinsics. The runtime dispatch will select the AVX2 kernel.
//    g++ -O3 -std=c++17 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// 3. Portable compilation (relies on '-march=native' or no specific SIMD flags; runtime dispatch handles fallbacks):
//    '-march=native' enables all instruction sets supported by the compiling CPU.
//    If compiling on Ryzen 7 6800HS, this will enable AVX2/FMA, and the AVX2 kernel will be used.
//    g++ -O3 -std=c++17 -march=native -fopenmp gemm.cpp -o gemm_portable
//
// Example usage:
// ./gemm_avx2 -M 2048 -N 2048 -K 2048 -t 16
// ./gemm_avx2 512 512 512 --dump-matrices --no-autotune
//

#include <immintrin.h> // For SIMD intrinsics (AVX2, AVX512)
#include <iostream>    // For console I/O (std::cout, std::cerr)
#include <vector>      // For std::vector
#include <cstring>     // For memset (not explicitly used but good for memory ops)
#include <chrono>      // For timing (std::chrono)
#include <random>      // For random matrix initialization (std::mt19937, std::uniform_real_distribution)
#include <cassert>     // For assert (runtime checks)
#include <numeric>     // For std::iota (not used in this version, but generally useful)
#include <algorithm>   // For std::min, std::max, std::copy, std::fill
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

// --- Forward Declarations ---
// These declarations ensure that the 'gemm' dispatcher can see all
// kernel signatures, regardless of which ISA support is compiled in.
void gemm_scalar(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);
void gemm_avx2(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);
void gemm_avx512(const float* A, const float* B, float* C, int M, int N, int K, int lda, int ldb, int ldc);

// --- Autotuning Parameters ---
// Micro-kernel dimensions (rows of C, columns of C)
// Chosen to fit registers well and maximize data reuse in the inner loops.
// The inner-most loop (micro-kernel) computes a C[MR_X_NR_REG_BLOCK] block.

// AVX2 specific parameters:
// VEC_F_AVX2: Number of floats per __m256 vector (8 floats).
// MR_AVX2: Number of rows of C computed concurrently in the micro-kernel.
//          With MR_AVX2=6, this means 6 rows of C are accumulated in registers.
// NC_VECS_AVX2: Number of __m256 vectors used to cover the N-dimension block within the micro-kernel.
//               With NC_VECS_AVX2=2, this means 2 __m256 vectors are used per row.
//               Total accumulators: MR_AVX2 * NC_VECS_AVX2 = 6 * 2 = 12 __m256 vectors.
//               This configuration efficiently utilizes 12 of the 16 YMM registers on AVX2-enabled CPUs,
//               providing a 6x16 (rows x cols) register block for C.
//               Remaining 4 YMM registers are sufficient for A broadcasts (1) and B loads (2).
// NR_REG_BLOCK_AVX2: Total columns of C computed concurrently by one micro-kernel call.
constexpr int VEC_F_AVX2 = 8;
constexpr int MR_AVX2 = 6;
constexpr int NC_VECS_AVX2 = 2; // Uses 2 YMM registers for N-dimension per M-row (16 floats total)
constexpr int NR_REG_BLOCK_AVX2 = VEC_F_AVX2 * NC_VECS_AVX2; // Total columns = 16

// AVX512 specific parameters:
// VEC_F_AVX512: Number of floats per __m512 vector (16 floats).
// MR_AVX512: Number of rows of C computed concurrently.
//            With MR_AVX512=8, this means 8 rows of C are accumulated.
// NC_VECS_AVX512: Number of __m512 vectors used to cover the N-dimension block within the micro-kernel.
//                With NC_VECS_AVX512=2, this means 2 __m512 vectors are used per row.
//                Total accumulators: MR_AVX512 * NC_VECS_AVX512 = 8 * 2 = 16 __m512 vectors.
//                This uses 16 ZMM registers (of 32 available), which is a good balance for throughput
//                and avoids potential power/thermal throttling often associated with full AVX-512 register use.
//                This configuration provides an 8x32 (rows x cols) register block for C.
// NR_REG_BLOCK_AVX512: Total columns of C computed concurrently by one micro-kernel call.
constexpr int VEC_F_AVX512 = 16;
constexpr int MR_AVX512 = 8;
constexpr int NC_VECS_AVX512 = 2; // Uses 2 ZMM registers for N-dimension per M-row (32 floats total)
constexpr int NR_REG_BLOCK_AVX512 = VEC_F_AVX512 * NC_VECS_AVX512; // Total columns = 32

// K-loop unroll factor for the micro-kernel.
// This is kept as constexpr to allow the compiler to truly unroll the loop,
// providing optimal instruction-level parallelism and reducing loop overhead.
constexpr int UNROLL_K = 8;

// Prefetch distance parameters
// For B and A within the micro-kernel's K-loop. Aims to bring future data into L1/L2 cache.
constexpr int PREFETCH_DISTANCE_K_MICRO = 4;
// For packing A from original matrix. Aims to bring A data for future rows into L1/L2 cache.
// These are for the row-major read from original A to a temporary buffer.
constexpr int PREFETCH_DISTANCE_PACK_A_ROW = 4;
// For packing B from original matrix. Aims to bring B data for future rows into L2 cache.
constexpr int PREFETCH_DISTANCE_PACK_B_ROW = 4;


// Blocking parameters (M-block, N-block, K-block)
// These define the tile sizes for processing A, B, C to optimize cache reuse.
// The goal is for `BM*BK` (A block) and `BK*BN` (B block) and `BM*BN` (C block)
// to fit within a specific cache level (L1/L2) for active threads.
//
// Target CPU: AMD Ryzen 7 6800HS
// L1 Data cache: 32KB per core (8 cores)
// L2 cache: 512KB per core
// L3 cache: 16MB shared
//
// Default values are a good starting point, aiming for L2 cache residency.
// These are declared static to allow runtime modification by the autotune harness.
static int BM = 96;
static int BN = 128;
static int BK = 256;

// Default values (used if autotuning is skipped or to reset)
constexpr int BM_DEFAULT = 96;
constexpr int BN_DEFAULT = 128;
constexpr int BK_DEFAULT = 256;

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

// Global type alias for aligned float vectors, making it accessible to all functions.
using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;

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
            ofs << matrix[static_cast<std::size_t>(i) * ld + j] << (j == cols - 1 ? "" : " ");
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
                sum += A[static_cast<std::size_t>(i) * lda + k] * B[static_cast<std::size_t>(k) * ldb + j];
            }
            C[static_cast<std::size_t>(i) * ldc + j] = sum;
        }
    }
}

// --- AVX2 Optimized Micro-kernel Implementation ---
#if defined(__AVX2__) && defined(__FMA__)
// This implementation contains the actual AVX2 intrinsics.
// It is only compiled if AVX2 and FMA are enabled.
static void gemm_avx2_micro_kernel_impl(const float* A_block_ptr, int A_block_ld, 
                                        const float* B_block_ptr, int B_block_ld, 
                                        float* C_ptr_at_tile, int C_full_ld,
                                        int m_offset_in_tile, int n_offset_in_tile,
                                        int k_block_len,
                                        int mr_actual, int nr_actual, bool is_first_k_block) {
    
    // C accumulators: MR_AVX2 rows, each storing NC_VECS_AVX2 __m256 vectors.
    // This provides a MR_AVX2 x NR_REG_BLOCK_AVX2 block of C in registers.
    __m256 c_acc[MR_AVX2][NC_VECS_AVX2];

    // Create masks for N-tail handling. Each __m256 vector needs its own mask.
    __m256i load_store_masks_avx2[NC_VECS_AVX2];
    for (int v_idx = 0; v_idx < NC_VECS_AVX2; ++v_idx) {
        alignas(32) int mask_array_avx2[VEC_F_AVX2];
        int current_vec_start_col = v_idx * VEC_F_AVX2;
        for (int i = 0; i < VEC_F_AVX2; ++i) {
            mask_array_avx2[i] = (current_vec_start_col + i < nr_actual) ? -1 : 0;
        }
        load_store_masks_avx2[v_idx] = _mm256_load_si256((const __m256i*)mask_array_avx2);
    }

    // Initialize C accumulators (load from C if not first K-block, else zero)
    for (int r = 0; r < MR_AVX2; ++r) {
        for (int v_idx = 0; v_idx < NC_VECS_AVX2; ++v_idx) {
            c_acc[r][v_idx] = _mm256_setzero_ps(); // Initialize to zero always
        }
        if (r < mr_actual && !is_first_k_block) { // Only load for valid rows and if not the first K-block
            for (int v_idx = 0; v_idx < NC_VECS_AVX2; ++v_idx) {
                // Calculate pointer to the start of the current C vector in the current C row for this micro-kernel.
                float* current_C_element_ptr = C_ptr_at_tile + static_cast<std::size_t>((m_offset_in_tile + r)) * C_full_ld + n_offset_in_tile + static_cast<std::size_t>(v_idx) * VEC_F_AVX2;
                // For subsequent K-blocks, load existing partial sums from C.
                // _mm256_maskload_ps loads elements from memory where the corresponding mask bit is set.
                // Masked-out elements are zeroed in the result.
                c_acc[r][v_idx] = _mm256_maskload_ps(current_C_element_ptr, load_store_masks_avx2[v_idx]);
            }
        }
    }

    // K-loop: Iterates through the K-dimension of the current block.
    // It's unrolled by UNROLL_K to reduce loop overhead and improve ILP.
    for (int k_idx = 0; k_idx < k_block_len; k_idx += UNROLL_K) {
        // Inner unrolled K-loop: processes UNROLL_K steps.
        for (int uk = 0; uk < UNROLL_K; ++uk) {
            int k_loop_idx = k_idx + uk;
            if (k_loop_idx >= k_block_len) break; // Check K-block boundary for tails

            // Prefetch for B_block_ptr: (row-major) Prefetch the B data for a future K-iteration.
            // This aims to bring the data into L1/L2 cache slightly ahead of time.
            if (k_loop_idx + PREFETCH_DISTANCE_K_MICRO < k_block_len) {
                const float* prefetch_b_row_ptr = &B_block_ptr[static_cast<std::size_t>(k_loop_idx + PREFETCH_DISTANCE_K_MICRO) * B_block_ld + n_offset_in_tile];
                _mm_prefetch((const char*)prefetch_b_row_ptr, _MM_HINT_T0); // Prefetch to L1 for immediate use
                // If the N-dimension block spans multiple vectors, prefetch the subsequent vector's start.
                if (NC_VECS_AVX2 > 1) {
                    _mm_prefetch((const char*)(prefetch_b_row_ptr + VEC_F_AVX2), _MM_HINT_T0);
                }
            }

            // Prefetch for A_block_ptr: (column-major) Prefetch the A data for a future K-iteration.
            // A_block_ld is current_BM. Prefetch the column slice for the micro-kernel's M-range.
            if (k_loop_idx + PREFETCH_DISTANCE_K_MICRO < k_block_len) {
                const float* prefetch_a_ptr = &A_block_ptr[static_cast<std::size_t>(k_loop_idx + PREFETCH_DISTANCE_K_MICRO) * A_block_ld + m_offset_in_tile];
                _mm_prefetch((const char*)prefetch_a_ptr, _MM_HINT_T0); // Prefetch to L1 for immediate use
                // If MR_AVX2 elements span more than one cache line (e.g. 64 bytes), prefetch the next segment.
                if (MR_AVX2 * sizeof(float) > 64) { 
                    _mm_prefetch((const char*)(prefetch_a_ptr + (64 / sizeof(float))), _MM_HINT_T0);
                }
            }


            // Load B vectors (NC_VECS_AVX2 of them) for the current K-column and N-block.
            // B_block_ptr is row-major, so these are contiguous loads along the N-dimension.
            __m256 b_vecs[NC_VECS_AVX2];
            for (int v_idx = 0; v_idx < NC_VECS_AVX2; ++v_idx) {
                b_vecs[v_idx] = _mm256_maskload_ps(&B_block_ptr[static_cast<std::size_t>(k_loop_idx) * B_block_ld + n_offset_in_tile + static_cast<std::size_t>(v_idx) * VEC_F_AVX2], load_store_masks_avx2[v_idx]);
            }

            // Perform MR_AVX2 Fused Multiply-Add (FMA) operations for each row.
            for (int r = 0; r < MR_AVX2; ++r) {
                if (r < mr_actual) { // Only for valid M rows
                    // Load A scalar: A_block_ptr[k_loop_idx * A_block_ld + (m_offset_in_tile + r)]
                    // A_packed (A_block_ptr) is column-major: A_packed[k][i].
                    // This access for `a_val` is sequential as 'r' changes, which is cache-efficient.
                    float a_val = A_block_ptr[static_cast<std::size_t>(k_loop_idx) * A_block_ld + (m_offset_in_tile + r)];
                    // Broadcast the A scalar value to all 8 lanes of an AVX2 vector.
                    __m256 a_broadcast = _mm256_set1_ps(a_val);

                    // Fused Multiply-Add for each B vector, accumulating into corresponding C accumulators.
                    for (int v_idx = 0; v_idx < NC_VECS_AVX2; ++v_idx) {
                        c_acc[r][v_idx] = _mm256_fmadd_ps(a_broadcast, b_vecs[v_idx], c_acc[r][v_idx]);
                    }
                }
            }
        }
    }

    // Store results back to C, handling M and N tails.
    for (int r = 0; r < mr_actual; ++r) { // Loop only up to mr_actual rows (handles M tails).
        for (int v_idx = 0; v_idx < NC_VECS_AVX2; ++v_idx) {
            float* current_C_element_ptr = C_ptr_at_tile + static_cast<std::size_t>((m_offset_in_tile + r)) * C_full_ld + n_offset_in_tile + static_cast<std::size_t>(v_idx) * VEC_F_AVX2;
            _mm256_maskstore_ps(current_C_element_ptr, load_store_masks_avx2[v_idx], c_acc[r][v_idx]);
        }
    }
}
#endif // __AVX2__ && __FMA__

// --- AVX512 Optimized Micro-kernel Implementation ---
#if defined(__AVX512F__) && defined(__FMA__)
// This implementation contains the actual AVX512 intrinsics.
// It is only compiled if AVX512F and FMA are enabled.
static void gemm_avx512_micro_kernel_impl(const float* A_block_ptr, int A_block_ld, 
                                          const float* B_block_ptr, int B_block_ld, 
                                          float* C_ptr_at_tile, int C_full_ld,
                                          int m_offset_in_tile, int n_offset_in_tile,
                                          int k_block_len,
                                          int mr_actual, int nr_actual, bool is_first_k_block) {
    
    // C accumulators: MR_AVX512 rows, each storing NC_VECS_AVX512 __m512 vectors.
    // This provides a MR_AVX512 x NR_REG_BLOCK_AVX512 block of C in registers.
    __m512 c_acc[MR_AVX512][NC_VECS_AVX512];

    // Create k-masks for N-tail handling. Each __m512 vector needs its own mask.
    __mmask16 load_store_masks_avx512[NC_VECS_AVX512];
    for (int v_idx = 0; v_idx < NC_VECS_AVX512; ++v_idx) {
        // Compute mask for the current 16-float vector part
        unsigned int mask_val = 0;
        int current_vec_start_col = v_idx * VEC_F_AVX512;
        for (int i = 0; i < VEC_F_AVX512; ++i) {
            if (current_vec_start_col + i < nr_actual) {
                mask_val |= (1U << i);
            }
        }
        load_store_masks_avx512[v_idx] = static_cast<__mmask16>(mask_val);
    }

    // Initialize C accumulators (load from C if not first K-block, else zero)
    for (int r = 0; r < MR_AVX512; ++r) {
        for (int v_idx = 0; v_idx < NC_VECS_AVX512; ++v_idx) {
            c_acc[r][v_idx] = _mm512_setzero_ps(); // Initialize to zero always
        }
        if (r < mr_actual && !is_first_k_block) { // Only load for valid rows and if not the first K-block
            for (int v_idx = 0; v_idx < NC_VECS_AVX512; ++v_idx) {
                float* current_C_element_ptr = C_ptr_at_tile + static_cast<std::size_t>((m_offset_in_tile + r)) * C_full_ld + n_offset_in_tile + static_cast<std::size_t>(v_idx) * VEC_F_AVX512;
                // Subsequent K-blocks, load existing partial sums from C.
                // _mm512_mask_loadu_ps takes an explicit 'src' (_mm512_setzero_ps()) for masked-out elements,
                // ensuring they are zero. This is robust for tail handling.
                c_acc[r][v_idx] = _mm512_mask_loadu_ps(_mm512_setzero_ps(), load_store_masks_avx512[v_idx], current_C_element_ptr);
            }
        }
    }

    // K-loop (main loop unrolled).
    for (int k_idx = 0; k_idx < k_block_len; k_idx += UNROLL_K) {
        for (int uk = 0; uk < UNROLL_K; ++uk) {
            int k_loop_idx = k_idx + uk;
            if (k_loop_idx >= k_block_len) break; // K-block tail

            // Prefetch for B_block_ptr: Prefetch the B data for a future K-iteration.
            if (k_loop_idx + PREFETCH_DISTANCE_K_MICRO < k_block_len) {
                const float* prefetch_b_row_ptr = &B_block_ptr[static_cast<std::size_t>(k_loop_idx + PREFETCH_DISTANCE_K_MICRO) * B_block_ld + n_offset_in_tile];
                _mm_prefetch((const char*)prefetch_b_row_ptr, _MM_HINT_T0); // Prefetch to L1
                // If the N-dimension block spans multiple vectors, prefetch the subsequent vector's start.
                if (NC_VECS_AVX512 > 1) {
                    _mm_prefetch((const char*)(prefetch_b_row_ptr + VEC_F_AVX512), _MM_HINT_T0);
                }
            }

            // Prefetch for A_block_ptr:
            if (k_loop_idx + PREFETCH_DISTANCE_K_MICRO < k_block_len) {
                const float* prefetch_a_ptr = &A_block_ptr[static_cast<std::size_t>(k_loop_idx + PREFETCH_DISTANCE_K_MICRO) * A_block_ld + m_offset_in_tile];
                _mm_prefetch((const char*)prefetch_a_ptr, _MM_HINT_T0);
                if (MR_AVX512 * sizeof(float) > 64) { 
                    _mm_prefetch((const char*)(prefetch_a_ptr + (64 / sizeof(float))), _MM_HINT_T0);
                }
            }

            // Load B vectors (NC_VECS_AVX512 of them) for the current K-column and N-block.
            // B_block_ptr is row-major, so these are contiguous loads along the N-dimension.
            __m512 b_vecs[NC_VECS_AVX512];
            for (int v_idx = 0; v_idx < NC_VECS_AVX512; ++v_idx) {
                b_vecs[v_idx] = _mm512_mask_loadu_ps(_mm512_setzero_ps(), load_store_masks_avx512[v_idx], 
                                                    &B_block_ptr[static_cast<std::size_t>(k_loop_idx) * B_block_ld + n_offset_in_tile + static_cast<std::size_t>(v_idx) * VEC_F_AVX512]);
            }

            // Perform MR_AVX512 FMA operations.
            for (int r = 0; r < MR_AVX512; ++r) {
                if (r < mr_actual) { // Only for valid M rows
                    // Load A scalar: A_block_ptr[k_loop_idx * A_block_ld + (m_offset_in_tile + r)].
                    // A_packed (A_block_ptr) is column-major: A_packed[k][i].
                    // This access for `a_val` is sequential as 'r' changes, which is cache-efficient.
                    float a_val = A_block_ptr[static_cast<std::size_t>(k_loop_idx) * A_block_ld + (m_offset_in_tile + r)];
                    // Broadcast A scalar to all 16 lanes of an AVX-512 vector.
                    __m512 a_broadcast = _mm512_set1_ps(a_val);

                    // Fused Multiply-Add for each B vector.
                    for (int v_idx = 0; v_idx < NC_VECS_AVX512; ++v_idx) {
                        c_acc[r][v_idx] = _mm512_fmadd_ps(a_broadcast, b_vecs[v_idx], c_acc[r][v_idx]);
                    }
                }
            }
        }
    }

    // Store results to C, handling M and N tails.
    for (int r = 0; r < mr_actual; ++r) { // Loop up to mr_actual rows (M tail)
        for (int v_idx = 0; v_idx < NC_VECS_AVX512; ++v_idx) {
            float* current_C_element_ptr = C_ptr_at_tile + static_cast<std::size_t>((m_offset_in_tile + r)) * C_full_ld + n_offset_in_tile + static_cast<std::size_t>(v_idx) * VEC_F_AVX512;
            _mm512_mask_storeu_ps(current_C_element_ptr, load_store_masks_avx512[v_idx], c_acc[r][v_idx]);
        }
    }
}
#endif // __AVX512F__ && __FMA__


// --- gemm_avx2: AVX2 Optimized GEMM Kernel (Public Interface) ---
// This function acts as a wrapper. It either calls the intrinsic-optimized
// implementation or falls back to scalar if AVX2 is not enabled at compile time.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
#if defined(__AVX2__) && defined(__FMA__)
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

            // Per-thread packed buffers for A and B blocks to improve cache locality and reduce striding.
            // These buffers are allocated inside the parallel region but outside the K-loop,
            // making them private to each thread and reused across K-blocks for a given M/N tile.
            // Using AlignedAllocator for optimal SIMD performance.
            AlignedFloatVector A_packed_vec(static_cast<std::size_t>(BM) * BK);
            AlignedFloatVector B_packed_vec(static_cast<std::size_t>(BK) * BN);
            // Temporary buffer for a row of A during packing. Max size BK floats.
            AlignedFloatVector A_row_temp_vec(static_cast<std::size_t>(BK)); 

            float* A_packed = A_packed_vec.data(); // Get raw pointer to packed A
            float* B_packed = B_packed_vec.data(); // Get raw pointer to packed B
            float* A_row_temp = A_row_temp_vec.data(); // Get raw pointer to temporary A row buffer

            // K-block loop: Iterates through the K-dimension.
            // This loop processes `K` in chunks, ensuring blocks of A and B fit in L2/L3 cache.
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                int current_BK = std::min(BK, K - k_block_start);

                // Prefetch for the *next* K-block of A and B (from original matrices)
                // This helps hide memory latency for fetching data for the next iteration of the k_block_start loop.
                // Using _MM_HINT_T1 to suggest prefetching to L2 cache for larger blocks.
                if (k_block_start + BK < K) {
                    const float* next_A_k_block_start_ptr = &A[static_cast<std::size_t>(m_block_start) * lda + (k_block_start + BK)];
                    const float* next_B_k_block_start_ptr = &B[static_cast<std::size_t>(k_block_start + BK) * ldb + n_block_start];
                    
                    // Prefetch a few rows/cache lines from the next A block
                    for(int i = 0; i < std::min(current_BM, 4); ++i) { 
                        _mm_prefetch((const char*)&next_A_k_block_start_ptr[static_cast<std::size_t>(i) * lda], _MM_HINT_T1); 
                    }
                    // Prefetch a few rows/cache lines from the next B block
                    for(int i = 0; i < std::min(current_BK, 4); ++i) { 
                        _mm_prefetch((const char*)&next_B_k_block_start_ptr[static_cast<std::size_t>(i) * ldb], _MM_HINT_T1);
                    }
                }

                // Pack current A block (current_BM x current_BK) into column-major memory within A_packed.
                // A_packed[k][i] = A[i][k] conceptually.
                // Loop over rows of A block (M-dimension) first to enable contiguous reads from original A.
                for (int i_idx = 0; i_idx < current_BM; ++i_idx) { 
                    // Prefetch the next row of the original A matrix for efficient packing.
                    // This prefetch targets a contiguous segment in A's row-major layout.
                    if (i_idx + PREFETCH_DISTANCE_PACK_A_ROW < current_BM) {
                        _mm_prefetch((const char*)&A[static_cast<std::size_t>((m_block_start + i_idx + PREFETCH_DISTANCE_PACK_A_ROW)) * lda + k_block_start], _MM_HINT_T1);
                    }
                    // Read a full row of A into a temporary buffer. This is a contiguous read from original A.
                    std::copy(&A[static_cast<std::size_t>((m_block_start + i_idx)) * lda + k_block_start],
                              &A[static_cast<std::size_t>((m_block_start + i_idx)) * lda + k_block_start + current_BK],
                              A_row_temp);
                    
                    // Distribute this row from A_row_temp into A_packed in column-major format.
                    // This involves strided writes into A_packed, but A_packed is a small, cache-local buffer.
                    for (int k_idx = 0; k_idx < current_BK; ++k_idx) { 
                        A_packed[static_cast<std::size_t>(k_idx) * current_BM + i_idx] = A_row_temp[k_idx];
                    }
                }
                int A_packed_ld = current_BM; // The leading dimension for A_packed is now current_BM (number of rows in the packed A block)

                // Pack current B block (current_BK x current_BN) into contiguous memory (row-major).
                // This provides sequential access for B vector loads in the micro-kernel.
                // The source B is accessed contiguously (row by row), which is cache-friendly.
                for (int i = 0; i < current_BK; ++i) {
                    // Prefetch the start of the next row of B from original matrix.
                    if (i + PREFETCH_DISTANCE_PACK_B_ROW < current_BK) {
                        _mm_prefetch((const char*)&B[static_cast<std::size_t>((k_block_start + i + PREFETCH_DISTANCE_PACK_B_ROW)) * ldb + n_block_start], _MM_HINT_T1);
                    }
                    std::copy(&B[static_cast<std::size_t>((k_block_start + i)) * ldb + n_block_start],
                              &B[static_cast<std::size_t>((k_block_start + i)) * ldb + n_block_start + current_BN],
                              &B_packed[static_cast<std::size_t>(i) * current_BN]); // B_packed is row-major, ld = current_BN
                }
                int B_packed_ld = current_BN; // The leading dimension for B_packed is current_BN.

                // Loops for micro-kernels within the current M, N, K blocks.
                // Iterate M-dimension in steps of MR_AVX2 (micro-kernel row count).
                for (int m_local = 0; m_local < current_BM; m_local += MR_AVX2) {
                    int mr_actual = std::min(MR_AVX2, current_BM - m_local);

                    // Iterate N-dimension in steps of NR_REG_BLOCK_AVX2 (total columns handled by micro-kernel).
                    for (int n_local = 0; n_local < current_BN; n_local += NR_REG_BLOCK_AVX2) {
                        int nr_actual = std::min(NR_REG_BLOCK_AVX2, current_BN - n_local);

                        // Call the AVX2 micro-kernel to compute the C sub-block.
                        gemm_avx2_micro_kernel_impl(A_packed, A_packed_ld,
                                                    B_packed, B_packed_ld,
                                                    C + static_cast<std::size_t>(m_block_start) * ldc + n_block_start, ldc, // Pass base pointer of C tile
                                                    m_local, n_local, // Offsets within the C tile
                                                    current_BK, // K-dimension of packed blocks
                                                    mr_actual, nr_actual,
                                                    k_block_start == 0); // Is this the very first K-block iteration?
                    }
                }
            }
        }
    }
#else // If AVX2 and FMA are not defined at compile time, provide a dummy function.
    std::cerr << "Warning: gemm_avx2 called but compiled without AVX2/FMA support. Falling back to scalar.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif // __AVX2__ && __FMA__
}


// --- gemm_avx512: AVX-512 Optimized GEMM Kernel (Public Interface) ---
// This function acts as a wrapper. It either calls the intrinsic-optimized
// implementation or falls back to scalar if AVX512 is not enabled at compile time.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
#if defined(__AVX512F__) && defined(__FMA__)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            
            int current_BM = std::min(BM, M - m_block_start);
            int current_BN = std::min(BN, N - n_block_start);

            AlignedFloatVector A_packed_vec(static_cast<std::size_t>(BM) * BK);
            AlignedFloatVector B_packed_vec(static_cast<std::size_t>(BK) * BN);
            AlignedFloatVector A_row_temp_vec(static_cast<std::size_t>(BK)); 
            float* A_packed = A_packed_vec.data();
            float* B_packed = B_packed_vec.data();
            float* A_row_temp = A_row_temp_vec.data();

            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                int current_BK = std::min(BK, K - k_block_start);

                // Prefetch for the *next* K-block of A and B (from original matrices)
                if (k_block_start + BK < K) {
                    const float* next_A_k_block_start_ptr = &A[static_cast<std::size_t>(m_block_start) * lda + (k_block_start + BK)];
                    const float* next_B_k_block_start_ptr = &B[static_cast<std::size_t>(k_block_start + BK) * ldb + n_block_start];
                    
                    for(int i = 0; i < std::min(current_BM, 4); ++i) { 
                        _mm_prefetch((const char*)&next_A_k_block_start_ptr[static_cast<std::size_t>(i) * lda], _MM_HINT_T1);
                    }
                    for(int i = 0; i < std::min(current_BK, 4); ++i) { 
                        _mm_prefetch((const char*)&next_B_k_block_start_ptr[static_cast<std::size_t>(i) * ldb], _MM_HINT_T1);
                    }
                }

                // Pack current A block (current_BM x current_BK) into column-major memory within A_packed.
                for (int i_idx = 0; i_idx < current_BM; ++i_idx) {
                    if (i_idx + PREFETCH_DISTANCE_PACK_A_ROW < current_BM) {
                        _mm_prefetch((const char*)&A[static_cast<std::size_t>((m_block_start + i_idx + PREFETCH_DISTANCE_PACK_A_ROW)) * lda + k_block_start], _MM_HINT_T1);
                    }
                    std::copy(&A[static_cast<std::size_t>((m_block_start + i_idx)) * lda + k_block_start],
                              &A[static_cast<std::size_t>((m_block_start + i_idx)) * lda + k_block_start + current_BK],
                              A_row_temp);
                    
                    for (int k_idx = 0; k_idx < current_BK; ++k_idx) {
                        A_packed[static_cast<std::size_t>(k_idx) * current_BM + i_idx] = A_row_temp[k_idx];
                    }
                }
                int A_packed_ld = current_BM;

                // Pack current B block (current_BK x current_BN) into contiguous memory (row-major).
                for (int i = 0; i < current_BK; ++i) {
                    if (i + PREFETCH_DISTANCE_PACK_B_ROW < current_BK) {
                        _mm_prefetch((const char*)&B[static_cast<std::size_t>((k_block_start + i + PREFETCH_DISTANCE_PACK_B_ROW)) * ldb + n_block_start], _MM_HINT_T1);
                    }
                    std::copy(&B[static_cast<std::size_t>((k_block_start + i)) * ldb + n_block_start],
                              &B[static_cast<std::size_t>((k_block_start + i)) * ldb + n_block_start + current_BN],
                              &B_packed[static_cast<std::size_t>(i) * current_BN]);
                }
                int B_packed_ld = current_BN;

                for (int m_local = 0; m_local < current_BM; m_local += MR_AVX512) {
                    int mr_actual = std::min(MR_AVX512, current_BM - m_local);

                    for (int n_local = 0; n_local < current_BN; n_local += NR_REG_BLOCK_AVX512) {
                        int nr_actual = std::min(NR_REG_BLOCK_AVX512, current_BN - n_local);

                        gemm_avx512_micro_kernel_impl(A_packed, A_packed_ld,
                                                      B_packed, B_packed_ld,
                                                      C + static_cast<std::size_t>(m_block_start) * ldc + n_block_start, ldc,
                                                      m_local, n_local,
                                                      current_BK,
                                                      mr_actual, nr_actual,
                                                      k_block_start == 0);
                    }
                }
            }
        }
    }
#else // If AVX512F and FMA are not defined at compile time, provide a dummy function.
    std::cerr << "Warning: gemm_avx512 called but compiled without AVX512F/FMA support. Falling back to scalar.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif // __AVX512F__ && __FMA__
}

// --- gemm: Main GEMM Dispatcher Function ---
// This function dynamically selects the best available SIMD kernel at runtime
// based on CPU feature detection. The selection is done once and cached.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

    // Define function pointer type for GEMM kernels
    using gemm_func_ptr = void (*)(const float*, const float*, float*, int, int, int, int, int, int);
    
    // Static variables to store the selected kernel and whether the message has been printed.
    // These are initialized only once across all calls to gemm.
    static gemm_func_ptr selected_kernel = nullptr;
    static bool printed_kernel_type = false;

    // Perform dispatch logic only on the first call
    if (selected_kernel == nullptr) {
        bool dispatched_successfully = false; // Flag to track if a non-scalar kernel was selected

        // Attempt to dispatch AVX-512 kernel first
#if defined(__AVX512F__) && defined(__FMA__)
        if (__builtin_cpu_supports("avx512f")) {
            selected_kernel = &gemm_avx512;
            if (!printed_kernel_type) { std::cout << "Using AVX-512 kernel (runtime dispatch).\n"; printed_kernel_type = true; }
            dispatched_successfully = true;
        }
#endif

        // If AVX-512 was not dispatched, attempt AVX2
#if defined(__AVX2__) && defined(__FMA__)
        if (!dispatched_successfully && __builtin_cpu_supports("avx2")) {
            selected_kernel = &gemm_avx2;
            if (!printed_kernel_type) { std::cout << "Using AVX2 kernel (runtime dispatch).\n"; printed_kernel_type = true; }
            dispatched_successfully = true;
        }
#endif
        
        // Fallback to scalar if no SIMD kernel was dispatched
        if (!dispatched_successfully) {
            selected_kernel = &gemm_scalar;
            if (!printed_kernel_type) { std::cout << "Using scalar kernel (fallback).\n"; printed_kernel_type = true; }
        }
    }
    
    // Call the selected kernel via the function pointer
    selected_kernel(A, B, C, M, N, K, lda, ldb, ldc);
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
    bool run_autotune = true;           // Flag to control autotuning

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
        } else if (arg == "--no-autotune") { // New flag to disable autotuning
            run_autotune = false;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M <rows>] [-N <cols>] [-K <depth>] [-s <seed>] [-t <threads>] [--dump-matrices] [--no-autotune]\n";
            std::cout << "  M N K: Positional arguments for matrix dimensions (optional, take precedence if flags not used).\n";
            std::cout << "  -M: Number of rows in A and C (default: 1024, or from positional arg)\n";
            std::cout << "  -N: Number of columns in B and C (default: 1024, or from positional arg)\n";
            std::cout << "  -K: Number of columns in A and rows in B (default: 1024, or from positional arg)\n";
            std::cout << "  -s: Seed for random matrix initialization (default: 42)\n";
            std::cout << "  -t: Number of OpenMP threads to use (default: OMP_NUM_THREADS or system default)\n";
            std::cout << "  --dump-matrices: Write matrices A, B, C to 'workspace/' directory\n";
            std::cout << "  --no-autotune: Skip runtime autotuning and use default block sizes (BM=" << BM_DEFAULT << ", BN=" << BN_DEFAULT << ", BK=" << BK_DEFAULT << ")\n";
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
    bool M_flag_set = false; 
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

    // --- Autotuning Harness ---
    if (run_autotune) {
        std::cout << "\nStarting autotuning for block sizes...\n";
        // Candidates for BM, BN, BK. These values are chosen to exercise different cache levels.
        std::vector<int> bm_candidates = {48, 64, 96, 128}; 
        std::vector<int> bn_candidates = {64, 96, 128, 192, 256}; 
        std::vector<int> bk_candidates = {128, 256, 384, 512, 768, 1024}; // Expanded BK candidates

        int best_bm = BM_DEFAULT;
        int best_bn = BN_DEFAULT;
        int best_bk = BK_DEFAULT;
        double min_time_ms = std::numeric_limits<double>::max();

        // Use a smaller problem size for autotuning to keep it time-boxed
        // Chosen to be small enough for fast tuning, large enough to be somewhat representative.
        int tune_M = 256;
        int tune_N = 256;
        int tune_K = 256;
        int tune_lda = tune_K;
        int tune_ldb = tune_N;
        int tune_ldc = tune_N;

        // Allocate tuning matrices using the aligned allocator.
        AlignedFloatVector tune_A_vec(static_cast<std::size_t>(tune_M) * tune_K);
        AlignedFloatVector tune_B_vec(static_cast<std::size_t>(tune_K) * tune_N);
        AlignedFloatVector tune_C_vec(static_cast<std::size_t>(tune_M) * tune_N);
        
        // Initialize for tuning (once)
        std::mt19937 tune_gen(seed);
        std::uniform_real_distribution<float> tune_dis(0.0f, 1.0f);
        for (std::size_t i = 0; i < static_cast<std::size_t>(tune_M) * tune_K; ++i) tune_A_vec[i] = tune_dis(tune_gen);
        for (std::size_t i = 0; i < static_cast<std::size_t>(tune_K) * tune_N; ++i) tune_B_vec[i] = tune_dis(tune_gen);

        const float* tune_A = tune_A_vec.data();
        const float* tune_B = tune_B_vec.data();
        float* tune_C = tune_C_vec.data();

        // Temporarily disable OpenMP for autotuning to measure single-threaded kernel performance accurately
        // The idea is to find good block sizes for the kernel itself, then apply OMP on top.
#ifdef _OPENMP
        int original_num_threads = omp_get_max_threads();
        omp_set_num_threads(1); 
#endif

        for (int cur_bm : bm_candidates) {
            for (int cur_bn : bn_candidates) {
                for (int cur_bk : bk_candidates) {
                    // Temporarily set global blocking parameters
                    BM = cur_bm;
                    BN = cur_bn;
                    BK = cur_bk;

                    // Warmup runs (e.g., to load code into cache, clear TLB, etc.)
                    for (int w = 0; w < 2; ++w) {
                        std::fill(tune_C_vec.begin(), tune_C_vec.end(), 0.0f); // Reset C for each run
                        gemm(tune_A, tune_B, tune_C, tune_M, tune_N, tune_K, tune_lda, tune_ldb, tune_ldc);
                    }

                    // Measurement runs
                    double current_total_time_ms = 0.0;
                    int num_measurements = 3;
                    for (int r = 0; r < num_measurements; ++r) {
                        std::fill(tune_C_vec.begin(), tune_C_vec.end(), 0.0f); // Reset C for each run
                        auto start_tune = std::chrono::high_resolution_clock::now();
                        gemm(tune_A, tune_B, tune_C, tune_M, tune_N, tune_K, tune_lda, tune_ldb, tune_ldc);
                        auto end_tune = std::chrono::high_resolution_clock::now();
                        current_total_time_ms += std::chrono::duration<double, std::milli>(end_tune - start_tune).count();
                    }
                    double avg_time_ms = current_total_time_ms / num_measurements;

                    // std::cout << "  Testing BM=" << cur_bm << ", BN=" << cur_bn << ", BK=" << cur_bk 
                    //           << ": " << avg_time_ms << " ms\n"; // Optional: print all results

                    if (avg_time_ms < min_time_ms) {
                        min_time_ms = avg_time_ms;
                        best_bm = cur_bm;
                        best_bn = cur_bn;
                        best_bk = cur_bk;
                    }
                }
            }
        }
        // Set the best configuration globally for the main GEMM computation
        BM = best_bm;
        BN = best_bn;
        BK = best_bk;
        std::cout << "Autotuning complete. Best configuration: BM=" << BM << ", BN=" << BN << ", BK=" << BK 
                  << " (Time: " << min_time_ms << " ms on small problem).\n";
        
#ifdef _OPENMP
        // Restore original OpenMP thread count
        omp_set_num_threads(original_num_threads);
#endif

    } else {
        // If autotuning is skipped, use the default values
        BM = BM_DEFAULT;
        BN = BN_DEFAULT;
        BK = BK_DEFAULT;
        std::cout << "\nAutotuning skipped. Using default block sizes: BM=" << BM << ", BN=" << BN << ", BK=" << BK << ".\n";
    }
    // --- End Autotuning Harness ---


    std::cout << "GEMM problem: M=" << M << ", N=" << N << ", K=" << K << ", Seed=" << seed << "\n";

    // Define matrix dimensions (leading dimensions) for row-major storage.
    // For a matrix (rows x cols), ld is typically cols.
    int lda = K;
    int ldb = N;
    int ldc = N;

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
            float diff = std::abs(C[static_cast<std::size_t>(i) * ldc + j] - C_ref[static_cast<std::size_t>(i) * ldc + j]);
            // Use a relative tolerance for non-zero values, absolute for near-zero
            // A common way to check floats is abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
            if (diff > tolerance * std::max(std::abs(C_ref[static_cast<std::size_t>(i) * ldc + j]), std::abs(C[static_cast<std::size_t>(i)*ldc + j])) && diff > tolerance) {
                std::cerr << "Mismatch at C[" << i << "][" << j << "]: Optimized="
                          << C[static_cast<std::size_t>(i) * ldc + j] << ", Reference=" << C_ref[static_cast<std::size_t>(i) * ldc + j]
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