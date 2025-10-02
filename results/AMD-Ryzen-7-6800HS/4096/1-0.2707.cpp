// Compile instructions (GCC/Clang):
// For AVX-512 (if available, e.g., Intel Skylake-X/Ice Lake/Rocket Lake or AMD Zen 4):
// g++ -O3 -march=x86-64-v4 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512 -std=c++17
// For AVX2 (Target CPU: AMD Ryzen 7 6800HS, Zen 3+ supports AVX2/FMA):
// g++ -O3 -march=x86-64-v3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2 -std=c++17
// For a portable binary (will use best available at runtime on current host, compile with -march=native):
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native -std=c++17
// Note: x86-64-v3 implies AVX2, FMA. x86-64-v4 implies AVX512F.
// -march=native is generally recommended as it optimizes for the host CPU's specific features.
// It will detect AVX2 on Ryzen 7 6800HS and use the AVX2 kernel.

// Required standard headers
#include <iostream>
#include <vector>
#include <cstring>   // For memcpy, memset
#include <chrono>    // For timing
#include <random>    // For matrix initialization
#include <cassert>   // For assertions
#include <fstream>   // For file I/O
#include <iomanip>   // For std::fixed, std::setprecision
#include <string>
#include <algorithm> // For std::min, std::max
#include <limits>    // For std::numeric_limits

// For OpenMP
#ifdef _OPENMP
#include <omp.h>
#else
// Define dummy OpenMP functions if not available, useful for serial debugging
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
inline void omp_set_num_threads(int) {} // dummy for setting thread count
#endif

// For SIMD intrinsics
#include <immintrin.h>

// For C++17 filesystem (create directory)
#if __has_include(<filesystem>) && __cplusplus >= 201703L
#include <filesystem>
namespace fs = std::filesystem;
#else
// Fallback for older compilers or environments without <filesystem>
#include <sys/stat.h> // For mkdir
#include <errno.h>    // For errno
#ifdef _WIN32
#include <direct.h>   // For _mkdir on Windows
#endif
namespace { // Anonymous namespace for local helper
bool create_directory_fallback(const std::string& path) {
    if (path.empty()) return true;
    #ifdef _WIN32
        return _mkdir(path.c_str()) == 0 || errno == EEXIST;
    #else
        return mkdir(path.c_str(), 0777) == 0 || errno == EEXIST;
    #endif
}
} // anonymous namespace
#endif

// --- Configuration Parameters (CPU-tuned constants) ---
// These parameters are crucial for cache efficiency and register blocking.
// Tuned for AMD Ryzen 7 6800HS (Zen 3+): L1d=32KB, L2=512KB/core, L3=16MB shared.
// L1 cache (32KB): ideal for A_micro_tile_K_block and B_micro_tile_K_block to fit.
// L2 cache (512KB): C_block can often fit here, but not critical with the current loop order.

// Micro-kernel Register Blocking Factors (MR x NR_VECS * VLEN floats)
// These define the number of accumulators (C sub-block) kept entirely in CPU registers.
// For AVX2: VLEN=8 floats per __m256 register.
constexpr int MR_AVX2 = 4;        // Rows of C (and A) processed simultaneously in registers.
constexpr int NR_AVX2_VECS = 2;   // Number of __m256 vectors for N-dimension of C (and B) in registers.
constexpr int NR_AVX2 = NR_AVX2_VECS * 8; // Total N columns processed by AVX2 micro-kernel.
// Accumulators for AVX2: MR_AVX2 * NR_AVX2_VECS = 4 * 2 = 8 __m256 registers = 8 * 32 bytes = 256 bytes.
// This small footprint ensures accumulators remain entirely in registers (L0 cache).

// For AVX-512: VLEN=16 floats per __m512 register.
constexpr int MR_AVX512 = 4;        // Rows of C (and A) processed simultaneously in registers.
constexpr int NR_AVX512_VECS = 1;   // Number of __m512 vectors for N-dimension of C (and B) in registers.
constexpr int NR_AVX512 = NR_AVX512_VECS * 16; // Total N columns processed by AVX-512 micro-kernel.
// Accumulators for AVX-512: MR_AVX512 * NR_AVX512_VECS = 4 * 1 = 4 __m512 registers = 4 * 64 bytes = 256 bytes.
// Also small enough to stay in registers (L0 cache).

// Cache-aware Tile Sizes (BM, BN, BK) for Outer Blocking
// These dimensions determine the size of blocks of A, B, and C that are processed together,
// aiming for reuse within L1/L2/L3 caches.
// For AVX2 (used by Ryzen 7 6800HS):
constexpr int BM_AVX2 = 96;   // Block M: Rows for C and A.
constexpr int BN_AVX2 = 128;  // Block N: Columns for C and B.
constexpr int BK_AVX2 = 32;   // Block K: Inner dimension, influences L1/L2 reuse of A and B.
// A_block_in_K (MR_AVX2 x BK_AVX2): 4 * 32 = 128 floats = 512 bytes. (fits L1d)
// B_block_in_K (BK_AVX2 x NR_AVX2): 32 * 16 = 512 floats = 2048 bytes. (fits L1d)
// These ensure the active portions of A and B for the K-block can be highly reused within L1 cache.
// The (BM x BN) block of C is processed by one thread for the entire K dimension.

// For AVX-512 (if available, generally allows slightly larger blocks due to wider vectors and more registers)
constexpr int BM_AVX512 = 96;
constexpr int BN_AVX512 = 128;
constexpr int BK_AVX512 = 32;
// Similar cache-size rationale applies.

// K-loop unroll factor: The inner K loop in micro-kernel is implicitly unrolled by 1 in this implementation.
// To achieve higher unrolling, the fmadd sequence would need to be manually duplicated multiple times,
// or rely on compiler auto-unrolling.
constexpr int UNROLL_K = 1;


// --- Helper for aligned memory allocation ---
// Custom allocator to ensure memory is aligned to cache line boundaries (e.g., 64 bytes for AVX-512,
// which also satisfies 32 bytes for AVX2). This prevents performance penalties from unaligned memory accesses.
template <typename T, size_t Alignment>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() noexcept = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(size_t num) {
        if (num == 0) return nullptr;
        // Check for overflow before multiplication
        if (num > std::numeric_limits<size_type>::max() / sizeof(T)) throw std::bad_alloc();
        void* ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(num * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc();
#else
        int ret = posix_memalign(&ptr, Alignment, num * sizeof(T));
        if (ret != 0) throw std::bad_alloc(); // posix_memalign returns 0 on success
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        free(p);
#endif
    }

    bool operator==(const AlignedAllocator& other) const { return true; }
    bool operator!=(const AlignedAllocator& other) const { return !(*this == other); }
};
using AlignedVector = std::vector<float, AlignedAllocator<float, 64>>; // 64-byte alignment for general compatibility

// --- Matrix I/O Helper ---
// Writes a matrix to a specified file, handling leading dimension correctly.
// Creates the directory if it does not exist.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::string dir_path = "";
    size_t last_slash = filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        dir_path = filename.substr(0, last_slash);
    }

#if __has_include(<filesystem>) && __cplusplus >= 201703L
    if (!dir_path.empty()) {
        fs::create_directories(dir_path);
    }
#else
    if (!dir_path.empty()) {
        create_directory_fallback(dir_path);
    }
#endif
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }
    ofs << std::fixed << std::setprecision(4);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// --- Scalar Reference GEMM Implementation ---
// This is a straightforward, unoptimized implementation for correctness comparison.
// It assumes row-major storage: A (M x K), B (K x N), C (M x N).
// The core operation is C[i][j] = A[i][k] * B[k][j], assuming C is pre-zeroed.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float c_val = 0.0f;
            for (int k = 0; k < K; ++k) {
                c_val += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = c_val;
        }
    }
}

// --- AVX2 GEMM Implementation (tuned for Ryzen 7 6800HS) ---
// This kernel uses AVX2 and FMA intrinsics with cache-aware tiling and OpenMP parallelization.
// It targets the specified AMD Ryzen 7 6800HS which supports AVX2 and FMA.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    constexpr int VLEN = 8; // Number of floats in __m256 (AVX2 vector)

    // Register blocking parameters for the micro-kernel (MR x NR floats)
    constexpr int MR = MR_AVX2;
    constexpr int NR_VECS = NR_AVX2_VECS;
    constexpr int NR = NR_AVX2;

    // Cache-aware tile sizes for outer blocking (BM x BN x BK)
    constexpr int BM = BM_AVX2;
    constexpr int BN = BN_AVX2;
    constexpr int BK = BK_AVX2;

    // OpenMP parallel region:
    // The `collapse(2)` clause parallelizes the outer two loops (M and N blocks).
    // `schedule(static)` distributes blocks evenly among threads for balanced load.
    // Each thread processes a distinct sub-matrix of C (a BM x BN block), eliminating race conditions.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            // Determine actual dimensions for the current C sub-block
            int m_block_actual = std::min(m_block_start + BM, M) - m_block_start;
            int n_block_actual = std::min(n_block_start + BN, N) - n_block_start;

            // Loop over M dimension using MR-sized micro-tiles (rows of C)
            // m_idx_rel is relative to m_block_start
            for (int m_idx_rel = 0; m_idx_rel < m_block_actual; m_idx_rel += MR) {
                int m_tail_rows = std::min(m_idx_rel + MR, m_block_actual) - m_idx_rel;

                // Loop over N dimension using NR-sized micro-tiles (columns of C)
                // n_idx_rel is relative to n_block_start
                for (int n_idx_rel = 0; n_idx_rel < n_block_actual; n_idx_rel += NR) {
                    int n_tail_cols = std::min(n_idx_rel + NR, n_block_actual) - n_idx_rel;

                    // C accumulators: array of __m256 registers, initialized to zero.
                    // These registers will accumulate the dot products for the entire K dimension
                    // for the current MR x NR micro-tile of C.
                    __m256 c_regs[MR][NR_VECS];
                    for (int i = 0; i < MR; ++i) {
                        for (int j = 0; j < NR_VECS; ++j) {
                            c_regs[i][j] = _mm256_setzero_ps();
                        }
                    }

                    // K-blocking loop: Iterate over K dimension in blocks (BK)
                    // This loop helps manage L1/L2 cache usage for A and B.
                    for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                        int k_block_actual = std::min(k_block_start + BK, K) - k_block_start;

                        // Inner K loop: Process K steps within the current K-block
                        // This is the core micro-kernel accumulation loop.
                        for (int k_step = 0; k_step < k_block_actual; ++k_step) {
                            int k_val_abs = k_block_start + k_step; // Absolute K index

                            // Prefetching: Hint to CPU to load data into cache. Often omitted after profiling.
                            // _mm_prefetch((const char*)(A + (m_block_start + m_idx_rel + MR) * lda + k_val_abs), _MM_HINT_T0);
                            // _mm_prefetch((const char*)(B + (k_val_abs + 1) * ldb + n_block_start + n_idx_rel), _MM_HINT_T0);

                            // Load A values: A[m_abs_row][k_val_abs] and broadcast to all lanes of a vector.
                            __m256 a_broadcast[MR];
                            for (int r = 0; r < m_tail_rows; ++r) {
                                a_broadcast[r] = _mm256_broadcast_ss(A + (m_block_start + m_idx_rel + r) * lda + k_val_abs);
                            }

                            // Loop over N-vectors (NR_VECS for AVX2)
                            for (int nv = 0; nv < NR_VECS; ++nv) {
                                int current_n_col_abs = n_block_start + n_idx_rel + nv * VLEN;
                                
                                // Determine number of valid elements in the current vector for B (tail handling)
                                int elements_in_vec = std::min(VLEN, n_tail_cols - nv * VLEN);
                                if (elements_in_vec <= 0) continue; // No elements for this vector, skip
                                
                                __m256 b_vec;
                                if (elements_in_vec == VLEN) { // Full vector load
                                    b_vec = _mm256_loadu_ps(B + k_val_abs * ldb + current_n_col_abs);
                                } else { // Partial vector load (tail) using memcpy to aligned temporary buffer.
                                         // This is safe for unaligned source `B + ...`
                                    alignas(32) float temp_b_tail[VLEN];
                                    std::memcpy(temp_b_tail, B + k_val_abs * ldb + current_n_col_abs, elements_in_vec * sizeof(float));
                                    // Zero out remaining elements in the temporary buffer to avoid junk values influencing FMA.
                                    std::memset(temp_b_tail + elements_in_vec, 0, (VLEN - elements_in_vec) * sizeof(float));
                                    b_vec = _mm256_load_ps(temp_b_tail); // Load from aligned temporary
                                }

                                // Fused Multiply-Add (FMA) operations: c_regs += a_broadcast * b_vec
                                for (int r = 0; r < m_tail_rows; ++r) {
                                    c_regs[r][nv] = _mm256_fmadd_ps(a_broadcast[r], b_vec, c_regs[r][nv]);
                                }
                            }
                        } // end k_step loop (accumulated for current K-block into c_regs)
                    } // end k_block_start loop (accumulated for full K dimension into c_regs)

                    // After accumulating for the entire K dimension, store results from registers back to C matrix.
                    for (int r = 0; r < m_tail_rows; ++r) {
                        for (int nv = 0; nv < NR_VECS; ++nv) {
                            int current_n_col_abs = n_block_start + n_idx_rel + nv * VLEN;
                            int elements_in_vec = std::min(VLEN, n_tail_cols - nv * VLEN);

                            if (elements_in_vec <= 0) continue; // No elements to process

                            if (elements_in_vec == VLEN) { // Full vector store
                                _mm256_storeu_ps(C + (m_block_start + m_idx_rel + r) * ldc + current_n_col_abs, c_regs[r][nv]);
                            } else { // Partial vector store (tail)
                                // Store to a temporary aligned buffer, then copy relevant parts to C
                                alignas(32) float temp_c_tail[VLEN];
                                _mm256_store_ps(temp_c_tail, c_regs[r][nv]);
                                std::memcpy(C + (m_block_start + m_idx_rel + r) * ldc + current_n_col_abs, temp_c_tail, elements_in_vec * sizeof(float));
                            }
                        }
                    }
                } // end n_idx_rel loop (micro-kernel N blocks)
            } // end m_idx_rel loop (micro-kernel M blocks)
        } // end n_block_start loop
    } // end m_block_start loop (OpenMP)
}

// --- AVX-512 GEMM Implementation ---
// This kernel uses AVX-512 and FMA intrinsics. It will only be compiled if __AVX512F__ is defined
// (e.g., with -mavx512f flag) and will only be dispatched to if the CPU supports it at runtime.
#if defined(__AVX512F__)
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    constexpr int VLEN = 16; // Number of floats in __m512 (AVX-512 vector)

    // Register blocking parameters for the micro-kernel (MR x NR floats)
    constexpr int MR = MR_AVX512;
    constexpr int NR_VECS = NR_AVX512_VECS;
    constexpr int NR = NR_AVX512;

    // Cache-aware tile sizes for outer blocking (BM x BN x BK)
    constexpr int BM = BM_AVX512;
    constexpr int BN = BN_AVX512;
    constexpr int BK = BK_AVX512;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            int m_block_actual = std::min(m_block_start + BM, M) - m_block_start;
            int n_block_actual = std::min(n_block_start + BN, N) - n_block_start;

            for (int m_idx_rel = 0; m_idx_rel < m_block_actual; m_idx_rel += MR) {
                int m_tail_rows = std::min(m_idx_rel + MR, m_block_actual) - m_idx_rel;

                for (int n_idx_rel = 0; n_idx_rel < n_block_actual; n_idx_rel += NR) {
                    int n_tail_cols = std::min(n_idx_rel + NR, n_block_actual) - n_idx_rel;

                    __m512 c_regs[MR][NR_VECS];
                    for (int i = 0; i < MR; ++i) {
                        for (int j = 0; j < NR_VECS; ++j) {
                            c_regs[i][j] = _mm512_setzero_ps();
                        }
                    }

                    for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {
                        int k_block_actual = std::min(k_block_start + BK, K) - k_block_start;

                        for (int k_step = 0; k_step < k_block_actual; ++k_step) {
                            int k_val_abs = k_block_start + k_step;

                            // Prefetching
                            // _mm_prefetch((const char*)(A + (m_block_start + m_idx_rel + MR) * lda + k_val_abs), _MM_HINT_T0);
                            // _mm_prefetch((const char*)(B + (k_val_abs + 1) * ldb + n_block_start + n_idx_rel), _MM_HINT_T0);

                            // Load A values and broadcast
                            __m512 a_broadcast[MR];
                            for (int r = 0; r < m_tail_rows; ++r) {
                                a_broadcast[r] = _mm512_broadcast_ss(A + (m_block_start + m_idx_rel + r) * lda + k_val_abs);
                            }

                            for (int nv = 0; nv < NR_VECS; ++nv) {
                                int current_n_col_abs = n_block_start + n_idx_rel + nv * VLEN;
                                int elements_in_vec = std::min(VLEN, n_tail_cols - nv * VLEN);

                                if (elements_in_vec <= 0) continue;

                                __m512 b_vec;
                                if (elements_in_vec == VLEN) { // Full vector load
                                    b_vec = _mm512_loadu_ps(B + k_val_abs * ldb + current_n_col_abs);
                                } else { // Partial vector load (tail) using mask. _mm512_maskz_loadu_ps zero-pads unused lanes.
                                    __mmask16 mask = (__mmask16)((1 << elements_in_vec) - 1);
                                    b_vec = _mm512_maskz_loadu_ps(mask, B + k_val_abs * ldb + current_n_col_abs);
                                }

                                for (int r = 0; r < m_tail_rows; ++r) {
                                    c_regs[r][nv] = _mm512_fmadd_ps(a_broadcast[r], b_vec, c_regs[r][nv]);
                                }
                            }
                        } // end k_step loop
                    } // end k_block_start loop

                    // Store accumulated results back to C
                    for (int r = 0; r < m_tail_rows; ++r) {
                        for (int nv = 0; nv < NR_VECS; ++nv) {
                            int current_n_col_abs = n_block_start + n_idx_rel + nv * VLEN;
                            int elements_in_vec = std::min(VLEN, n_tail_cols - nv * VLEN);

                            if (elements_in_vec <= 0) continue;

                            if (elements_in_vec == VLEN) { // Full vector store
                                _mm512_storeu_ps(C + (m_block_start + m_idx_rel + r) * ldc + current_n_col_abs, c_regs[r][nv]);
                            } else { // Partial vector store (tail) using mask
                                __mmask16 mask = (__mmask16)((1 << elements_in_vec) - 1);
                                _mm512_mask_storeu_ps(C + (m_block_start + m_idx_rel + r) * ldc + current_n_col_abs, mask, c_regs[r][nv]);
                            }
                        }
                    }
                } // end n_idx_rel loop
            } // end m_idx_rel loop
        } // end n_block_start loop
    } // end m_block_start loop (OpenMP)
}
#else
// Placeholder for AVX-512 if not compiled with AVX-512 support.
// This function will never be called if __builtin_cpu_supports("avx512f") returns false
// and the compiler was not instructed to target AVX-512.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    // This code path indicates that AVX-512 was not enabled at compile time (missing -mavx512f),
    // or if it was, the CPU does not support it, and the function was called directly.
    // The runtime dispatch in `gemm()` should prevent this from being called on a CPU without AVX-512.
    // However, if such a call occurs, falling back to AVX2 is a reasonable action.
    std::cerr << "Warning: AVX-512 kernel called but compiled without AVX-512 support "
              << "or on a non-AVX-512 CPU. Falling back to AVX2 kernel.\n";
    gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc); // Fallback to AVX2
}
#endif // defined(__AVX512F__)


// --- Top-level GEMM function with runtime dispatch ---
// This function serves as the primary API for GEMM. It inspects the CPU's capabilities
// at runtime using `__builtin_cpu_supports` (GCC/Clang extension) and dispatches
// the call to the most optimized available kernel (AVX-512, AVX2, or scalar).
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {

#if defined(__GNUC__) || defined(__clang__)
    // Prioritize AVX-512 if available at runtime AND compiled with AVX-512.
    // On Ryzen 7 6800HS, avx512f will be false.
    if (__builtin_cpu_supports("avx512f")) {
        // std::cout << "DEBUG: Dispatching to AVX-512 kernel.\n"; // Uncomment for debug info
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    } else if (__builtin_cpu_supports("avx2")) {
        // std::cout << "DEBUG: Dispatching to AVX2 kernel.\n"; // Uncomment for debug info
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    } else {
        // std::cout << "DEBUG: Dispatching to scalar kernel.\n"; // Uncomment for debug info
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
#else
    // Fallback if __builtin_cpu_supports is not available (e.g., MSVC).
    // In this scenario, the dispatch logic relies on compile-time feature flags.
    // Note: This path might behave differently if compiled with -march=native on some compilers.
    #if defined(__AVX512F__)
        // std::cout << "DEBUG: Dispatching to AVX-512 kernel (compile-time assumed).\n";
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    #elif defined(__AVX2__)
        // std::cout << "DEBUG: Dispatching to AVX2 kernel (compile-time assumed).\n";
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    #elif defined(__AVX__)
        // If only AVX is enabled at compile time, we fall back to scalar as we don't have
        // a dedicated AVX-only kernel (AVX2 includes AVX).
        // std::cout << "DEBUG: Dispatching to scalar kernel (AVX-only, no AVX2 kernel).\n";
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    #else
        // std::cout << "DEBUG: Dispatching to scalar kernel (no SIMD support detected).\n";
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    #endif
#endif
}

// --- Autotuner Harness ---
// This function benchmarks the different GEMM kernel implementations (scalar, AVX2, AVX-512)
// on a small problem size to report their relative performance on the current system.
// It does not dynamically adjust `BM/BN/BK` as those are `constexpr` within the kernels.
void autotune_gemm_report(const float* A_orig, const float* B_orig, float* C_orig,
                          int M, int N, int K,
                          int lda_orig, int ldb_orig, int ldc_orig) {
    // Suppress unused parameter warnings for parameters required by signature but not used in this specific function.
    (void)A_orig; (void)B_orig; (void)C_orig; (void)lda_orig; (void)ldb_orig; (void)ldc_orig;

    std::cout << "\nStarting autotuning report for M=" << M << ", N=" << N << ", K=" << K << "...\n";

    // Small problem size for quick warm-up and tuning
    int tune_M = std::min(M, 128);
    int tune_N = std::min(N, 128);
    int tune_K = std::min(K, 128);

    // Create small dummy matrices for tuning (using packed lda/ldb/ldc for simplicity)
    AlignedVector A_tune(static_cast<size_t>(tune_M) * tune_K);
    AlignedVector B_tune(static_cast<size_t>(tune_K) * tune_N);
    AlignedVector C_tune(static_cast<size_t>(tune_M) * tune_N);

    // Initialize with some deterministic data
    std::mt19937 gen(0); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for(size_t i=0; i < A_tune.size(); ++i) A_tune[i] = dis(gen);
    for(size_t i=0; i < B_tune.size(); ++i) B_tune[i] = dis(gen);
    
    // Perform a warm-up run to ensure CPU frequency scaling and cache are active
    // Use the scalar kernel for warm-up to avoid specific SIMD kernel overheads influencing results.
    gemm_scalar(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
    std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float)); // Clear for actual benchmark

    double best_time = std::numeric_limits<double>::max();
    std::string best_kernel_name = "None (No kernels benchmarked)";

#if defined(__GNUC__) || defined(__clang__)
    // Benchmark AVX-512 if supported at runtime
    if (__builtin_cpu_supports("avx512f")) {
        std::cout << "  Benchmarking AVX-512 kernel... ";
        auto start_time = std::chrono::high_resolution_clock::now();
        gemm_avx512(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double time = duration.count();
        std::cout << "Time: " << time * 1000 << " ms\n";
        if (time < best_time) {
            best_time = time;
            best_kernel_name = "AVX-512";
        }
        std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float)); // Clear C_tune for next benchmark
    }
    // Benchmark AVX2 if supported at runtime
    if (__builtin_cpu_supports("avx2")) {
        std::cout << "  Benchmarking AVX2 kernel... ";
        auto start_time = std::chrono::high_resolution_clock::now();
        gemm_avx2(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double time = duration.count();
        std::cout << "Time: " << time * 1000 << " ms\n";
        if (time < best_time) {
            best_time = time;
            best_kernel_name = "AVX2";
        }
        std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float)); // Clear C_tune for next benchmark
    }
    // Benchmark Scalar (always available)
    std::cout << "  Benchmarking Scalar kernel... ";
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm_scalar(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;
    double time = duration.count();
    std::cout << "Time: " << time * 1000 << " ms\n";
    if (time < best_time) {
        best_time = time;
        best_kernel_name = "Scalar";
    }
    std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float)); // Clear after use
#else
    // Fallback for non-GCC/Clang compilers (rely on compile-time flags)
    // In a production setup, one might use a CPUID library for runtime detection here.
    #if defined(__AVX512F__)
        std::cout << "  Benchmarking AVX-512 kernel (compile-time assumed)... ";
        auto start_time = std::chrono::high_resolution_clock::now();
        gemm_avx512(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double time = duration.count();
        std::cout << "Time: " << time * 1000 << " ms\n";
        if (time < best_time) {
            best_time = time;
            best_kernel_name = "AVX-512";
        }
        std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float));
    #elif defined(__AVX2__)
        std::cout << "  Benchmarking AVX2 kernel (compile-time assumed)... ";
        auto start_time = std::chrono::high_resolution_clock::now();
        gemm_avx2(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double time = duration.count();
        std::cout << "Time: " << time * 1000 << " ms\n";
        if (time < best_time) {
            best_time = time;
            best_kernel_name = "AVX2";
        }
        std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float));
    #else
        std::cout << "  Benchmarking Scalar kernel (no SIMD support detected)... ";
        auto start_time = std::chrono::high_resolution_clock::now();
        gemm_scalar(A_tune.data(), B_tune.data(), C_tune.data(), tune_M, tune_N, tune_K, tune_K, tune_N, tune_N);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double time = duration.count();
        std::cout << "Time: " << time * 1000 << " ms\n";
        if (time < best_time) {
            best_time = time;
            best_kernel_name = "Scalar";
        }
        std::memset(C_tune.data(), 0, C_tune.size() * sizeof(float));
    #endif
#endif
    std::cout << "Autotuning finished. Best performing kernel: " << best_kernel_name << ".\n";
}


// --- Main function for demonstration and benchmarking ---
int main(int argc, char* argv[]) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    unsigned int seed = static_cast<unsigned int>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    bool dump_matrices = false;
    int num_threads_arg = 0; // 0 means use OMP default
    bool run_autotune_report = true;

    std::cout << "GEMM Benchmarking Tool\n";
    std::cout << "----------------------\n";

    // Parse command-line arguments
    int current_arg_idx = 1;
    // Check for M, N, K as first three arguments
    if (argc > current_arg_idx + 2 &&
        std::all_of(argv + current_arg_idx, argv + current_arg_idx + 3, [](const char* s){ return std::isdigit(s[0]); })) {
        M = std::stoi(argv[current_arg_idx++]);
        N = std::stoi(argv[current_arg_idx++]);
        K = std::stoi(argv[current_arg_idx++]);
    } else if (argc > current_arg_idx && std::string(argv[current_arg_idx]) == "--help") {
        std::cout << "Usage: " << argv[0] << " [M N K] [--seed <val>] [--threads <num>] [--dump-matrices] [--no-autotune] [--help]\n";
        return 0;
    }

    // Parse optional flags
    for (; current_arg_idx < argc; ++current_arg_idx) {
        std::string arg = argv[current_arg_idx];
        if (arg == "--seed" && current_arg_idx + 1 < argc) {
            seed = std::stoul(argv[++current_arg_idx]);
        } else if (arg == "--threads" && current_arg_idx + 1 < argc) {
            num_threads_arg = std::stoi(argv[++current_arg_idx]);
        } else if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "--no-autotune") {
            run_autotune_report = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [M N K] [--seed <val>] [--threads <num>] [--dump-matrices] [--no-autotune] [--help]\n";
            return 0;
        } else {
            std::cerr << "Error: Unknown argument or invalid format: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " [M N K] [--seed <val>] [--threads <num>] [--dump-matrices] [--no-autotune] [--help]\n";
            return 1;
        }
    }

    if (num_threads_arg > 0) {
        omp_set_num_threads(num_threads_arg);
    }
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Using " << omp_get_max_threads() << " threads.\n";

    // For row-major matrices, leading dimension (ld) is typically the width of the matrix.
    // We assume dense, packed matrices without explicit padding here.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Allocate matrices using custom aligned allocator (64 bytes for AVX-512, sufficient for AVX2)
    AlignedVector A(static_cast<size_t>(M) * lda);
    AlignedVector B(static_cast<size_t>(K) * ldb);
    AlignedVector C_optimized(static_cast<size_t>(M) * ldc);
    AlignedVector C_reference(static_cast<size_t>(M) * ldc);

    // Initialize A and B with random values, C with zeros
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    std::cout << "Initializing matrices (seed: " << seed << ")...\n";
    for (size_t i = 0; i < static_cast<size_t>(M) * lda; ++i) A[i] = dis(gen);
    for (size_t i = 0; i < static_cast<size_t>(K) * ldb; ++i) B[i] = dis(gen);
    // C matrices are initialized to zero, as the GEMM kernels compute C = A * B.
    std::memset(C_optimized.data(), 0, static_cast<size_t>(M) * ldc * sizeof(float));
    std::memset(C_reference.data(), 0, static_cast<size_t>(M) * ldc * sizeof(float));

    if (dump_matrices) {
        std::cout << "Dumping input matrices A and B to 'workspace/'...\n";
        write_matrix_to_file("workspace/A.txt", A.data(), M, K, lda);
        write_matrix_to_file("workspace/B.txt", B.data(), K, N, ldb);
    }

    // --- Autotune and report ---
    if (run_autotune_report) {
        autotune_gemm_report(A.data(), B.data(), C_optimized.data(), M, N, K, lda, ldb, ldc);
    } else {
        std::cout << "Autotuning report skipped (--no-autotune).\n";
    }
    
    // --- Optimized GEMM (using runtime dispatch) ---
    std::cout << "\nRunning optimized GEMM (dispatched dynamically)...\n";
    auto start_opt = std::chrono::high_resolution_clock::now();
    // Calls the top-level gemm() function, which dispatches based on CPU features.
    gemm(A.data(), B.data(), C_optimized.data(), M, N, K, lda, ldb, ldc);
    auto end_opt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_opt = end_opt - start_opt;

    double gflops_opt = (2.0 * M * N * K) / (duration_opt.count() * 1e9);
    std::cout << "Optimized GEMM completed in " << duration_opt.count() * 1000 << " ms, "
              << gflops_opt << " GFLOP/s\n";

    if (dump_matrices) {
        std::cout << "Dumping output matrix C to 'workspace/C.txt'...\n";
        write_matrix_to_file("workspace/C.txt", C_optimized.data(), M, N, ldc);
    }

    // --- Reference GEMM for correctness check ---
    std::cout << "\nRunning scalar reference GEMM for correctness check...\n";
    auto start_ref = std::chrono::high_resolution_clock::now();
    gemm_scalar(A.data(), B.data(), C_reference.data(), M, N, K, lda, ldb, ldc);
    auto end_ref = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_ref = end_ref - start_ref;
    std::cout << "Reference GEMM completed in " << duration_ref.count() * 1000 << " ms.\n";

    // --- Correctness Check ---
    float max_diff = 0.0f;
    float max_relative_diff = 0.0f;
    for (size_t i = 0; i < static_cast<size_t>(M) * ldc; ++i) {
        float diff = std::abs(C_optimized[i] - C_reference[i]);
        max_diff = std::max(max_diff, diff);
        if (std::abs(C_reference[i]) > 1e-6f) { // Avoid division by zero for relative diff
            max_relative_diff = std::max(max_relative_diff, diff / std::abs(C_reference[i]));
        }
    }

    float tolerance = 1e-4f; // A reasonable tolerance for float comparisons (single precision)
    if (max_diff < tolerance) {
        std::cout << "\nCorrectness check PASSED. Max absolute difference: " << max_diff << "\n";
    } else {
        std::cout << "\nCorrectness check FAILED. Max absolute difference: " << max_diff
                  << ", Max relative difference: " << max_relative_diff << "\n";
    }

    return 0;
}