// Compile instructions:
//
// For AVX-512 target (Intel 11th Gen Core i7-1195G7 supports AVX-512):
// g++ -O3 -std=c++17 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
// clang++ -O3 -std=c++17 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//
// For AVX2 target (fallback if AVX-512 not desired or unavailable):
// g++ -O3 -std=c++17 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
// clang++ -O3 -std=c++17 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// For portable compilation (runtime dispatch will select best available ISA on host):
// g++ -O3 -std=c++17 -march=native -fopenmp gemm.cpp -o gemm_native
// clang++ -O3 -std=c++17 -march=native -fopenmp gemm.cpp -o gemm_native
//
// (Note: -march=native will enable all features of the compiling machine.
// __builtin_cpu_supports at runtime will ensure the correct kernel is used.)

#include <iostream>     // For std::cout, std::cerr
#include <vector>       // For std::vector
#include <cstring>      // For std::memset
#include <chrono>       // For std::chrono
#include <random>       // For std::mt19937, std::uniform_real_distribution
#include <cassert>      // For assert
#include <cmath>        // For std::fabs, std::sqrt
#include <string>       // For std::string, std::stoi, std::stoul
#include <fstream>      // For std::ofstream
#include <algorithm>    // For std::min, std::max, std::fill
#include <filesystem>   // For std::filesystem::create_directory (C++17)

#ifdef _OPENMP
#include <omp.h>        // For OpenMP directives
#endif

// Include SIMD intrinsics for x86-64
#ifdef __GNUC__ // For GCC and Clang __builtin_cpu_supports
#include <immintrin.h>  // Provides AVX, AVX2, AVX-512 intrinsics
#else
// If not GCC/Clang, __builtin_cpu_supports is not available.
// A fallback would involve platform-specific CPUID checks (e.g., __cpuid for MSVC).
// For this problem, we assume GCC/Clang on Linux.
#warning "Compiler not GCC/Clang. Runtime dispatch with __builtin_cpu_supports may not work."
#endif

// --- Autotuning Parameters ---
// These parameters are crucial for performance and should be tuned for the specific CPU.
// Target CPU: Intel i7-1195G7 (Tiger Lake)
// Cache hierarchy:
// L1d: 48KB (per core)
// L2:  1.25MB (per core)
// L3:  12MB (shared, 4 cores, 8 threads)
// A float is 4 bytes.
//
// Cache blocking factors (BM x BK, BK x BN, BM x BN blocks for A, B, C respectively).
// These define the size of blocks processed by a single thread to maximize L1/L2 cache reuse.
// The goal is to fit A_block (BM x BK) and B_block (BK x BN) into L2/L3, and C_block (BM x BN) into L1.
// BM * BK * 4 bytes + BK * BN * 4 bytes <= L2/L3 size (for streaming B_block or A_block)
// BM * BN * 4 bytes <= L1d size
constexpr int BM = 96;   // Block size for M dimension (rows of A, C). ~96*96*4B = ~36KB (for C), 96*256*4B = ~96KB (for A)
constexpr int BN = 128;  // Block size for N dimension (columns of B, C). ~128*128*4B = ~64KB (for C), 256*128*4B = ~128KB (for B)
constexpr int BK = 256;  // Block size for K dimension (inner-most loop).
                         // These values are chosen considering L1d (48KB) and L2 (1.25MB) sizes.
                         // BMxBN block (for C) is 96*128*4 bytes = 49152 bytes (~48KB), fits in L1d.
                         // BMxBK block (for A) is 96*256*4 bytes = 98304 bytes (~96KB), fits in L2.
                         // BKxBN block (for B) is 256*128*4 bytes = 131072 bytes (~128KB), fits in L2.
                         // Both A and B blocks for a given K iteration fit in L2 cache.

// Micro-kernel register blocking factors (MR x NR blocks of C).
// NR refers to the number of vector registers, not total floats.
//
// For AVX-512 (ZMM registers):
constexpr int VEC_SIZE_AVX512 = 16; // 16 floats per __m512 (64 bytes)
constexpr int MR_AVX512 = 6;        // Accumulate 6 rows of C
constexpr int NR_AVX512 = 4;        // Accumulate 4 vectors (64 floats) across N dimension for C
// Micro-kernel computes a 6x64 block of C. Uses 6 * 4 = 24 __m512 accumulators (zmm0-zmm23).
// This fits well within the 32 ZMM registers available on AVX-512 CPUs.
// (24 * 16 floats * 4 bytes/float) = 1536 bytes of C data in registers.

// For AVX2 (YMM registers):
constexpr int VEC_SIZE_AVX2 = 8;    // 8 floats per __m256 (32 bytes)
constexpr int MR_AVX2 = 4;          // Accumulate 4 rows of C
constexpr int NR_AVX2 = 4;          // Accumulate 4 vectors (32 floats) across N dimension for C
// Micro-kernel computes a 4x32 block of C. Uses 4 * 4 = 16 __m256 accumulators (ymm0-ymm15).
// This configuration uses all 16 YMM registers, which is optimal to avoid register spills.
// (16 * 8 floats * 4 bytes/float) = 512 bytes of C data in registers.

// Inner K loop unroll factor for micro-kernels.
// A value of 1 implies no explicit unrolling, letting the compiler handle it.
// Larger values (e.g., 2, 4) might expose more instruction-level parallelism,
// but can also increase register pressure. Keeping it at 1 for simplicity and
// relying on compiler auto-unrolling.
constexpr int UNROLL_K = 1;

// Default number of threads to use (can be overridden by OMP_NUM_THREADS or CLI)
constexpr int DEFAULT_NUM_THREADS = 8; // For a 4-core, 8-thread CPU (like i7-1195G7)

// Row-major convention: A[row * lda + col]
// All matrices A, B, C are assumed to be row-major.

// Helper for aligned memory allocation
template <typename T>
struct AlignedAllocator {
    // Alignment needs to be at least 64 bytes for AVX-512 to ensure optimal performance.
    // AVX2 uses 32-byte vectors, but 64-byte alignment is generally good practice for cache lines.
    static constexpr size_t ALIGNMENT = 64;

    using value_type = T;

    AlignedAllocator() = default;
    template <class U> AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        T* ptr = nullptr;
        // posix_memalign is standard on Linux/macOS. For Windows, _aligned_malloc.
        if (posix_memalign(reinterpret_cast<void**>(&ptr), ALIGNMENT, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void deallocate(T* ptr, std::size_t) noexcept {
        free(ptr);
    }
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return true; }
template <typename T, typename U>
bool operator!=(const AlignedAllocator<T>&, const AlignedAllocator<U>&) { return false; }


// --- Matrix Utility Functions ---

// Writes a matrix to a text file. Handles leading dimension.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing.\n";
        return;
    }

    ofs.precision(6);
    ofs << std::fixed;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}


// --- Scalar GEMM Implementation (Reference) ---

// `gemm_scalar` computes C = A * B using a naive triple loop.
// It serves as a correctness reference and a fallback for unsupported architectures.
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = sum;
        }
    }
}


// --- AVX2 GEMM Implementation ---

#if defined(__AVX2__) && defined(__FMA__)

// `gemm_avx2` implements GEMM using AVX2 and FMA intrinsics with cache blocking and register blocking.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Helper micro-kernel function for a small block of C (MR_AVX2 x (NR_AVX2 * VEC_SIZE_AVX2))
    // This function accumulates into C_base.
    auto avx2_inner_micro_kernel = [&](int current_m, int current_n, int current_k_start,
                                      int total_M, int total_N, int block_k,
                                      const float* A_base, const float* B_base, float* C_base) {

        // C accumulators, initialized to zero. MR_AVX2 rows x NR_AVX2 vectors.
        // E.g., for MR_AVX2=4, NR_AVX2=4, this is 16 __m256 registers.
        __m256 c_regs[MR_AVX2][NR_AVX2];
        for (int i = 0; i < MR_AVX2; ++i) {
            for (int j = 0; j < NR_AVX2; ++j) {
                c_regs[i][j] = _mm256_setzero_ps();
            }
        }

        // K-loop: Perform multiplications and accumulations for the `block_k` dimension.
        // UNROLL_K is 1, so no explicit unrolling here. Compiler may unroll.
        for (int k = 0; k < block_k; ++k) {
            const int k_abs = current_k_start + k; // Absolute K index

            // Load A values (broadcast scalar to all elements of a vector)
            // A elements are A[m_abs][k_abs]
            __m256 a_vals[MR_AVX2];
            for (int r = 0; r < MR_AVX2; ++r) {
                const int m_abs = current_m + r; // Absolute M index
                if (m_abs < total_M) { // Boundary check for M
                    // Broadcast A[m_abs][k_abs] element across the vector
                    a_vals[r] = _mm256_broadcast_ss(A_base + m_abs * lda + k_abs);
                } else {
                    a_vals[r] = _mm256_setzero_ps(); // If out of bounds, effectively multiply by zero
                }
            }

            // Load B vectors
            // B elements are B[k_abs][n_abs...n_abs+VEC_SIZE_AVX2-1]
            __m256 b_vecs[NR_AVX2];
            for (int v = 0; v < NR_AVX2; ++v) {
                const int n_abs = current_n + v * VEC_SIZE_AVX2; // Absolute N index for vector start
                int elements_to_load_b = 0;
                if (n_abs < total_N) {
                    elements_to_load_b = std::min(VEC_SIZE_AVX2, total_N - n_abs);
                }

                if (elements_to_load_b == VEC_SIZE_AVX2) {
                    // Full vector load, can use unaligned load
                    b_vecs[v] = _mm256_loadu_ps(B_base + k_abs * ldb + n_abs);
                } else if (elements_to_load_b > 0) {
                    // Partial vector load: copy to a temporary aligned buffer, then load from it.
                    // This handles unaligned and partial vectors safely for AVX2.
                    alignas(32) float temp_b_buf[VEC_SIZE_AVX2] = {0.0f}; // Initialize to 0
                    for (int i = 0; i < elements_to_load_b; ++i) {
                        temp_b_buf[i] = B_base[k_abs * ldb + n_abs + i];
                    }
                    b_vecs[v] = _mm256_load_ps(temp_b_buf); // Load from aligned buffer
                } else { // n_abs >= total_N
                    b_vecs[v] = _mm256_setzero_ps(); // Out of bounds, treat as zero
                }
            }

            // FMA operations: c_regs += a_vals * b_vecs
            // Each c_reg[r][v] accumulates products for C[current_m+r][current_n + v*VEC_SIZE ... ]
            for (int r = 0; r < MR_AVX2; ++r) {
                for (int v = 0; v < NR_AVX2; ++v) {
                    c_regs[r][v] = _mm256_fmadd_ps(a_vals[r], b_vecs[v], c_regs[r][v]);
                }
            }
            // Optional prefetching for next K iteration or next block.
            // _mm_prefetch((const char*)(A_base + (current_m + MR_AVX2) * lda + k_abs), _MM_HINT_T0); // Prefetch A for next M-row group
            // _mm_prefetch((const char*)(B_base + (k_abs + 1) * ldb + current_n), _MM_HINT_T0); // Prefetch B for next K-row
        }

        // Add accumulated C values to existing C and store back
        for (int r = 0; r < MR_AVX2; ++r) {
            for (int v = 0; v < NR_AVX2; ++v) {
                const int m_abs = current_m + r;
                const int n_abs = current_n + v * VEC_SIZE_AVX2;

                if (m_abs < total_M && n_abs < total_N) { // Boundary check for C store
                    __m256 c_old_val;
                    int n_remaining = total_N - n_abs; // Number of valid elements in this vector
                    
                    if (n_remaining < VEC_SIZE_AVX2) {
                        // Partial vector C load for accumulation, then masked store.
                        // AVX2 does not have `_mm256_maskz_loadu_ps`, so we load into a temporary buffer.
                        alignas(32) float temp_c_buf[VEC_SIZE_AVX2];
                        std::fill(temp_c_buf, temp_c_buf + VEC_SIZE_AVX2, 0.0f); // Zero init to handle elements beyond n_remaining
                        for(int i = 0; i < n_remaining; ++i) {
                            temp_c_buf[i] = C_base[m_abs * ldc + n_abs + i];
                        }
                        c_old_val = _mm256_load_ps(temp_c_buf); // Load existing C from aligned buffer

                        __m256 c_new = _mm256_add_ps(c_old_val, c_regs[r][v]);

                        // Create an integer mask for _mm256_maskstore_ps.
                        // Each 32-bit lane in the mask vector is either all ones (-1) to enable, or all zeros (0) to disable.
                        unsigned int mask_val_bitmask = (1U << n_remaining) - 1; // e.g., if remaining=5, bitmask=0b0011111
                        __m256i mask_i = _mm256_set_epi32(
                            (mask_val_bitmask & (1<<7)) ? -1 : 0, (mask_val_bitmask & (1<<6)) ? -1 : 0,
                            (mask_val_bitmask & (1<<5)) ? -1 : 0, (mask_val_bitmask & (1<<4)) ? -1 : 0,
                            (mask_val_bitmask & (1<<3)) ? -1 : 0, (mask_val_bitmask & (1<<2)) ? -1 : 0,
                            (mask_val_bitmask & (1<<1)) ? -1 : 0, (mask_val_bitmask & (1<<0)) ? -1 : 0
                        );
                        _mm256_maskstore_ps(C_base + m_abs * ldc + n_abs, mask_i, c_new);
                    } else {
                        // Full vector load and store
                        c_old_val = _mm256_loadu_ps(C_base + m_abs * ldc + n_abs);
                        __m256 c_new = _mm256_add_ps(c_old_val, c_regs[r][v]);
                        _mm256_storeu_ps(C_base + m_abs * ldc + n_abs, c_new);
                    }
                }
            }
        }
    }; // end avx2_inner_micro_kernel lambda

    // Outer loops for cache blocking (M, N, K dimensions)
    // Parallelize over M and N blocks for load balancing across threads.
#ifdef _OPENMP
    // Using dynamic schedule to help with load balancing for potentially uneven tail blocks
    // and to better utilize cores given varying system loads.
    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(DEFAULT_NUM_THREADS)
#endif
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {

                // Current block sizes (can be smaller at matrix boundaries)
                int current_bm = std::min(BM, M - m_block_start);
                int current_bn = std::min(BN, N - n_block_start);
                int current_bk = std::min(BK, K - k_block_start);

                // Loops over micro-kernels within the current BM x BN block
                for (int m_micro_start = 0; m_micro_start < current_bm; m_micro_start += MR_AVX2) {
                    for (int n_micro_start = 0; n_micro_start < current_bn; n_micro_start += NR_AVX2 * VEC_SIZE_AVX2) {
                        
                        // Call the micro-kernel. It handles its own M/N boundary conditions
                        // (within the micro-block, relative to total M/N).
                        avx2_inner_micro_kernel(m_block_start + m_micro_start,
                                                n_block_start + n_micro_start,
                                                k_block_start,
                                                M, N, current_bk, // Pass total M, N for global bounds checks
                                                A, B, C);
                    }
                }
            }
        }
    }
}

#endif // __AVX2__ && __FMA__


// --- AVX-512 GEMM Implementation ---

#if defined(__AVX512F__) && defined(__FMA__)

// `gemm_avx512` implements GEMM using AVX-512 and FMA intrinsics with cache blocking and register blocking.
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    // Helper micro-kernel function for a small block of C (MR_AVX512 x (NR_AVX512 * VEC_SIZE_AVX512))
    // This function accumulates into C_base.
    auto avx512_inner_micro_kernel = [&](int current_m, int current_n, int current_k_start,
                                         int total_M, int total_N, int block_k,
                                         const float* A_base, const float* B_base, float* C_base) {

        // C accumulators, initialized to zero. MR_AVX512 rows x NR_AVX512 vectors.
        // E.g., for MR_AVX512=6, NR_AVX512=4, this is 24 __m512 registers.
        __m512 c_regs[MR_AVX512][NR_AVX512];
        for (int i = 0; i < MR_AVX512; ++i) {
            for (int j = 0; j < NR_AVX512; ++j) {
                c_regs[i][j] = _mm512_setzero_ps();
            }
        }

        // K-loop: Perform multiplications and accumulations for the `block_k` dimension.
        // UNROLL_K is 1, so no explicit unrolling here. Compiler may unroll.
        for (int k = 0; k < block_k; ++k) {
            const int k_abs = current_k_start + k; // Absolute K index

            // Load A values (broadcast scalar to all elements of a vector)
            // A elements are A[m_abs][k_abs]
            __m512 a_vals[MR_AVX512];
            for (int r = 0; r < MR_AVX512; ++r) {
                const int m_abs = current_m + r; // Absolute M index
                if (m_abs < total_M) { // Boundary check for M
                    // Broadcast A[m_abs][k_abs]
                    a_vals[r] = _mm512_set1_ps(*(A_base + m_abs * lda + k_abs)); // Correct intrinsic for broadcasting a float
                } else {
                    a_vals[r] = _mm512_setzero_ps(); // If out of bounds, effectively multiply by zero
                }
            }

            // Load B vectors
            // B elements are B[k_abs][n_abs...n_abs+VEC_SIZE_AVX512-1]
            __m512 b_vecs[NR_AVX512];
            for (int v = 0; v < NR_AVX512; ++v) {
                const int n_abs = current_n + v * VEC_SIZE_AVX512; // Absolute N index for vector start
                if (n_abs < total_N) { // Boundary check for N
                    // If n_abs + VEC_SIZE_AVX512 > total_N, it's a partial vector load.
                    // Use mask to load only valid elements, zeroing the rest with _mm512_maskz_loadu_ps.
                    unsigned int n_remaining = total_N - n_abs;
                    if (n_remaining < VEC_SIZE_AVX512) {
                        __mmask16 k_mask = (__mmask16)((1U << n_remaining) - 1); // Create a bitmask for the valid elements
                        b_vecs[v] = _mm512_maskz_loadu_ps(k_mask, B_base + k_abs * ldb + n_abs);
                    } else {
                        b_vecs[v] = _mm512_loadu_ps(B_base + k_abs * ldb + n_abs);
                    }
                } else {
                    b_vecs[v] = _mm512_setzero_ps(); // Out of bounds, treat as zero
                }
            }

            // FMA operations: c_regs += a_vals * b_vecs
            // Each c_reg[r][v] accumulates products for C[current_m+r][current_n + v*VEC_SIZE ... ]
            for (int r = 0; r < MR_AVX512; ++r) {
                for (int v = 0; v < NR_AVX512; ++v) {
                    c_regs[r][v] = _mm512_fmadd_ps(a_vals[r], b_vecs[v], c_regs[r][v]);
                }
            }
            // Optional prefetching for next K iteration or next block.
            // _mm_prefetch((const char*)(A_base + (current_m + MR_AVX512) * lda + k_abs), _MM_HINT_T0); // Prefetch A for next M-row group
            // _mm_prefetch((const char*)(B_base + (k_abs + 1) * ldb + current_n), _MM_HINT_T0); // Prefetch B for next K-row
        }

        // Add accumulated C values to existing C and store back
        for (int r = 0; r < MR_AVX512; ++r) {
            for (int v = 0; v < NR_AVX512; ++v) {
                const int m_abs = current_m + r;
                const int n_abs = current_n + v * VEC_SIZE_AVX512;

                if (m_abs < total_M && n_abs < total_N) { // Boundary check for C store
                    __m512 c_old_val;
                    unsigned int n_remaining = total_N - n_abs; // Number of valid elements in this vector
                    
                    if (n_remaining < VEC_SIZE_AVX512) {
                        // Masked load for C_old and masked store for C_new
                        __mmask16 k_mask = (__mmask16)((1U << n_remaining) - 1); // Create a bitmask for the valid elements
                        c_old_val = _mm512_maskz_loadu_ps(k_mask, C_base + m_abs * ldc + n_abs); // Load existing C with mask, zeroing outside mask
                        c_regs[r][v] = _mm512_add_ps(c_old_val, c_regs[r][v]); // Add new accumulation
                        _mm512_mask_storeu_ps(C_base + m_abs * ldc + n_abs, k_mask, c_regs[r][v]); // Store with mask
                    } else {
                        // Full vector load and store
                        c_old_val = _mm512_loadu_ps(C_base + m_abs * ldc + n_abs);
                        c_regs[r][v] = _mm512_add_ps(c_old_val, c_regs[r][v]);
                        _mm512_storeu_ps(C_base + m_abs * ldc + n_abs, c_regs[r][v]);
                    }
                }
            }
        }
    }; // end avx512_inner_micro_kernel lambda

    // Outer loops for cache blocking (M, N, K dimensions)
    // Parallelize over M and N blocks for load balancing across threads.
#ifdef _OPENMP
    // Using dynamic schedule to help with load balancing for potentially uneven tail blocks
    // and to better utilize cores given varying system loads.
    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(DEFAULT_NUM_THREADS)
#endif
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {

                // Current block sizes (can be smaller at matrix boundaries)
                int current_bm = std::min(BM, M - m_block_start);
                int current_bn = std::min(BN, N - n_block_start);
                int current_bk = std::min(BK, K - k_block_start);

                // Loops over micro-kernels within the current BM x BN block
                for (int m_micro_start = 0; m_micro_start < current_bm; m_micro_start += MR_AVX512) {
                    for (int n_micro_start = 0; n_micro_start < current_bn; n_micro_start += NR_AVX512 * VEC_SIZE_AVX512) {
                        
                        // Call the micro-kernel. It handles its own M/N boundary conditions
                        // (within the micro-block, relative to total M/N).
                        avx512_inner_micro_kernel(m_block_start + m_micro_start,
                                                  n_block_start + n_micro_start,
                                                  k_block_start,
                                                  M, N, current_bk, // Pass total M, N for global bounds checks
                                                  A, B, C);
                    }
                }
            }
        }
    }
}

#endif // __AVX512F__ && __FMA__


// --- Top-level GEMM Function (Runtime Dispatch) ---

// Function pointer for runtime dispatch, initialized once.
void (*gemm_impl_ptr)(const float*, const float*, float*, int, int, int, int, int, int) = nullptr;

// `gemm` is the primary entry point, performing runtime CPU feature detection
// and dispatching to the most optimized kernel available.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    if (gemm_impl_ptr == nullptr) {
        // One-time dispatch initialization.
        // Handle empty matrices upfront or let the chosen kernel manage (scalar does).
        if (M <= 0 || N <= 0 || K <= 0) {
             std::cout << "GEMM dimensions are non-positive (" << M << "x" << N << "x" << K << "), returning early via scalar kernel.\n";
             gemm_impl_ptr = gemm_scalar; // Assign scalar and let it handle 0-dim
        } else {
            std::cout << "Auto-detecting optimal GEMM kernel...\n";
            // __builtin_cpu_supports is a GCC/Clang extension for runtime CPU feature detection.
            // It safely queries the CPU capabilities.
            #if defined(__AVX512F__) && defined(__FMA__)
            if (__builtin_cpu_supports("avx512f")) {
                gemm_impl_ptr = gemm_avx512;
                std::cout << "  Using AVX-512 kernel.\n";
            } else
            #endif
            #if defined(__AVX2__) && defined(__FMA__)
            if (__builtin_cpu_supports("avx2")) {
                gemm_impl_ptr = gemm_avx2;
                std::cout << "  Using AVX2 kernel.\n";
            } else
            #endif
            {
                gemm_impl_ptr = gemm_scalar;
                std::cout << "  Using scalar kernel (no AVX2/AVX-512 support detected or compiled out).\n";
            }
        }
    }
    // Execute the chosen implementation
    gemm_impl_ptr(A, B, C, M, N, K, lda, ldb, ldc);
}


// --- Main Function (CLI parsing, timing, verification) ---

int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    unsigned int seed = 42;
    int num_threads = DEFAULT_NUM_THREADS;
    bool dump_matrices = false;

    // Parse command line arguments
    // First, attempt to parse M, N, K as positional arguments
    int arg_idx = 1;
    if (argc > arg_idx) {
        try { M = std::stoi(argv[arg_idx]); arg_idx++; } catch(...) { /* Not an integer, skip to flag parsing */ }
    }
    if (argc > arg_idx) {
        try { N = std::stoi(argv[arg_idx]); arg_idx++; } catch(...) { /* Not an integer, skip to flag parsing */ }
    }
    if (argc > arg_idx) {
        try { K = std::stoi(argv[arg_idx]); arg_idx++; } catch(...) { /* Not an integer, skip to flag parsing */ }
    }

    // Then parse flags, allowing them to override positional arguments
    for (; arg_idx < argc; ++arg_idx) {
        std::string arg = argv[arg_idx];
        if (arg == "-M" && arg_idx + 1 < argc) {
            M = std::stoi(argv[++arg_idx]);
        } else if (arg == "-N" && arg_idx + 1 < argc) {
            N = std::stoi(argv[++arg_idx]);
        } else if (arg == "-K" && arg_idx + 1 < argc) {
            K = std::stoi(argv[++arg_idx]);
        } else if (arg == "-s" && arg_idx + 1 < argc) {
            seed = std::stoul(argv[++arg_idx]);
        } else if (arg == "-t" && arg_idx + 1 < argc) {
            num_threads = std::stoi(argv[++arg_idx]);
        } else if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [M] [N] [K] [-M <rows>] [-N <cols>] [-K <inner>] [-s <seed>] [-t <threads>] [--dump-matrices]\n";
            std::cout << "  M, N, K: Positional arguments for matrix dimensions (default: 1024 1024 1024)\n";
            std::cout << "  -M <rows>: Number of rows in A and C (overrides positional M)\n";
            std::cout << "  -N <cols>: Number of columns in B and C (overrides positional N)\n";
            std::cout << "  -K <inner>: Inner dimension for A and B (overrides positional K)\n";
            std::cout << "  -s <seed>: Random seed (default: 42)\n";
            std::cout << "  -t <threads>: Number of OpenMP threads (default: " << DEFAULT_NUM_THREADS << ")\n";
            std::cout << "  --dump-matrices: Write A, B, C matrices to files in 'workspace/'\n";
            return 0;
        } else {
            std::cerr << "Unknown argument or invalid format: " << arg << ". Use --help for usage.\n";
            return 1;
        }
    }

    std::cout << "Running GEMM with M=" << M << ", N=" << N << ", K=" << K << ", Threads=" << num_threads << "\n";

    // Set OpenMP threads, potentially overriding environment variable OMP_NUM_THREADS
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
#else
    if (num_threads > 1) {
        std::cerr << "Warning: OpenMP not enabled during compilation. Falling back to single thread.\n";
    }
#endif

    // Allocate matrices using aligned allocator
    using FloatVector = std::vector<float, AlignedAllocator<float>>;
    FloatVector A_vec(static_cast<size_t>(M) * K);
    FloatVector B_vec(static_cast<size_t>(K) * N);
    FloatVector C_vec(static_cast<size_t>(M) * N);
    FloatVector C_ref_vec(static_cast<size_t>(M) * N); // For correctness check

    // Use raw pointers for GEMM functions to match signature
    float* A = A_vec.data();
    float* B = B_vec.data();
    float* C = C_vec.data();
    float* C_ref = C_ref_vec.data();

    // For simplicity, assume lda = K, ldb = N, ldc = N (dense row-major)
    // If leading dimensions were different, e.g., for sub-matrices, this would be adjusted.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Initialize matrices with random data between -1.0 and 1.0
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < A_vec.size(); ++i) A[i] = dis(gen);
    for (size_t i = 0; i < B_vec.size(); ++i) B[i] = dis(gen);
    std::memset(C, 0, C_vec.size() * sizeof(float));
    std::memset(C_ref, 0, C_ref_vec.size() * sizeof(float));

    // Create workspace directory if dumping matrices
    if (dump_matrices) {
        std::filesystem::path workspace_dir("workspace");
        if (!std::filesystem::exists(workspace_dir)) {
            try {
                std::filesystem::create_directory(workspace_dir);
                std::cout << "Created directory: " << workspace_dir << "\n";
            } catch (const std::filesystem::filesystem_error& e) {
                std::cerr << "Error creating directory " << workspace_dir << ": " << e.what() << "\n";
                return 1;
            }
        }
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt\n";
    }

    // Warm-up run for dispatch and cache (especially important for dispatch logic to initialize gemm_impl_ptr)
    std::cout << "Starting warm-up run...\n";
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    std::memset(C, 0, C_vec.size() * sizeof(float)); // Clear C for actual timed run
    std::cout << "Warm-up complete. Starting timed run...\n";

    // Time the optimized GEMM
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();

    // Calculate execution time and GFLOP/s
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    long long operations = 2LL * M * N * K; // 2 operations per FMA (1 mul + 1 add)
    double gflops = static_cast<double>(operations) / (elapsed_ms * 1e6);

    std::cout << "GEMM computation finished.\n";
    std::cout << "Time taken: " << elapsed_ms << " ms\n";
    std::cout << "Performance: " << gflops << " GFLOP/s\n";

    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Matrix C dumped to workspace/C.txt\n";
    }

    // Verify correctness with scalar GEMM
    std::cout << "Verifying correctness with scalar GEMM...\n";
    gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

    float max_diff = 0.0f;
    // Calculate sum of squares difference for a more robust relative error metric
    double sq_diff_sum = 0.0;
    double sq_norm_C_ref = 0.0;

    for (size_t i = 0; i < C_vec.size(); ++i) {
        float diff = std::fabs(C[i] - C_ref[i]);
        max_diff = std::max(max_diff, diff);
        sq_diff_sum += static_cast<double>(diff * diff);
        sq_norm_C_ref += static_cast<double>(C_ref[i] * C_ref[i]);
    }

    const float abs_tolerance = 1e-3f; // Absolute tolerance for float comparison
    // Calculate relative error if C_ref is not zero
    double relative_error = 0.0;
    if (sq_norm_C_ref > 0) {
        relative_error = std::sqrt(sq_diff_sum) / std::sqrt(sq_norm_C_ref);
    }
    
    // A common rule of thumb for relative error for single precision floats is around 1e-6 to 1e-5.
    // For large matrices and many accumulations, this can degrade. Using 1e-3 for robustness.
    const double rel_tolerance = 1e-3; // Relative tolerance

    if (max_diff < abs_tolerance || relative_error < rel_tolerance) {
        std::cout << "Correctness check PASSED.\n";
        std::cout << "  Max absolute difference: " << max_diff << "\n";
        std::cout << "  Relative error (Frobenius norm): " << relative_error << "\n";
    } else {
        std::cerr << "Correctness check FAILED.\n";
        std::cerr << "  Max absolute difference: " << max_diff << " (tolerance: " << abs_tolerance << ")\n";
        std::cerr << "  Relative error (Frobenius norm): " << relative_error << " (tolerance: " << rel_tolerance << ")\n";
        // Optionally print a few differing values for debugging
        int diff_count = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                if (std::fabs(C[i * ldc + j] - C_ref[i * ldc + j]) >= abs_tolerance) {
                    if (diff_count < 5) { // Print first 5 differences
                        std::cerr << "C[" << i << "," << j << "]: Optimized=" << C[i * ldc + j] << ", Scalar=" << C_ref[i * ldc + j] << ", Diff=" << std::fabs(C[i * ldc + j] - C_ref[i * ldc + j]) << "\n";
                    }
                    diff_count++;
                }
            }
        }
        std::cerr << "Total differing elements (above abs_tolerance): " << diff_count << "\n";
        return 1; // Indicate failure
    }

    return 0;
}