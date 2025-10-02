// g++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native
// clang++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512_clang
// clang++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2_clang
// clang++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native_clang

#include <iostream>
#include <vector>
#include <cstring>   // For memcpy, strerror
#include <chrono>    // For timing
#include <random>    // For random matrix initialization
#include <cassert>   // For assertions
#include <fstream>   // For file I/O
#include <string>
#include <algorithm> // For std::min, std::max, std::fill
#include <cmath>     // For std::abs, std::sqrt
#include <limits>    // For std::numeric_limits
#include <cstdlib>   // For posix_memalign, aligned_alloc, free
#include <new>       // For std::bad_alloc
#include <errno.h>   // For errno, used by mkdir

// Intrinsics
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE__)
#include <immintrin.h>
#endif

// OpenMP
#ifdef _OPENMP
#include <omp.h>
#endif

// For mkdir (Linux specific)
#include <sys/stat.h>
#include <sys/types.h>

// --- Autotuning Parameters ---
// These parameters define the tiling strategy and micro-kernel dimensions.
// They are chosen based on typical cache sizes (L1/L2/L3) and register availability.
// For AMD Ryzen 7 6800HS:
// L1d: 32KB per core (data cache)
// L2d: 512KB per core (unified cache)
// L3: 16MB shared (unified cache)
//
// VEC_WIDTH: Number of floats per SIMD register (8 for AVX2, 16 for AVX-512)
// MR: Micro-kernel M-dimension (number of rows of C computed simultaneously)
// NR: Micro-kernel N-dimension (number of columns of C computed simultaneously), must be a multiple of VEC_WIDTH
// UNROLL_K: Inner K-loop unroll factor
// BM, BN, BK: Outer tile sizes for M, N, K dimensions respectively. These aim to fit in L2/L3 cache.

// AVX2 specific micro-kernel parameters (tuned for Ryzen 6800HS)
constexpr int VEC_WIDTH_AVX2 = 8; // __m256 holds 8 floats
constexpr int MR_AVX2 = 6;        // Accumulate 6 rows of C (fits 6*256-bit registers = 192 bytes for C, plus A & B loads)
constexpr int NR_AVX2 = 16;       // Accumulate 16 columns of C (2 __m256 registers, 2*8=16)
constexpr int UNROLL_K_AVX2 = 4;  // Unroll K loop by 4 (to hide FMA latency and increase instruction-level parallelism)

// AVX-512 specific micro-kernel parameters (provided as requested, but not used on Ryzen 6800HS)
constexpr int VEC_WIDTH_AVX512 = 16; // __m512 holds 16 floats
constexpr int MR_AVX512 = 6;         // Accumulate 6 rows of C
constexpr int NR_AVX512 = 16;        // Accumulate 16 columns of C (1 __m512 register, 1*16=16)
constexpr int UNROLL_K_AVX512 = 4;   // Unroll K loop by 4

// Default outer tile sizes (can be adjusted via command line or autotuner)
// These are chosen to be large enough to amortize overhead but small enough to fit in L3 cache (16MB).
// A block of A: BM * BK * 4 bytes
// A block of B: BK * BN * 4 bytes
// A block of C: BM * BN * 4 bytes
// For BM=128, BN=256, BK=128:
// A block: 128*128*4 = 64KB (fits L2)
// B block: 128*256*4 = 128KB (fits L2)
// C block: 128*256*4 = 128KB (fits L2)
// These combined can fit L3.
constexpr int DEFAULT_BM = 128;
constexpr int DEFAULT_BN = 256;
constexpr int DEFAULT_BK = 128;

// --- Helper for aligned memory allocation ---
// Custom allocator to ensure specified alignment for vector elements.
// This allocator fully conforms to the C++11 Allocator requirements,
// ensuring compatibility with std::vector across various standard library implementations.
// It uses a pragmatic approach for `rebind` to maximize compatibility with
// different `libstdc++` versions (some might expect `type`, others `other`).
template <typename T, size_t Alignment>
struct AlignedAllocator {
    static_assert(Alignment > 0 && (Alignment & (Alignment - 1)) == 0, "Alignment must be a power of two and positive");
    static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");

    // Standard allocator member types
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // The rebind structure is CRITICAL for std::vector.
    // Providing both `type` and `other` handles potential compiler/library variations.
    template <typename U>
    struct rebind {
        using type = AlignedAllocator<U, Alignment>;  // For older/specific libstdc++ versions that might look for 'type'
        using other = AlignedAllocator<U, Alignment>; // C++11 standard requires 'other'
    };

    // Constructors
    AlignedAllocator() noexcept = default;
    // Copy constructor template for rebinding (needed for stateless allocators)
    template <typename U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    // Allocation/Deallocation functions
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<size_type>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = nullptr;
        // Use std::aligned_alloc for C++17 and later.
        // Fallback to posix_memalign for older C++ standards or when std::aligned_alloc is not available.
#if __cplusplus >= 201703L
        ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        if (ptr == nullptr) { // std::aligned_alloc returns nullptr on failure
            throw std::bad_alloc();
        }
#else
        // posix_memalign is POSIX standard, widely available on Linux/Unix systems.
        // It returns an error code (0 for success), unlike aligned_alloc which returns nullptr.
        int ret = posix_memalign(&ptr, Alignment, n * sizeof(T));
        if (ret != 0) {
            std::cerr << "posix_memalign failed: " << strerror(ret) << std::endl;
            throw std::bad_alloc();
        }
#endif
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        std::free(p); // free works for both aligned_alloc and posix_memalign
    }

    // Comparison operators are required for allocators.
    // For stateless allocators (like this one), they are always equal.
    bool operator==(const AlignedAllocator& other) const noexcept { return true; }
    bool operator!=(const AlignedAllocator& other) const noexcept { return !(*this == other); }
};

// --- Matrix I/O Helper ---
// Writes a matrix to a text file in space-separated format.
// Handles leading dimension (ld) correctly for row-major matrices.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file.precision(6); // Set precision for float output
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[static_cast<size_t>(i) * ld + j] << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

// --- Scalar GEMM Implementation (Reference) ---
// Computes C = A * B using a simple triple-nested loop.
// A is M x K (lda = K for row-major)
// B is K x N (ldb = N for row-major)
// C is M x N (ldc = N for row-major)
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<size_t>(i) * lda + k] * B[static_cast<size_t>(k) * ldb + j];
            }
            C[static_cast<size_t>(i) * ldc + j] = sum;
        }
    }
}

// --- AVX2 GEMM Implementation ---
#if defined(__AVX2__) && defined(__FMA__)
// Micro-kernel for AVX2. Computes MR_AVX2 rows x NR_AVX2 columns of C.
// This micro-kernel assumes exact dimensions MR_AVX2 x NR_AVX2 for optimal register usage.
// It loads existing C values, accumulates, and stores back.
// Parameters:
//   K_iter: Number of iterations for the inner K loop.
//   C_ptr: Pointer to the current micro-panel of C (MR_AVX2 rows, NR_AVX2 cols).
//   ldc: Leading dimension of C.
//   A_ptr: Pointer to the current micro-panel of A (MR_AVX2 rows, K-dimension).
//   lda: Leading dimension of A.
//   packed_B_ptr: Pointer to the packed B block for the current K-slice and N-offset.
//                 B is packed in K-major, so `packed_B_ptr + k_offset * packed_B_stride` accesses B[k][n_start...].
//   packed_B_stride: The actual N-dimension (cur_N from the outer loop) of the packed B buffer.
void sgemm_avx2_ukernel(int K_iter, float* C_ptr, int ldc,
                        const float* A_ptr, int lda,
                        const float* packed_B_ptr, int packed_B_stride) {

    // Initialize C accumulators by loading existing values from C matrix.
    // This is crucial for correctly accumulating across K-blocks.
    // c[row_idx][col_vec_idx] corresponds to C[row_idx][col_vec_idx * VEC_WIDTH_AVX2 ...].
    // We use MR_AVX2 rows and (NR_AVX2 / VEC_WIDTH_AVX2) vector columns (2 for NR_AVX2=16).
    __m256 c[MR_AVX2][NR_AVX2 / VEC_WIDTH_AVX2];
    for (int i = 0; i < MR_AVX2; ++i) {
        for (int j_vec = 0; j_vec < NR_AVX2 / VEC_WIDTH_AVX2; ++j_vec) {
            // Load existing C values from the current C micro-panel into AVX2 accumulators.
            c[i][j_vec] = _mm256_loadu_ps(C_ptr + static_cast<size_t>(i) * ldc + static_cast<size_t>(j_vec) * VEC_WIDTH_AVX2);
        }
    }

    // Pointers to traverse A and packed_B in K-dimension.
    const float* a_cur = A_ptr;
    const float* b_cur_packed_k_start = packed_B_ptr; // Points to the start of the K-row for the current N-block within packed B
    
    // Loop over K dimension with unrolling by UNROLL_K_AVX2.
    for (int k_idx = 0; k_idx < K_iter; k_idx += UNROLL_K_AVX2) {
        // Handle K tails for the unrolling factor (if K_iter is not divisible by UNROLL_K_AVX2).
        int k_eff = std::min(UNROLL_K_AVX2, K_iter - k_idx);

        for (int uk = 0; uk < k_eff; ++uk) {
            // Load A values: MR_AVX2 scalar values.
            // A is row-major. A_panel_ptr points to A[m_block + i][k_block].
            // a_cur[i * lda + uk] accesses A[m_block + i + i_ukernel_row][k_block + k_idx + uk].
            // Each A value is broadcasted to an __m256 register.
            __m256 a_vals[MR_AVX2];
            for (int i = 0; i < MR_AVX2; ++i) {
                a_vals[i] = _mm256_set1_ps(a_cur[static_cast<size_t>(i) * lda + uk]);
            }

            // Load B values: (NR_AVX2 / VEC_WIDTH_AVX2) vector values.
            // B is packed in K-major order for the current tile (K x cur_N).
            // b_cur_packed_k_start points to the (k_block, n_block + j) origin.
            // uk * packed_B_stride moves to the correct k-row within the packed B tile.
            __m256 b_vecs[NR_AVX2 / VEC_WIDTH_AVX2];
            // Using _mm256_loadu_ps for unaligned loads, which is safe and common practice.
            b_vecs[0] = _mm256_loadu_ps(b_cur_packed_k_start + static_cast<size_t>(uk) * packed_B_stride);
            b_vecs[1] = _mm256_loadu_ps(b_cur_packed_k_start + static_cast<size_t>(uk) * packed_B_stride + VEC_WIDTH_AVX2); // Second vector for NR_AVX2=16

            // Perform FMA (Fused Multiply-Add) operations: C += A * B
            // For each C accumulator (c[i][j_vec]), multiply a broadcasted A value (a_vals[i])
            // with a B vector (b_vecs[j_vec]) and add to the accumulator.
            for (int i = 0; i < MR_AVX2; ++i) {
                for (int j_vec = 0; j_vec < NR_AVX2 / VEC_WIDTH_AVX2; ++j_vec) {
                    c[i][j_vec] = _mm256_fmadd_ps(a_vals[i], b_vecs[j_vec], c[i][j_vec]);
                }
            }
        }
        // Advance pointers for the next K_UNROLL block.
        // b_cur_packed_k_start advances in the K-dimension of the packed B buffer.
        b_cur_packed_k_start += static_cast<size_t>(UNROLL_K_AVX2) * packed_B_stride;
        a_cur += UNROLL_K_AVX2; // A pointer advances in K-dimension.
    }

    // Store accumulated results back to C matrix.
    // C_ptr points to C[m_block + i][n_block + j].
    // i * ldc moves to the next C row. j_vec * VEC_WIDTH_AVX2 moves to the next vector column.
    for (int i = 0; i < MR_AVX2; ++i) {
        for (int j_vec = 0; j_vec < NR_AVX2 / VEC_WIDTH_AVX2; ++j_vec) {
            _mm256_storeu_ps(C_ptr + static_cast<size_t>(i) * ldc + static_cast<size_t>(j_vec) * VEC_WIDTH_AVX2, c[i][j_vec]);
        }
    }
}

// AVX2 Main GEMM function with tiling and packing.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Tile sizes for L2/L3 cache (can be dynamically tuned).
    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN;
    const int BK = DEFAULT_BK;

    // OpenMP parallelization strategy:
    // A #pragma omp parallel region is created first. Each thread gets its own private stack-allocated
    // `packed_B_buffer_vec`. This ensures thread-safety without `firstprivate` clause complexities on `std::vector`.
    // Then, #pragma omp for collapse(2) parallelizes the nested M and N loops, distributing work chunks
    // (M-blocks and N-blocks) to available threads. `schedule(guided)` helps with load balancing.
#ifdef _OPENMP
#pragma omp parallel
    {
        // Allocate buffer for packing B tiles. Each thread gets its own copy, initialized once.
        // Alignment is 32-byte for AVX2.
        std::vector<float, AlignedAllocator<float, 32>> packed_B_buffer_vec(static_cast<size_t>(BK) * BN);
        float* packed_B_buffer = packed_B_buffer_vec.data();

        // Worksharing construct for M and N blocks
#pragma omp for collapse(2) schedule(guided)
        for (int m_block = 0; m_block < M; m_block += BM) {
            for (int n_block = 0; n_block < N; n_block += BN) {
                // Iterate over K dimension blocks.
                for (int k_block = 0; k_block < K; k_block += BK) {
                    // Compute current block dimensions, handling matrix edges.
                    int cur_M = std::min(BM, M - m_block);
                    int cur_N = std::min(BN, N - n_block); // N-dimension of the current B tile
                    int cur_K = std::min(BK, K - k_block);

                    // --- Pack B block ---
                    // This section packs a cur_K x cur_N sub-block of B into `packed_B_buffer`.
                    // Original B is (K x N) with ldb=N. The packed buffer stores B tile in K-major order
                    // for the micro-kernel (better cache locality for B loads).
                    // `packed_B_buffer[kb * cur_N + nb]` corresponds to `B[k_block + kb][n_block + nb]`.
                    for (int kb = 0; kb < cur_K; ++kb) {
                        // Prefetch B if useful, though packing already improves locality greatly.
                        // _mm_prefetch((char*)&B[(k_block + kb + PREFETCH_OFFSET) * ldb + n_block], _MM_HINT_T0);
                        std::memcpy(packed_B_buffer + static_cast<size_t>(kb) * cur_N,
                                    B + static_cast<size_t>(k_block + kb) * ldb + n_block,
                                    static_cast<size_t>(cur_N) * sizeof(float));
                    }

                    // Iterate over sub-blocks of C for the micro-kernel (MR_AVX2 x NR_AVX2).
                    // This is the register-blocking part.
                    for (int i = 0; i < cur_M; i += MR_AVX2) {
                        int mr_actual = std::min(MR_AVX2, cur_M - i); // Actual rows for current micro-panel of C

                        for (int j = 0; j < cur_N; j += NR_AVX2) {
                            int nr_actual = std::min(NR_AVX2, cur_N - j); // Actual columns for current micro-panel of C

                            // Pointers for the current micro-kernel operation.
                            const float* A_panel_ptr = A + static_cast<size_t>(m_block + i) * lda + k_block;
                            float* C_panel_ptr = C + static_cast<size_t>(m_block + i) * ldc + (n_block + j);

                            // Call the micro-kernel if micro-panel dimensions match exactly the micro-kernel's target.
                            // For non-perfectly fitting micro-panels (tails in M or N dimension), fall back to scalar.
                            // This simplifies tail handling at the cost of some performance at edges.
                            // Note: C is assumed to be pre-zeroed by main(). The micro-kernel handles loading 
                            // current C values and accumulating. This is important for correctness across K-blocks.
                            if (mr_actual == MR_AVX2 && nr_actual == NR_AVX2) {
                                sgemm_avx2_ukernel(cur_K, C_panel_ptr, ldc,
                                                   A_panel_ptr, lda,
                                                   packed_B_buffer + j, // Offset into packed B for current N-block
                                                   cur_N); // Pass the current N-dimension of packed_B for correct stride
                            } else {
                                // Scalar fallback for micro-panel tails (M or N dimension is not a multiple of MR/NR).
                                // This computes the partial MR x NR block using scalar ops.
                                for (int r = 0; r < mr_actual; ++r) {
                                    for (int c_idx = 0; c_idx < nr_actual; ++c_idx) {
                                        float sum = C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx]; // Load existing C value for accumulation
                                        for (int k_inner = 0; k_inner < cur_K; ++k_inner) {
                                            // A_panel_ptr[r * lda + k_inner] refers to A[m_block + i + r][k_block + k_inner]
                                            // packed_B_buffer[k_inner * cur_N + (j + c_idx)] refers to B[k_block + k_inner][n_block + j + c_idx]
                                            sum += A_panel_ptr[static_cast<size_t>(r) * lda + k_inner] * packed_B_buffer[static_cast<size_t>(k_inner) * cur_N + (j + c_idx)];
                                        }
                                        C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx] = sum; // Store back the accumulated value
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // End of omp parallel region
#else // No OpenMP: single-threaded execution, similar logic but no pragmas
    std::vector<float, AlignedAllocator<float, 32>> packed_B_buffer_vec(static_cast<size_t>(BK) * BN);
    float* packed_B_buffer = packed_B_buffer_vec.data();
    for (int m_block = 0; m_block < M; m_block += BM) {
        for (int n_block = 0; n_block < N; n_block += BN) {
            for (int k_block = 0; k_block < K; k_block += BK) {
                int cur_M = std::min(BM, M - m_block);
                int cur_N = std::min(BN, N - n_block);
                int cur_K = std::min(BK, K - k_block);

                for (int kb = 0; kb < cur_K; ++kb) {
                    std::memcpy(packed_B_buffer + static_cast<size_t>(kb) * cur_N,
                                B + static_cast<size_t>(k_block + kb) * ldb + n_block,
                                static_cast<size_t>(cur_N) * sizeof(float));
                }

                for (int i = 0; i < cur_M; i += MR_AVX2) {
                    int mr_actual = std::min(MR_AVX2, cur_M - i);
                    for (int j = 0; j < cur_N; j += NR_AVX2) {
                        int nr_actual = std::min(NR_AVX2, cur_N - j);

                        const float* A_panel_ptr = A + static_cast<size_t>(m_block + i) * lda + k_block;
                        float* C_panel_ptr = C + static_cast<size_t>(m_block + i) * ldc + (n_block + j);

                        if (mr_actual == MR_AVX2 && nr_actual == NR_AVX2) {
                            sgemm_avx2_ukernel(cur_K, C_panel_ptr, ldc,
                                               A_panel_ptr, lda,
                                               packed_B_buffer + j,
                                               cur_N);
                        } else {
                            for (int r = 0; r < mr_actual; ++r) {
                                for (int c_idx = 0; c_idx < nr_actual; ++c_idx) {
                                    float sum = C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx];
                                    for (int k_inner = 0; k_inner < cur_K; ++k_inner) {
                                        sum += A_panel_ptr[static_cast<size_t>(r) * lda + k_inner] * packed_B_buffer[static_cast<size_t>(k_inner) * cur_N + (j + c_idx)];
                                    }
                                    C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif // _OPENMP
}
#endif // __AVX2__ && __FMA__

// --- AVX-512 GEMM Implementation ---
// This kernel is provided as requested, but a Ryzen 7 6800HS CPU does NOT support AVX-512.
// It will only be compiled and dispatched if the compiler defines __AVX512F__ and the CPU supports it at runtime.
#if defined(__AVX512F__) && defined(__FMA__)
void sgemm_avx512_ukernel(int K_iter, float* C_ptr, int ldc,
                          const float* A_ptr, int lda,
                          const float* packed_B_ptr, int packed_B_stride) {
    
    __m512 c[MR_AVX512][NR_AVX512 / VEC_WIDTH_AVX512];
    for (int i = 0; i < MR_AVX512; ++i) {
        for (int j_vec = 0; j_vec < NR_AVX512 / VEC_WIDTH_AVX512; ++j_vec) {
            // Load existing C values from the current C micro-panel into AVX-512 accumulators.
            c[i][j_vec] = _mm512_loadu_ps(C_ptr + static_cast<size_t>(i) * ldc + static_cast<size_t>(j_vec) * VEC_WIDTH_AVX512);
        }
    }

    const float* a_cur = A_ptr;
    const float* b_cur_packed_k_start = packed_B_ptr;

    for (int k_idx = 0; k_idx < K_iter; k_idx += UNROLL_K_AVX512) {
        int k_eff = std::min(UNROLL_K_AVX512, K_iter - k_idx);

        for (int uk = 0; uk < k_eff; ++uk) {
            __m512 a_vals[MR_AVX512];
            for (int i = 0; i < MR_AVX512; ++i) {
                a_vals[i] = _mm512_set1_ps(a_cur[static_cast<size_t>(i) * lda + uk]);
            }

            __m512 b_vecs[NR_AVX512 / VEC_WIDTH_AVX512];
            b_vecs[0] = _mm512_loadu_ps(b_cur_packed_k_start + static_cast<size_t>(uk) * packed_B_stride);

            for (int i = 0; i < MR_AVX512; ++i) {
                for (int j_vec = 0; j_vec < NR_AVX512 / VEC_WIDTH_AVX512; ++j_vec) {
                    c[i][j_vec] = _mm512_fmadd_ps(a_vals[i], b_vecs[j_vec], c[i][j_vec]);
                }
            }
        }
        b_cur_packed_k_start += static_cast<size_t>(UNROLL_K_AVX512) * packed_B_stride;
        a_cur += UNROLL_K_AVX512;
    }

    for (int i = 0; i < MR_AVX512; ++i) {
        for (int j_vec = 0; j_vec < NR_AVX512 / VEC_WIDTH_AVX512; ++j_vec) {
            _mm512_storeu_ps(C_ptr + static_cast<size_t>(i) * ldc + static_cast<size_t>(j_vec) * VEC_WIDTH_AVX512, c[i][j_vec]);
        }
    }
}

void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {

    const int BM = DEFAULT_BM;
    const int BN = DEFAULT_BN;
    const int BK = DEFAULT_BK;

#ifdef _OPENMP
#pragma omp parallel
    {
        // Alignment is 64-byte for AVX-512.
        std::vector<float, AlignedAllocator<float, 64>> packed_B_buffer_vec(static_cast<size_t>(BK) * BN);
        float* packed_B_buffer = packed_B_buffer_vec.data();

#pragma omp for collapse(2) schedule(guided)
        for (int m_block = 0; m_block < M; m_block += BM) {
            for (int n_block = 0; n_block < N; n_block += BN) {
                for (int k_block = 0; k_block < K; k_block += BK) {
                    int cur_M = std::min(BM, M - m_block);
                    int cur_N = std::min(BN, N - n_block);
                    int cur_K = std::min(BK, K - k_block);

                    for (int kb = 0; kb < cur_K; ++kb) {
                        std::memcpy(packed_B_buffer + static_cast<size_t>(kb) * cur_N,
                                    B + static_cast<size_t>(k_block + kb) * ldb + n_block,
                                    static_cast<size_t>(cur_N) * sizeof(float));
                    }

                    for (int i = 0; i < cur_M; i += MR_AVX512) {
                        int mr_actual = std::min(MR_AVX512, cur_M - i);

                        for (int j = 0; j < cur_N; j += NR_AVX512) {
                            int nr_actual = std::min(NR_AVX512, cur_N - j);

                            const float* A_panel_ptr = A + static_cast<size_t>(m_block + i) * lda + k_block;
                            float* C_panel_ptr = C + static_cast<size_t>(m_block + i) * ldc + (n_block + j);

                            if (mr_actual == MR_AVX512 && nr_actual == NR_AVX512) {
                                sgemm_avx512_ukernel(cur_K, C_panel_ptr, ldc,
                                                     A_panel_ptr, lda,
                                                     packed_B_buffer + j,
                                                     cur_N);
                            } else {
                                for (int r = 0; r < mr_actual; ++r) {
                                    for (int c_idx = 0; c_idx < nr_actual; ++c_idx) {
                                        float sum = C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx];
                                        for (int k_inner = 0; k_inner < cur_K; ++k_inner) {
                                            sum += A_panel_ptr[static_cast<size_t>(r) * lda + k_inner] * packed_B_buffer[static_cast<size_t>(k_inner) * cur_N + (j + c_idx)];
                                        }
                                        C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx] = sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#else // No OpenMP: single-threaded execution
    std::vector<float, AlignedAllocator<float, 64>> packed_B_buffer_vec(static_cast<size_t>(BK) * BN);
    float* packed_B_buffer = packed_B_buffer_vec.data();
    for (int m_block = 0; m_block < M; m_block += BM) {
        for (int n_block = 0; n_block < N; n_block += BN) {
            for (int k_block = 0; k_block < K; k_block += BK) {
                int cur_M = std::min(BM, M - m_block);
                int cur_N = std::min(BN, N - n_block);
                int cur_K = std::min(BK, K - k_block);

                for (int kb = 0; kb < cur_K; ++kb) {
                    std::memcpy(packed_B_buffer + static_cast<size_t>(kb) * cur_N,
                                B + static_cast<size_t>(k_block + kb) * ldb + n_block,
                                static_cast<size_t>(cur_N) * sizeof(float));
                }

                for (int i = 0; i < cur_M; i += MR_AVX512) {
                    int mr_actual = std::min(MR_AVX512, cur_M - i);
                    for (int j = 0; j < cur_N; j += NR_AVX512) {
                        int nr_actual = std::min(NR_AVX512, cur_N - j);

                        const float* A_panel_ptr = A + static_cast<size_t>(m_block + i) * lda + k_block;
                        float* C_panel_ptr = C + static_cast<size_t>(m_block + i) * ldc + (n_block + j);

                        if (mr_actual == MR_AVX512 && nr_actual == NR_AVX512) {
                            sgemm_avx512_ukernel(cur_K, C_panel_ptr, ldc,
                                                 A_panel_ptr, lda,
                                                 packed_B_buffer + j,
                                                 cur_N);
                        } else {
                            for (int r = 0; r < mr_actual; ++r) {
                                for (int c_idx = 0; c_idx < nr_actual; ++c_idx) {
                                    float sum = C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx];
                                    for (int k_inner = 0; k_inner < cur_K; ++k_inner) {
                                        sum += A_panel_ptr[static_cast<size_t>(r) * lda + k_inner] * packed_B_buffer[static_cast<size_t>(k_inner) * cur_N + (j + c_idx)];
                                    }
                                    C_panel_ptr[static_cast<size_t>(r) * ldc + c_idx] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif // _OPENMP
}
#endif // __AVX512F__ && __FMA__

// --- Top-level GEMM Dispatcher ---
// This function determines which optimized kernel to call based on CPU features at runtime.
// The preprocessor guards ensure that only supported kernels are compiled.
// For example, if compiled with -mavx2, the AVX-512 block is skipped at compile time.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    // Runtime dispatch using __builtin_cpu_supports (GCC/Clang specific).
    // This assumes the binary was compiled with support for the respective ISA.
    // E.g., for AVX-512, must be compiled with -mavx512f.
#if defined(__AVX512F__) && defined(__FMA__)
    if (__builtin_cpu_supports("avx512f")) {
        std::cout << "Using AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif
    // Fallback to AVX2 if available and compiled with __AVX2__ support.
#if defined(__AVX2__) && defined(__FMA__)
    if (__builtin_cpu_supports("avx2")) {
        std::cout << "Using AVX2 kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif
    // Fallback to scalar if no SIMD instruction set is available or supported.
    std::cout << "Using Scalar kernel." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}

// --- Main function for demonstration and testing ---
int main(int argc, char* argv[]) {
    int M = 512, N = 512, K = 512;
    unsigned int seed = 12345;
    int num_threads = 0; // 0 means OpenMP default
    bool dump_matrices = false;
    bool check_correctness = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "--check-correctness") {
            check_correctness = true;
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoul(argv[++i]);
        } else if (i + 2 < argc) { // M N K as positional arguments
            M = std::stoi(argv[i]);
            N = std::stoi(argv[i+1]);
            K = std::stoi(argv[i+2]);
            i += 2;
        } else {
            std::cerr << "Usage: " << argv[0] << " [M N K] [--seed SEED] [--threads NUM_THREADS] [--dump-matrices] [--check-correctness]\n";
            return 1;
        }
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    } else {
        // If not explicitly set by --threads, use system default (or OMP_NUM_THREADS env var).
        // omp_get_max_threads() gets the maximum number of threads available in the current OpenMP environment.
        num_threads = omp_get_max_threads();
    }
    std::cout << "Using " << num_threads << " OpenMP threads." << std::endl;
#else
    if (num_threads > 0) {
        std::cerr << "Warning: OpenMP not enabled, --threads argument ignored. Running on single thread." << std::endl;
    }
    std::cout << "OpenMP not available. Running on a single thread." << std::endl;
#endif

    // Matrix dimensions and leading dimensions (row-major storage convention)
    int lda = K; // A is M x K, so leading dimension (stride to next row) is K
    int ldb = N; // B is K x N, so leading dimension (stride to next row) is N
    int ldc = N; // C is M x N, so leading dimension (stride to next row) is N

    // Allocate matrices using aligned allocator (64-byte alignment to be compatible with AVX-512 if available,
    // otherwise 32-byte (for AVX2) or 16-byte (for SSE) would also be sufficient, but 64 is safe max).
    std::vector<float, AlignedAllocator<float, 64>> A_vec(static_cast<size_t>(M) * K);
    std::vector<float, AlignedAllocator<float, 64>> B_vec(static_cast<size_t>(K) * N);
    std::vector<float, AlignedAllocator<float, 64>> C_vec(static_cast<size_t>(M) * N);
    
    float* A = A_vec.data();
    float* B = B_vec.data();
    float* C = C_vec.data();

    std::vector<float, AlignedAllocator<float, 64>> C_ref_vec;
    float* C_ref = nullptr;
    if (check_correctness) {
        C_ref_vec.resize(static_cast<size_t>(M) * N);
        C_ref = C_ref_vec.data();
    }

    // Initialize matrices A and B with random values, C with zeros
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> distrib(-1.0f, 1.0f);

    for (size_t i = 0; i < static_cast<size_t>(M) * K; ++i) A[i] = distrib(gen);
    for (size_t i = 0; i < static_cast<size_t>(K) * N; ++i) B[i] = distrib(gen);
    std::fill(C, C + static_cast<size_t>(M) * N, 0.0f); // Initialize C to zero before computation

    if (dump_matrices) {
        // Create 'workspace' directory if it doesn't exist. Permissions: rwx for owner, rwx for group, rwx for others.
        int status = mkdir("workspace", S_IRWXU | S_IRWXG | S_IRWXO);
        if (status != 0 && errno != EEXIST) { // If creation failed and error is not "already exists"
            std::cerr << "Warning: Could not create directory 'workspace' (error: " << strerror(errno) << "). Dumping matrices skipped." << std::endl;
            dump_matrices = false; // Disable dumping if directory creation failed
        }
    }

    if (dump_matrices) {
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt\n";
    }

    // Perform GEMM and time it
    std::cout << "Running GEMM for M=" << M << ", N=" << N << ", K=" << K << "...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc); // Call the top-level dispatcher
    auto end_time = std::chrono::high_resolution_clock::now();

    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Matrix C dumped to workspace/C.txt\n";
    }

    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    // GFLOPs calculation: 2 * M * N * K (multiply and add for each element of C).
    double total_operations = 2.0 * static_cast<double>(M) * N * K;
    double gflops = (elapsed_ms.count() > 0) ? (total_operations / (elapsed_ms.count() / 1000.0) / 1e9) : 0.0;

    std::cout << "GEMM finished in " << elapsed_ms.count() << " ms.\n";
    std::cout << "Performance: " << gflops << " GFLOP/s\n";

    // Correctness check against scalar reference
    if (check_correctness) {
        std::cout << "Running scalar reference for correctness check...\n";
        std::fill(C_ref, C_ref + static_cast<size_t>(M) * N, 0.0f); // Ensure C_ref is zeroed
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

        float max_diff = 0.0f;
        float total_diff_sq = 0.0f;
        for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) {
            float diff = std::abs(C[i] - C_ref[i]);
            max_diff = std::max(max_diff, diff);
            total_diff_sq += diff * diff;
        }
        float rmse = std::sqrt(total_diff_sq / (static_cast<double>(M) * N));

        std::cout << "Correctness Check:\n";
        std::cout << "  Max absolute difference: " << max_diff << "\n";
        std::cout << "  RMSE: " << rmse << "\n";

        // Adjusted tolerance: The sum of K products accumulates floating-point error.
        // A common heuristic for expected error is machine_epsilon * K * max_abs_value_in_C_expected.
        float max_val_A = 0.0f, max_val_B = 0.0f;
        for(size_t i=0; i < static_cast<size_t>(M)*K; ++i) max_val_A = std::max(max_val_A, std::abs(A[i]));
        for(size_t i=0; i < static_cast<size_t>(K)*N; ++i) max_val_B = std::max(max_val_B, std::abs(B[i]));
        
        float expected_max_C = (K == 0) ? 0.0f : (max_val_A * max_val_B * K);
        // Using a tolerance that accounts for accumulation, multiplied by a safety factor (e.g., 100).
        // `std::numeric_limits<float>::epsilon()` is typically 1.19e-7.
        float tolerance = std::numeric_limits<float>::epsilon() * static_cast<float>(K) * expected_max_C * 100.0f;
        
        // Ensure a reasonable minimum tolerance even for very small or zero-valued matrices.
        if (tolerance < std::numeric_limits<float>::epsilon() * 10.0f) tolerance = std::numeric_limits<float>::epsilon() * 10.0f;
        // If expected_max_C is truly zero, the tolerance should be small absolute value.
        if (expected_max_C == 0.0f) tolerance = std::numeric_limits<float>::epsilon(); 

        if (max_diff < tolerance) {
            std::cout << "Result is correct within tolerance (tol=" << tolerance << ").\n";
        } else {
            std::cerr << "ERROR: Result differs from scalar reference! Max diff: " << max_diff << ", Tolerance: " << tolerance << "\n";
        }
    }

    return 0;
}