// Compile with GCC/Clang:
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// For minimal scalar-only version:
// g++ -O3 -march=native gemm.cpp -o gemm_scalar_only

#include <iostream>
#include <vector>
#include <cstring> // For memcpy, memset
#include <chrono>  // For timing
#include <random>  // For random numbers
#include <cassert> // For assert
#include <fstream> // For file I/O
#include <string>  // For string manipulation
#include <iomanip> // For std::setprecision, std::fixed
#include <algorithm> // For std::min, std::max
#include <cmath> // For std::sqrt, std::abs

#if defined(__AVX2__)
#include <immintrin.h> // For AVX2 intrinsics
#endif

#ifdef _OPENMP
#include <omp.h> // For OpenMP parallelism
#else
// Define dummy OpenMP functions if not available, allowing compilation without OpenMP
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
void omp_set_num_threads(int) {}
#endif

// --- Autotuning Parameters ---
// VEC_WIDTH for float (AVX2: 8 floats in __m256)
constexpr int VEC_WIDTH = 8;

// Micro-kernel dimensions (MR x NR)
// This kernel processes MR rows of C and NR columns of C simultaneously.
// NR must be a multiple of VEC_WIDTH.
//
// Optimized for AVX2 on AMD Ryzen 7 6800HS (Zen 3 architecture), which typically has 16 YMM registers
// and two 256-bit FMA units per core.
//
// MR_AVX = 4: Processes 4 rows of A and C for each micro-kernel.
// NR_AVX = 2 * VEC_WIDTH = 16: Processes 16 columns of B and C for each micro-kernel (2 __m256 vectors).
// This micro-kernel uses:
// - C_regs[MR_AVX][NR_AVX / VEC_WIDTH] = C_regs[4][2] = 8 __m256 accumulators for C.
// - A_vals[MR_AVX] = A_vals[4] = 4 __m256 for broadcasted A elements.
// - B_vals[NR_AVX / VEC_WIDTH] = B_vals[2] = 2 __m256 for packed B vectors.
// Total YMM registers: 8 (C) + 4 (A) + 2 (B) = 14. This configuration fits perfectly within the 16 available
// AVX2 YMM registers, preventing spills and maximizing Instruction-Level Parallelism (ILP) by
// allowing both FMA units to be utilized effectively.
constexpr int MR_AVX = 4;
constexpr int NR_AVX = 2 * VEC_WIDTH; 

// Cache Blocking/Tiling parameters
// BM: Block size for M dimension
// BN: Block size for N dimension
// BK: Block size for K dimension
// These parameters are crucial for cache reuse.
// BM should be a multiple of MR_AVX. BN should be a multiple of NR_AVX.
// The goal is for a block of C (BMxBN), a block of A (BMxBK), and a block of B (BKxBN)
// to fit efficiently into L1/L2 cache during computation within a tile.
// For Ryzen 6800HS: L1d (32KB/core), L2 (512KB/core).
// The previous iteration was Compute Bound. The current block sizes aim to keep the working set
// in L2 cache to maximize data reuse and keep the FMA units fed.
// - A block of A (BMxBK * sizeof(float)) = 192 * 192 * 4 bytes = 147456 bytes (~144KB). Aims for L2 cache.
// - A block of B (BKxBN * sizeof(float)) = 192 * 192 * 4 bytes = 147456 bytes (~144KB). Aims for L2 cache.
// - A block of C (BMxBN * sizeof(float)) = 192 * 192 * 4 bytes = 147456 bytes (~144KB). Aims for L2 cache.
// The total working set for a K-block pass (A block + packed B block + C block) is about ~432KB,
// which fits comfortably within the 512KB L2 cache per core on Ryzen.
constexpr int BM = 192;   // A block height. Must be a multiple of MR_AVX (192 = 48 * 4).
constexpr int BN = 192;  // C/B block width. Must be a multiple of NR_AVX (192 = 12 * 16).
constexpr int BK = 192;  // A/B block depth.

// Inner K-loop unroll factor for the micro-kernel.
// Unrolls the K loop by this factor to reduce loop overhead and expose Instruction-Level Parallelism (ILP).
// Increased to 8 from 4 to further expose ILP for compute-bound scenarios on Zen 3.
constexpr int UNROLL_K = 8;

// Prefetching distances
// _MM_HINT_T0 (L1), _MM_HINT_T1 (L2), _MM_HINT_T2 (L3), _MM_HINT_NTA (non-temporal, bypassing caches).
// T0 is generally a good hint for data needed soon in the innermost loops.
// Prefetch distances are aligned with UNROLL_K to ensure data is brought into cache in time.
constexpr int PREFETCH_DISTANCE_A = UNROLL_K; // Prefetch `A` elements for next K iteration
constexpr int PREFETCH_DISTANCE_B = UNROLL_K; // Prefetch `B` elements for next K iteration (ahead in K for current N-panel)

// --- Memory Allocation Helper ---
// Custom allocator for aligned memory. Essential for SIMD performance.
// AVX2 typically requires 32-byte alignment for optimal performance, especially for load/store intrinsics.
template <typename T>
T* aligned_alloc(size_t num_elements, size_t alignment = 32) {
    void* ptr = nullptr;
    #if defined(_MSC_VER)
        ptr = _aligned_malloc(num_elements * sizeof(T), alignment);
        if (ptr == nullptr) {
            throw std::bad_alloc();
        }
    #else
        int ret = posix_memalign(&ptr, alignment, num_elements * sizeof(T));
        if (ret != 0) {
            throw std::bad_alloc();
        }
    #endif
    return static_cast<T*>(ptr);
}

void aligned_free(void* ptr) {
    #if defined(_MSC_VER)
        _aligned_free(ptr);
    #else
        free(ptr);
    #endif
}

// --- Matrix I/O Helper ---
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    ofs << std::fixed << std::setprecision(6); // For consistent output precision

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// --- GEMM Implementations ---

// Matrix storage convention: Row-major
// A: M x K, stored as A[row*lda + col]
// B: K x N, stored as B[row*ldb + col]
// C: M x N, stored as C[row*ldc + col]

/**
 * @brief Reference scalar general matrix multiplication (GEMM). C = A * B + C
 * Implemented with simple triple nested loops (I-J-K order), which is generally
 * good for C access locality (row-major) and A access locality.
 * B access has a stride 'ldb' when iterating 'j', which can be poor.
 * This version is single-threaded and used for correctness verification.
 * @param A Pointer to the M x K matrix A (row-major).
 * @param B Pointer to the K x N matrix B (row-major).
 * @param C Pointer to the M x N matrix C (row-major).
 * @param M Number of rows in A and C.
 * @param N Number of columns in B and C.
 * @param K Number of columns in A and rows in B.
 * @param lda Leading dimension (stride) of A.
 * @param ldb Leading dimension (stride) of B.
 * @param ldc Leading dimension (stride) of C.
 */
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float c_val = C[i * ldc + j]; // Load C_ij once
            for (int k = 0; k < K; ++k) {
                c_val += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = c_val; // Store C_ij once
        }
    }
}

#if defined(__AVX2__)

// Helper to pack a block of B for better cache locality in the micro-kernel.
// B is K x N. We pack a BK x BN sub-block into `packed_B`.
// The packing rearranges elements from row-major (K rows, N columns) to 
// a layout where VEC_WIDTH columns are contiguous for each K, then the next VEC_WIDTH columns, etc.
// This allows `_mm256_load_ps` to be used for B columns efficiently within the micro-kernel.
//
// Packed_B layout (N-major within VEC_WIDTH blocks, K-major overall):
// for k from 0 to BK_size-1:
//   for n_block from 0 to BN_aligned/VEC_WIDTH-1:
//     for v from 0 to VEC_WIDTH-1:
//       packed_B[k * BN_ALIGNED_PACKED + n_block * VEC_WIDTH + v]
//
// @param packed_B Destination buffer for the packed block.
// @param B Source matrix B.
// @param K_start Starting row index (in K dimension) of the block in B.
// @param N_start Starting column index (in N dimension) of the block in B.
// @param BK_size Actual height (K-dimension) of the block.
// @param BN_size Actual width (N-dimension) of the block.
// @param ldb Leading dimension of source matrix B.
// @param BN_ALIGNED_PACKED The aligned width for the packed buffer (guaranteed multiple of VEC_WIDTH).
void pack_B_block(float* packed_B, const float* B, int K_start, int N_start, int BK_size, int BN_size, int ldb, int BN_ALIGNED_PACKED) {
    for (int k = 0; k < BK_size; ++k) {
        // Pointer to the start of the current row in the packed buffer
        float* packed_row_ptr = &packed_B[k * BN_ALIGNED_PACKED];
        // Pointer to the start of the current row in the original B matrix
        const float* B_row_ptr = &B[(K_start + k) * ldb + N_start];

        for (int n = 0; n < BN_size; ++n) {
            // Map B[K_start+k][N_start+n] to packed_B
            // The logic: (n / VEC_WIDTH) * VEC_WIDTH navigates to the start of the VEC_WIDTH-aligned block.
            // (n % VEC_WIDTH) selects the element within that VEC_WIDTH block.
            packed_row_ptr[(n / VEC_WIDTH) * VEC_WIDTH + (n % VEC_WIDTH)] = B_row_ptr[n];
        }
        // Zero-fill padding for the rest of the BN_ALIGNED_PACKED row. This ensures _mm256_load_ps
        // will always fetch valid data (either from B or zero) even if current_BN is not a multiple of VEC_WIDTH.
        for (int n = BN_size; n < BN_ALIGNED_PACKED; ++n) {
            packed_row_ptr[n] = 0.0f;
        }
    }
}


/**
 * @brief AVX2 optimized general matrix multiplication (GEMM). C = A * B + C
 * Uses OpenMP for multi-threading, cache blocking, and an AVX2+FMA micro-kernel.
 * Favors row-major storage. Implements packing for B.
 *
 * The GEMM is structured with M-N-K tiling:
 * 1. Outer loops (`m_idx`, `n_idx`) iterate over `BM x BN` blocks of C, parallelized by OpenMP.
 * 2. Middle loop (`k_idx`) iterates over `BK` depth blocks.
 *    - Inside this loop, a `BK x BN` block of B is packed into a thread-local buffer to ensure contiguous access.
 * 3. Inner loops (`i`, `jj`) iterate over `MR_AVX x NR_AVX` micro-panels of C.
 *    - The micro-kernel accumulates results into AVX2 registers using FMA instructions.
 *    - Prefetching is used for A and the packed B data.
 *
 * Edge cases (tails not divisible by tile sizes or vector widths) are handled:
 * - `current_BM`, `current_BN`, `current_BK`: Adjust block sizes at matrix edges for tiling.
 * - `current_MR_inner`: Adjust micro-kernel rows at block edges.
 * - `VEC_WIDTH` (N-dimension) tail: Handled by `_mm256_maskload_ps` for partial vector loads of C
 *   and `_mm256_maskstore_ps` for partial vector stores of C.
 *   Packed B is zero-padded by `pack_B_block` to simplify loads in the micro-kernel.
 *
 * @param A Pointer to the M x K matrix A (row-major).
 * @param B Pointer to the K x N matrix B (row-major).
 * @param C Pointer to the M x N matrix C (row-major).
 * @param M Number of rows in A and C.
 * @param N Number of columns in B and C.
 * @param K Number of columns in A and rows in B.
 * @param lda Leading dimension (stride) of A.
 * @param ldb Leading dimension (stride) of B.
 * @param ldc Leading dimension (stride) of C.
 */
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Calculate aligned BN for packed_B. This ensures that `_mm256_load_ps` on packed_B
    // is always aligned and reads full vectors, up to the padded width.
    const int BN_ALIGNED_PACKED = ((BN + VEC_WIDTH - 1) / VEC_WIDTH) * VEC_WIDTH;
    
    // Declare thread-local packed B buffer. This avoids false sharing and contention between threads.
    // The maximum possible size for this buffer is BK * BN_ALIGNED_PACKED.
    // Allocated once per thread for its entire lifetime.
    float* packed_B_buffer = nullptr;

    #pragma omp parallel private(packed_B_buffer)
    {
        // Allocate thread-local aligned packed B buffer.
        // The size is BK * BN_ALIGNED_PACKED * sizeof(float). E.g., 192 * 192 * 4 = 144KB, which fits in L2.
        packed_B_buffer = aligned_alloc<float>(BK * BN_ALIGNED_PACKED);

        // OpenMP parallel region for M and N block loops (outermost loops)
        // `collapse(2)` ensures that both `m_idx` and `n_idx` loops are part of the work-sharing construct,
        // allowing for finer-grained task distribution across threads.
        // `schedule(static)` is used for load balancing, assuming blocks are of relatively uniform size.
        // `nowait` can potentially improve performance by allowing threads to proceed without barrier if possible.
        #pragma omp for collapse(2) schedule(static) nowait
        for (int m_idx = 0; m_idx < M; m_idx += BM) {
            for (int n_idx = 0; n_idx < N; n_idx += BN) {
                // Determine actual block sizes for current M and N tiles (handles matrix edges)
                const int current_BM = std::min(BM, M - m_idx);
                const int current_BN = std::min(BN, N - n_idx);

                for (int k_idx = 0; k_idx < K; k_idx += BK) {
                    const int current_BK = std::min(BK, K - k_idx);

                    // --- Stage 1: Pack B block for cache efficiency ---
                    // Packing the current_BK x current_BN block of B into contiguous memory.
                    // This dramatically improves memory access patterns for B within the micro-kernel
                    // by making subsequent vector loads aligned and contiguous.
                    pack_B_block(packed_B_buffer, B, k_idx, n_idx, current_BK, current_BN, ldb, BN_ALIGNED_PACKED);

                    // --- Stage 2: Compute C += A * packed_B ---
                    // Iterate over rows of A / C within the current M block, in steps of MR_AVX (micro-kernel height)
                    for (int i = 0; i < current_BM; i += MR_AVX) {
                        const int current_MR_inner = std::min(MR_AVX, current_BM - i);

                        // Iterate over columns of C within the current N block, in steps of NR_AVX (micro-kernel width)
                        for (int jj = 0; jj < current_BN; jj += NR_AVX) {
                            // current_NR_inner is used for accumulator initialization and storage,
                            // ensuring operations stay within the bounds of the actual N-block.
                            const int current_NR_inner = std::min(NR_AVX, current_BN - jj); 

                            // Accumulator registers for C micro-panel (MR_AVX rows x NR_AVX columns)
                            // NR_AVX / VEC_WIDTH is the number of __m256 vectors horizontally (e.g., 16/8 = 2)
                            __m256 C_regs[MR_AVX][NR_AVX / VEC_WIDTH];

                            // Initialize C accumulators from C matrix or to zero.
                            // Handles M-dimension and N-dimension tails for C loads.
                            for (int r = 0; r < MR_AVX; ++r) {
                                for (int c_vec = 0; c_vec < NR_AVX / VEC_WIDTH; ++c_vec) {
                                    __m256 current_C_vec_val = _mm256_setzero_ps(); // Default to zero

                                    // Only attempt to load from C if this row is within the actual micro-panel rows
                                    if (r < current_MR_inner) {
                                        int current_n_start_for_load = n_idx + jj + c_vec * VEC_WIDTH;
                                        
                                        // Only attempt to load if this vector block conceptually starts within the actual micro-panel columns
                                        if (current_n_start_for_load < n_idx + current_BN) {
                                            // Check if the entire vector fits within C's N dimension bounds for this block.
                                            if (current_n_start_for_load + VEC_WIDTH <= n_idx + current_BN) {
                                                current_C_vec_val = _mm256_loadu_ps(&C[(m_idx + i + r) * ldc + current_n_start_for_load]);
                                            } else {
                                                // Handle N-tail (partial vector load) using a mask.
                                                int remaining_cols_in_vector = (n_idx + current_BN) - current_n_start_for_load;
                                                // It's guaranteed remaining_cols_in_vector > 0 if current_n_start_for_load < n_idx + current_BN
                                                // and not (current_n_start_for_load + VEC_WIDTH <= n_idx + current_BN).
                                                alignas(32) int mask_array[VEC_WIDTH];
                                                for(int m_i = 0; m_i < VEC_WIDTH; ++m_i) {
                                                    mask_array[m_i] = (m_i < remaining_cols_in_vector) ? -1 : 0;
                                                }
                                                __m256i mask = _mm256_load_si256((__m256i*)mask_array);
                                                current_C_vec_val = _mm256_maskload_ps(&C[(m_idx + i + r) * ldc + current_n_start_for_load], mask);
                                            }
                                        }
                                    }
                                    C_regs[r][c_vec] = current_C_vec_val;
                                }
                            }

                            // Base pointers for the current micro-panel of A and packed_B
                            // A_panel_base_ptr_row_0 points to A[m_idx + i][k_idx]
                            const float* A_panel_base_ptr_row_0 = A + (m_idx + i) * lda + k_idx;
                            // B_panel_base_ptr_jj points to packed_B_buffer[0][jj] for the current k-block within the packed buffer
                            const float* B_panel_base_ptr_jj = packed_B_buffer + jj;

                            // Inner K loop for the micro-kernel (MR_AVX x current_NR_inner x current_BK)
                            // This loop is unrolled by UNROLL_K to reduce loop overhead and expose ILP.
                            for (int k_unroll = 0; k_unroll < current_BK; k_unroll += UNROLL_K) {
                                // Loop body is unrolled `UNROLL_K` times to schedule more instructions concurrently.
                                // For K-tails, the `if (k_offset >= current_BK) break;` handles early exit.
                                for (int uk = 0; uk < UNROLL_K; ++uk) {
                                    int k_offset = k_unroll + uk; // Represents the K-offset within current_BK block
                                    if (k_offset >= current_BK) break; // Handle K-tail if current_BK is not divisible by UNROLL_K

                                    // Prefetch A and packed_B for future iterations
                                    // Prefetch `A` elements for next `k` iterations.
                                    // Prefetch a small block of A (current_MR_inner rows) at the predicted future k-column.
                                    if (k_offset + PREFETCH_DISTANCE_A < current_BK) {
                                        for(int r_pref = 0; r_pref < current_MR_inner; ++r_pref) { // Prefetch only for active rows
                                            _mm_prefetch((char*)&A_panel_base_ptr_row_0[r_pref * lda + (k_offset + PREFETCH_DISTANCE_A)], _MM_HINT_T0);
                                        }
                                    }
                                    // Prefetch packed `B` elements for next `k` iterations.
                                    // Prefetch the relevant segment of packed_B for the current micro-panel (starting from jj).
                                    if (k_offset + PREFETCH_DISTANCE_B < current_BK) {
                                        _mm_prefetch((char*)&B_panel_base_ptr_jj[(k_offset + PREFETCH_DISTANCE_B) * BN_ALIGNED_PACKED], _MM_HINT_T0);
                                    }
                                    
                                    // Load A values for current K, MR_AVX rows.
                                    // Each A_vals[r] will contain `A[row][k]` broadcasted to all 8 floats.
                                    __m256 A_vals[MR_AVX];
                                    for(int r = 0; r < MR_AVX; ++r) {
                                        if (r < current_MR_inner) {
                                            A_vals[r] = _mm256_set1_ps(A_panel_base_ptr_row_0[r * lda + k_offset]);
                                        } else {
                                            A_vals[r] = _mm256_setzero_ps(); // Use zero if past actual MR_inner, effectively no-op in FMA
                                        }
                                    }

                                    // Load B values for current K, NR_AVX columns from packed_B_buffer.
                                    // These are already contiguous and aligned due to packing and zero-padding in pack_B_block.
                                    __m256 B_vals[NR_AVX / VEC_WIDTH];
                                    for (int c_vec = 0; c_vec < NR_AVX / VEC_WIDTH; ++c_vec) {
                                        // The packed_B_buffer stores data aligned to BN_ALIGNED_PACKED, which is a multiple of VEC_WIDTH.
                                        // pack_B_block already zero-fills any padding needed within the current_BN and BN_ALIGNED_PACKED.
                                        // Thus, _mm256_load_ps is safe here and will load correct values or zeros.
                                        B_vals[c_vec] = _mm256_load_ps(B_panel_base_ptr_jj + k_offset * BN_ALIGNED_PACKED + c_vec * VEC_WIDTH);
                                    }

                                    // Perform MR_AVX * (NR_AVX / VEC_WIDTH) FMA operations
                                    // C_regs[r][c_vec] += A_vals[r] * B_vals[c_vec]
                                    for (int r = 0; r < MR_AVX; ++r) {
                                        for (int c_vec = 0; c_vec < NR_AVX / VEC_WIDTH; ++c_vec) {
                                            C_regs[r][c_vec] = _mm256_fmadd_ps(A_vals[r], B_vals[c_vec], C_regs[r][c_vec]);
                                        }
                                    }
                                } // end uk (UNROLL_K) loop
                            } // end k_unroll loop (K-loop for micro-kernel)

                            // Store accumulated C values back to C matrix
                            for (int r = 0; r < current_MR_inner; ++r) { // Only iterate for actual rows to store
                                for (int c_vec = 0; c_vec < NR_AVX / VEC_WIDTH; ++c_vec) {
                                    int current_n_start_for_store = n_idx + jj + c_vec * VEC_WIDTH;
                                    
                                    // Only store if this vector block is within the current_NR_inner.
                                    // This also inherently checks against current_BN as current_NR_inner is bounded by current_BN.
                                    if ((c_vec * VEC_WIDTH) < current_NR_inner) {
                                        if (current_n_start_for_store + VEC_WIDTH <= n_idx + current_BN) {
                                            // If it fits entirely, use an unaligned store. C matrix elements are not guaranteed to be 32-byte aligned.
                                            _mm256_storeu_ps(&C[(m_idx + i + r) * ldc + current_n_start_for_store], C_regs[r][c_vec]);
                                        } else {
                                            // Handle N-tail for C_regs when storing using a mask.
                                            // Calculate remaining elements in this vector slice that are part of the matrix.
                                            int remaining_cols_in_vector = (n_idx + current_BN) - current_n_start_for_store;
                                            if (remaining_cols_in_vector > 0) {
                                                // Create a mask to store only the valid elements.
                                                // -1 (all bits set) for active lanes, 0 for inactive.
                                                alignas(32) int mask_array[VEC_WIDTH];
                                                for(int m_i = 0; m_i < VEC_WIDTH; ++m_i) {
                                                    mask_array[m_i] = (m_i < remaining_cols_in_vector) ? -1 : 0;
                                                }
                                                __m256i mask = _mm256_load_si256((__m256i*)mask_array);
                                                // Masked store writes only to lanes where the corresponding mask bit is set.
                                                _mm256_maskstore_ps(&C[(m_idx + i + r) * ldc + current_n_start_for_store], mask, C_regs[r][c_vec]);
                                            }
                                        }
                                    }
                                }
                            }
                        } // end jj-loop (NR_AVX micro-panel step)
                    } // end i-loop (MR_AVX micro-panel step)

                } // end k_idx loop (BK block)
            } // end n_idx loop (BN block)
        } // end m_idx loop (BM block)

        // Free thread-local packed B buffer
        aligned_free(packed_B_buffer);
    } // end omp parallel region
}

#endif // __AVX2__

// --- Main function for demonstration and testing ---
int main(int argc, char* argv[]) {
    int M = 512, N = 512, K = 512;
    bool dump_matrices = false;
    int num_threads = omp_get_max_threads(); // Default to max available on the system, for Ryzen 6800HS (8 cores, SMT=2) this is 16.

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices] [--threads <num_threads>]" << std::endl;
        return 1;
    }

    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);
    K = std::stoi(argv[3]);

    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dump-matrices") {
            dump_matrices = true;
        } else if (arg == "--threads") {
            if (i + 1 < argc) {
                num_threads = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --threads requires an argument." << std::endl;
                return 1;
            }
        }
    }

    // Set the number of OpenMP threads
    // For "Compute Bound" workloads on SMT CPUs like Ryzen, sometimes using only physical cores
    // (e.g., 8 for an 8-core CPU with SMT-2) can perform better than using all logical cores (16).
    // This reduces resource contention within a physical core.
    omp_set_num_threads(num_threads);
    std::cout << "Running GEMM with M=" << M << ", N=" << N << ", K=" << K
              << ", Threads=" << num_threads << " (requested: " << num_threads << ")" << std::endl;
    std::cout << "Blocking: BM=" << BM << ", BN=" << BN << ", BK=" << BK
              << ", Micro-kernel: MR=" << MR_AVX << ", NR=" << NR_AVX << ", UNROLL_K=" << UNROLL_K << std::endl;

    // For this demo, lda=K, ldb=N, ldc=N assuming matrices are fully utilized and contiguous.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Allocate matrices using aligned memory
    float* A = aligned_alloc<float>(M * lda);
    float* B = aligned_alloc<float>(K * ldb);
    float* C = aligned_alloc<float>(M * ldc);
    float* C_ref = nullptr;

    if (dump_matrices) {
        C_ref = aligned_alloc<float>(M * ldc);
    }

    // Initialize A and B with random values, C and C_ref with zeros
    std::mt19937 gen(0); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < M * lda; ++i) A[i] = dis(gen);
    for (int i = 0; i < K * ldb; ++i) B[i] = dis(gen);
    std::fill(C, C + M * ldc, 0.0f);

    if (dump_matrices) {
        std::fill(C_ref, C_ref + M * ldc, 0.0f);
    }

    std::string workspace_dir = "workspace"; // Declare workspace_dir here to make it visible in both dump_matrices blocks

    if (dump_matrices) {
        // Create workspace directory for dumping matrices
        std::string command = "mkdir -p " + workspace_dir;
        // Ignoring return value of system for simplicity in demo, but in production, it should be checked.
        (void)system(command.c_str());

        write_matrix_to_file(workspace_dir + "/A.txt", A, M, K, lda);
        write_matrix_to_file(workspace_dir + "/B.txt", B, K, N, ldb);

        std::cout << "Running scalar GEMM for reference..." << std::endl;
        auto start_scalar = std::chrono::high_resolution_clock::now();
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        auto end_scalar = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> scalar_duration = end_scalar - start_scalar;
        std::cout << "Scalar GEMM took " << scalar_duration.count() << " seconds." << std::endl;
    }

    std::cout << "Running optimized AVX2 GEMM..." << std::endl;
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    #if defined(__AVX2__)
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    #else
        std::cerr << "AVX2 intrinsics not available (compiler flag -mavx2 missing or target CPU unsupported). Falling back to scalar." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc); // Fallback to scalar if AVX2 not compiled
    #endif
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> avx2_duration = end_avx2 - start_avx2;
    double gflops = (2.0 * M * N * K) / (avx2_duration.count() * 1e9);
    std::cout << "Optimized AVX2 GEMM took " << avx2_duration.count() << " seconds. GFLOPS: " << gflops << std::endl;


    if (dump_matrices) {
        write_matrix_to_file(workspace_dir + "/C.txt", C, M, N, ldc);

        // Correctness check
        double max_diff = 0.0;
        double sum_sq_diff = 0.0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                double diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
                max_diff = std::max(max_diff, diff);
                sum_sq_diff += diff * diff;
            }
        }
        double rmse = std::sqrt(sum_sq_diff / (M * N));
        // Floating point arithmetic is not perfectly associative, so a small tolerance is needed.
        // E.g., for M=N=K=512, a tolerance of 1e-4 is typically safe for float GEMM.
        double tolerance = 1e-4; 

        if (max_diff < tolerance) {
            std::cout << "Internal check: PASSED. Max difference: " << max_diff << ", RMSE: " << rmse << std::endl;
        } else {
            std::cout << "Internal check: FAILED. Max difference: " << max_diff << ", RMSE: " << rmse << std::endl;
            // Optionally print a few differing elements to help debug
            int diff_count = 0;
            for (int i = 0; i < M && diff_count < 10; ++i) {
                for (int j = 0; j < N && diff_count < 10; ++j) {
                    double diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
                    if (diff > tolerance) {
                        std::cerr << "Mismatch at C[" << i << "][" << j << "]: AVX2=" << C[i * ldc + j] << ", Ref=" << C_ref[i * ldc + j] << ", Diff=" << diff << std::endl;
                        diff_count++;
                    }
                }
            }
        }
    }

    // Free allocated memory
    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    if (C_ref) {
        aligned_free(C_ref);
    }

    return 0;
}