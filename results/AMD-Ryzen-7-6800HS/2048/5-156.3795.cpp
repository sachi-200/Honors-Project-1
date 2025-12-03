// Compile instructions:
// For gemm_scalar only:
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm
// For gemm_avx2 (as requested for target platform - AMD Ryzen 7 6800HS supporting AVX2, FMA):
// g++ -O3 -march=znver3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm
// clang++ -O3 -march=znver3 -mavx2 -mfma -fopenmp gemm.cpp -o gemm
// (Note: For AMD Zen 3/3+ (like 6800HS), `-march=native` would enable AVX2/FMA automatically.
// Explicitly `mavx2 -mfma` is robust, and `znver3` targets the specific microarchitecture.)

#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <random>
#include <cassert>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm> // For std::min

// Intrinsics header
#ifdef __AVX2__
#include <immintrin.h> // For AVX2 intrinsics
#endif

// OpenMP header
#ifdef _OPENMP
#include <omp.h>
#endif

// --- Autotuning Parameters (exposed as constants) ---
// Block sizes for M, N, K dimensions for cache-aware tiling.
// Chosen to be multiples of vector width and to fit in L1/L2 cache.
// Target CPU (AMD Ryzen 7 6800HS / Zen3/3+): 32KB L1d per core, 512KB L2 per core, 16MB L3 shared.

// BM=32 chosen to ensure the working set fits comfortably within the 32KB L1d cache.
constexpr int BM = 32;  // Block size for M (rows of C/A). Must be a multiple of MR_REG_BLOCK (4).
constexpr int BN = 64;  // Changed from 96 to 64. Block size for N (columns of C/B). Must be a multiple of NR_REG_BLOCK (16).
                        // Reduced to mitigate observed increased memory traffic and L1d misses, aiming for a smaller working set.
constexpr int BK = 32;  // Block size for K (inner dimension). Must be a multiple of UNROLL_K (4).

// Working Set Calculation for BM=32, BN=64, BK=32:
// A block (BMxBK): 32 * 32 * 4 bytes = 4KB
// B block (BKxBN): 32 * 64 * 4 bytes = 8KB
// C block (BMxBN): 32 * 64 * 4 bytes = 8KB
// Total active data: 4KB (A) + 8KB (B) + 8KB (C) = 20KB.
// This reduced working set (20KB) fits very comfortably within the 32KB L1d cache,
// aiming to further reduce L1 misses and improve performance under higher memory pressure.

constexpr int UNROLL_K = 4; // Unroll factor for the innermost K-loop in the micro-kernel.
                            // Reduces loop overhead and exposes more instruction-level parallelism.

// Register blocking for AVX2 micro-kernel.
// This defines how many elements of C are accumulated concurrently in SIMD registers.
constexpr int VECTOR_WIDTH = 8; // Number of float elements in a __m256 AVX2 vector.
constexpr int MR_REG_BLOCK = 4; // Number of rows of C (and corresponding A elements) processed per micro-kernel iteration.
                                // This means 4 A-scalars are broadcasted.
constexpr int NR_VEC_REG_BLOCK = 2; // Number of __m256 vectors for the N dimension of C.
                                    // This means 2 B-vectors are loaded. Reverted to 2 from 4 in a previous iteration
                                    // as 8 __m256 accumulators showed better performance balance on this architecture.
constexpr int NR_REG_BLOCK = NR_VEC_REG_BLOCK * VECTOR_WIDTH; // Total N columns processed by the micro-kernel (2 * 8 = 16).
// Total C accumulators: MR_REG_BLOCK * NR_VEC_REG_BLOCK = 4 * 2 = 8 __m256 registers.
// This configuration achieved optimal performance in a previous iteration, indicating a better balance
// for the target architecture's execution units and instruction scheduling.

// --- Helper for aligned memory allocation/deallocation ---
// Using posix_memalign for Linux (GCC/Clang) and _aligned_malloc for MSVC.
// This ensures memory is aligned to 32 bytes, required for some AVX2 instructions.
void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else // GCC/Clang
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

void aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else // GCC/Clang
    free(ptr);
#endif
}

// --- Matrix Helper Function ---
// Writes a matrix to a file in a space-separated format.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    ofs << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[(size_t)i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// --- GEMM Scalar Reference Implementation ---
// Computes C = A * B + C using standard triple-nested loops.
// Assumes row-major storage for A (M x K), B (K x N), C (M x N).
// Parameters lda, ldb, ldc are leading dimensions (strides) for row-major.
// Using `__restrict__` keyword to help the compiler with pointer aliasing assumptions.
void gemm_scalar(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[(size_t)i * lda + k] * B[(size_t)k * ldb + j];
            }
            C[(size_t)i * ldc + j] += sum;
        }
    }
}

// --- GEMM AVX2 Optimized Implementation ---
// Implements C = A * B + C using AVX2 and FMA intrinsics with cache-aware tiling and OpenMP.
// Assumes row-major storage.
// Using `__restrict__` keyword to help the compiler with pointer aliasing assumptions.
#if defined(__AVX2__) && defined(__FMA__)
void gemm_avx2(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // Outer loops for tiling M and N dimensions.
    // Parallelize over M and N blocks using OpenMP.
    // `schedule(static)` is chosen for large, uniform workloads to minimize scheduling overhead.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i_tile_start = 0; i_tile_start < M; i_tile_start += BM) {
        for (int j_tile_start = 0; j_tile_start < N; j_tile_start += BN) {
            // Inner loops for K dimension tiling.
            // This loop iterates over K-blocks.
            for (int k_tile_start = 0; k_tile_start < K; k_tile_start += BK) {

                // Determine effective block boundaries for current tiles, handling matrix edges.
                const int i_block_end = std::min(i_tile_start + BM, M);
                const int j_block_end = std::min(j_tile_start + BN, N);
                const int k_block_end = std::min(k_tile_start + BK, K);

                // Micro-kernel processing C in MR_REG_BLOCK x NR_REG_BLOCK blocks.
                // Iterates through the current M and N tiles, processing sub-blocks of C.
                for (int i = i_tile_start; i < i_block_end; i += MR_REG_BLOCK) {
                    for (int j = j_tile_start; j < j_block_end; j += NR_REG_BLOCK) {

                        // --- Pre-calculation for N-dimension tail handling (moved outside k_val loop) ---
                        // Determine if each NR_VEC_REG_BLOCK segment for B and C is a full vector or needs masking.
                        // This avoids redundant mask generation and conditional branches in the hottest inner loop.
                        bool is_full_vector_n[NR_VEC_REG_BLOCK];
                        alignas(32) int mask_elements[NR_VEC_REG_BLOCK][VECTOR_WIDTH]; // Backing store for mask values
                        __m256i m256i_masks[NR_VEC_REG_BLOCK]; // Pre-loaded __m256i masks for _mm256_maskload/store

                        for (int nv_idx = 0; nv_idx < NR_VEC_REG_BLOCK; ++nv_idx) {
                            const int current_j_offset = j + nv_idx * VECTOR_WIDTH;
                            if (current_j_offset + VECTOR_WIDTH <= j_block_end) {
                                is_full_vector_n[nv_idx] = true;
                            } else {
                                is_full_vector_n[nv_idx] = false;
                                const int remaining_N_in_vector = j_block_end - current_j_offset;
                                // Populate mask elements: -1 for active, 0 for inactive
                                for (int m_idx = 0; m_idx < VECTOR_WIDTH; ++m_idx) {
                                    mask_elements[nv_idx][m_idx] = (m_idx < remaining_N_in_vector) ? -1 : 0;
                                }
                                m256i_masks[nv_idx] = _mm256_load_si256((__m256i*)mask_elements[nv_idx]);
                            }
                        }
                        // --- End pre-calculation for N-tail ---

                        // Initialize C accumulators to zero for this C block (MR_REG_BLOCK x NR_REG_BLOCK).
                        // These __m256 registers will hold partial sums for a C sub-block.
                        __m256 c_acc[MR_REG_BLOCK][NR_VEC_REG_BLOCK];
                        for (int mr_idx = 0; mr_idx < MR_REG_BLOCK; ++mr_idx) {
                            for (int nv_idx = 0; nv_idx < NR_VEC_REG_BLOCK; ++nv_idx) {
                                c_acc[mr_idx][nv_idx] = _mm256_setzero_ps();
                            }
                        }

                        // Accumulate over K-dimension for the current C block.
                        // Iterates through the current K-tile, advancing by UNROLL_K steps.
                        for (int k_inner = k_tile_start; k_inner < k_block_end; k_inner += UNROLL_K) {
                            const int k_unroll_limit = std::min(k_inner + UNROLL_K, k_block_end);

                            // --- Removed Explicit Prefetch for B matrix ---
                            // Previous attempts with _mm_prefetch (both _MM_HINT_T0 and _MM_HINT_NTA) for the B matrix
                            // led to performance regression on the target architecture. This indicates that the
                            // hardware prefetchers are likely more effective for this strided access pattern, or
                            // explicit prefetching caused too much overhead/cache pollution.
                            // The prefetch block has been removed to allow hardware prefetchers to operate optimally.

                            // Inner-most K-loop, unrolled by UNROLL_K.
                            // Each iteration processes one k-value.
                            for (int k_val = k_inner; k_val < k_unroll_limit; ++k_val) {
                                // Load NR_VEC_REG_BLOCK vectors from B.
                                // B is K x N (row-major), so B[k_val * ldb + j_current + offset].
                                __m256 b_vec[NR_VEC_REG_BLOCK];
                                for (int nv_idx = 0; nv_idx < NR_VEC_REG_BLOCK; ++nv_idx) {
                                    const int current_j_offset = j + nv_idx * VECTOR_WIDTH;
                                    if (is_full_vector_n[nv_idx]) { // Use pre-calculated status
                                        b_vec[nv_idx] = _mm256_loadu_ps(B + (size_t)k_val * ldb + current_j_offset);
                                    } else { // Masked load for partial vector at N-tail
                                        b_vec[nv_idx] = _mm256_maskload_ps(B + (size_t)k_val * ldb + current_j_offset, m256i_masks[nv_idx]);
                                    }
                                }

                                // For each row in the C block (MR_REG_BLOCK rows).
                                for (int mr_idx = 0; mr_idx < MR_REG_BLOCK; ++mr_idx) {
                                    const int current_i = i + mr_idx;
                                    if (current_i >= i_block_end) {
                                        continue; // Skip if beyond M block boundary (M-tail processing)
                                    }

                                    // Load A scalar (A[current_i * lda + k_val]) and broadcast it into a vector.
                                    // This scalar is multiplied with B vectors.
                                    __m256 a_scalar_broadcast = _mm256_broadcast_ss(A + (size_t)current_i * lda + k_val);

                                    // Perform FMA (Fused Multiply-Add) with all B vectors for this A scalar.
                                    // c_acc += a_scalar_broadcast * b_vec
                                    for (int nv_idx = 0; nv_idx < NR_VEC_REG_BLOCK; ++nv_idx) {
                                        c_acc[mr_idx][nv_idx] = _mm256_fmadd_ps(a_scalar_broadcast, b_vec[nv_idx], c_acc[mr_idx][nv_idx]);
                                    }
                                }
                            } // end k_val (unroll) loop
                        } // end k_inner loop (K-block accumulation)

                        // Store results from c_acc to C matrix.
                        // C is M x N (row-major), so C[current_i * ldc + current_j_offset].
                        for (int mr_idx = 0; mr_idx < MR_REG_BLOCK; ++mr_idx) {
                            const int current_i = i + mr_idx;
                            if (current_i >= i_block_end) {
                                continue; // Skip if beyond M block boundary
                            }

                            for (int nv_idx = 0; nv_idx < NR_VEC_REG_BLOCK; ++nv_idx) {
                                const int current_j_offset = j + nv_idx * VECTOR_WIDTH;
                                if (is_full_vector_n[nv_idx]) { // Use pre-calculated status
                                    // Load existing C, add accumulated value, then store.
                                    _mm256_storeu_ps(C + (size_t)current_i * ldc + current_j_offset,
                                                     _mm256_add_ps(c_acc[mr_idx][nv_idx], _mm256_loadu_ps(C + (size_t)current_i * ldc + current_j_offset)));
                                } else { // Masked store for partial vector at N-tail
                                    // Load existing C, add accumulated value, then masked store.
                                    __m256 c_existing = _mm256_maskload_ps(C + (size_t)current_i * ldc + current_j_offset, m256i_masks[nv_idx]);
                                    __m256 c_new = _mm256_add_ps(c_existing, c_acc[mr_idx][nv_idx]);
                                    _mm256_maskstore_ps(C + (size_t)current_i * ldc + current_j_offset, m256i_masks[nv_idx], c_new);
                                }
                            }
                        }
                    } // end j (NR_REG_BLOCK) loop
                } // end i (MR_REG_BLOCK) loop
            } // end k_tile_start loop (K-tiling)
        } // end j_tile_start loop (N-tiling)
    } // end i_tile_start loop (M-tiling)
}
#else // AVX2 or FMA not defined, provide a fallback (will print an error).
void gemm_avx2(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "Error: gemm_avx2 called but AVX2 or FMA intrinsics are not enabled/supported by the compiler or target architecture." << std::endl;
    std::cerr << "Falling back to scalar implementation." << std::endl;
    // Fallback to scalar for correctness, though performance will be poor.
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif

// --- Main Function ---
int main(int argc, char* argv[]) {
    // Parse command-line arguments: M N K [--dump-matrices]
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump_matrices = false;

    if (argc >= 5 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    // Assert positive dimensions for valid matrix operations.
    assert(M > 0 && N > 0 && K > 0 && "M, N, K must be positive dimensions.");

    // Leading dimensions (strides) for row-major storage.
    // For contiguous row-major, lda = K, ldb = N, ldc = N.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Allocate matrices using aligned_malloc for AVX2 compatibility (32-byte alignment).
    const size_t alignment = 32; // AVX2 vectors are 32 bytes.
    float* A_mat = static_cast<float*>(aligned_malloc(static_cast<size_t>(M) * lda * sizeof(float), alignment));
    float* B_mat = static_cast<float*>(aligned_malloc(static_cast<size_t>(K) * ldb * sizeof(float), alignment));
    float* C_opt = static_cast<float*>(aligned_malloc(static_cast<size_t>(M) * ldc * sizeof(float), alignment));

    if (!A_mat || !B_mat || !C_opt) {
        std::cerr << "Memory allocation failed for A, B, or C_opt!" << std::endl;
        aligned_free(A_mat);
        aligned_free(B_mat);
        aligned_free(C_opt);
        return 1;
    }

    float* C_ref = nullptr;
    if (dump_matrices) {
        C_ref = static_cast<float*>(aligned_malloc(static_cast<size_t>(M) * ldc * sizeof(float), alignment));
        if (!C_ref) {
            std::cerr << "Memory allocation for C_ref failed!" << std::endl;
            aligned_free(A_mat);
            aligned_free(B_mat);
            aligned_free(C_opt);
            return 1;
        }
    }

    // Initialize matrices with random floats.
    // C matrices are initialized to zero as the GEMM computes C = A*B + C.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Values between 0.0 and 1.0

    for (size_t i = 0; i < static_cast<size_t>(M) * lda; ++i) A_mat[i] = dis(gen);
    for (size_t i = 0; i < static_cast<size_t>(K) * ldb; ++i) B_mat[i] = dis(gen);
    for (size_t i = 0; i < static_cast<size_t>(M) * ldc; ++i) C_opt[i] = 0.0f;
    if (dump_matrices) {
        for (size_t i = 0; i < static_cast<size_t>(M) * ldc; ++i) C_ref[i] = 0.0f;
    }

    if (dump_matrices) {
        // --- Dump Mode: Compute reference, optimized, dump matrices, and verify ---
        std::cout << "--- Dump Mode ---" << std::endl;
        std::cout << "Dumping matrices to workspace/A.txt, workspace/B.txt, workspace/C.txt" << std::endl;
        
        // Note: For cross-platform directory creation, C++17 <filesystem> would be ideal.
        // For this specific context (Linux), `mkdir -p workspace` command line prior to execution is assumed.
        // `write_matrix_to_file` will log an error if the directory doesn't exist.

        write_matrix_to_file("workspace/A.txt", A_mat, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B_mat, K, N, ldb);

        std::cout << "Running gemm_scalar for reference result..." << std::endl;
        gemm_scalar(A_mat, B_mat, C_ref, M, N, K, lda, ldb, ldc);

        std::cout << "Running gemm_avx2 for optimized result..." << std::endl;
        gemm_avx2(A_mat, B_mat, C_opt, M, N, K, lda, ldb, ldc);

        write_matrix_to_file("workspace/C.txt", C_opt, M, N, ldc);

        // Correctness check: Compare optimized C with reference C_ref
        float max_diff = 0.0f;
        const float epsilon = 1e-4f; // A reasonable epsilon for float comparisons
        bool passed = true;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float diff = std::abs(C_opt[(size_t)i * ldc + j] - C_ref[(size_t)i * ldc + j]);
                if (diff > epsilon) {
                    passed = false;
                    if (diff > max_diff) {
                        max_diff = diff;
                    }
                }
            }
        }

        if (passed) {
            std::cout << "Internal check: PASSED" << std::endl;
        } else {
            std::cerr << "Internal check: FAILED (Max diff: " << max_diff << ")" << std::endl;
        }

    } else {
        // --- Performance Mode: Run optimized kernel and report time/GFLOPS ---
        std::cout << "--- Performance Mode ---" << std::endl;
        std::cout << "Running gemm_avx2 for M=" << M << ", N=" << N << ", K=" << K << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();
        gemm_avx2(A_mat, B_mat, C_opt, M, N, K, lda, ldb, ldc);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end_time - start_time;
        double seconds = duration.count();
        // GFLOPS = (2 * M * N * K) / (time_in_seconds * 1e9)
        // 2 operations per element of C: one multiply and one add.
        double gflops = (2.0 * M * N * K) / (seconds * 1e9);

        std::cout << "Execution time: " << std::fixed << std::setprecision(6) << seconds << " seconds" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
    }

    // Free all allocated memory.
    aligned_free(A_mat);
    aligned_free(B_mat);
    aligned_free(C_opt);
    if (dump_matrices) {
        aligned_free(C_ref);
    }

    return 0;
}