// Compile with g++ or clang++:
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm
// clang++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm
// For older GCC/Clang or if specific v2 not supported, use -march=native
// to enable ISA extensions detected on the host machine.

#include <immintrin.h> // AVX2 intrinsics
#include <iostream>    // For std::cout, std::cerr
#include <vector>      // For std::vector
#include <cstring>     // For std::memcpy
#include <chrono>      // For std::chrono high_resolution_clock
#include <random>      // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <cassert>     // For assert
#include <fstream>     // For std::ofstream
#include <string>      // For std::string
#include <iomanip>     // For std::fixed, std::setprecision
#include <algorithm>   // For std::min, std::max
#include <cmath>       // For std::abs

// For OpenMP parallelism
#if defined(_OPENMP)
#include <omp.h>
#endif

// --- Autotuning Parameters ---
// These parameters control the blocking/tiling strategy and micro-kernel behavior.
// They are chosen to favor cache reuse and SIMD efficiency on x86-64 AVX2 CPUs.

// BM: Block M size (rows of A and C).
// Controls the number of M-rows processed in a larger block, suitable for L2/L3 cache.
// A larger BM helps amortize setup costs and allows more parallelism.
// CRITICAL PERFORMANCE FIX (from 128:5): Changed from 96 to 32 for better parallelism on smaller matrices.
constexpr int BM = 32;

// BN: Block N size (columns of B and C).
// Controls the number of N-columns processed in a larger block, suitable for L2/L3 cache.
// A larger BN helps amortize setup costs and allows more parallelism.
// CRITICAL PERFORMANCE FIX (from 128:5): Changed from 128 to 32 for better parallelism on smaller matrices.
constexpr int BN = 32;

// BK: Block K size (columns of A / rows of B).
// This is the "inner" blocking dimension. A block of A (BMxBK) and B (BKxBN) is loaded
// and reused for computing a BMxBN block of C. A smaller BK fits better in L1/L2 cache.
// Crucial for data reuse of B's packed data.
constexpr int BK = 128; // On AMD Ryzen 7 6800HS: 32KB L1D, 512KB L2 per core.
                        // With BM=32, BN=32, BK=128:
                        // A_block = BM * BK * sizeof(float) = 32 * 128 * 4 = 16384 bytes (~16KB)
                        // B_packed_block = BK * BN * sizeof(float) = 128 * 32 * 4 = 16384 bytes (~16KB)
                        // Total working set ~32KB, which fits comfortably within L1D (32KB L1D/core).
                        // This improves L1 cache hit rates for B, which is now streamed much more efficiently.

// MR: Micro-kernel M-dimension (scalar rows of C).
// Number of rows of C computed simultaneously in the innermost micro-kernel.
// This directly translates to the number of A-values broadcast and C-accumulators.
constexpr int MR = 4; // We use 4 `__m256` accumulators for 4 rows of C.

// NR: Micro-kernel N-dimension (vectorized columns of C).
// This must be equal to the vector width of the chosen SIMD instruction set.
// For AVX2 (float), `__m256` holds 8 floats.
constexpr int VEC_FLOATS_F32 = 8;
constexpr int NR = VEC_FLOATS_F32;

// UNROLL_K: Unroll factor for the innermost K loop.
// Helps to expose Instruction-Level Parallelism (ILP) and hide FMA latency.
// A larger unroll factor means more independent operations can be scheduled.
// On Zen architectures, 4 is a good balance for throughput and latency hiding with 2 FMA units.
constexpr int UNROLL_K = 4;

// NUM_THREADS: Default number of threads to use if OMP_NUM_THREADS is not set.
// Ryzen 7 6800HS has 8 physical cores (16 logical). 8 is a good starting point for physical cores.
constexpr int NUM_THREADS = 8;

// --- OpenMP Helper Macro ---
#if defined(_OPENMP)
#define OMP_SET_NUM_THREADS(n) omp_set_num_threads(n)
#else
#define OMP_SET_NUM_THREADS(n)
#endif

// --- Matrix storage convention ---
// All matrices are assumed to be in row-major order (C-style).
// An element at (row, col) in a matrix with leading dimension `ld` is accessed as `matrix[row * ld + col]`.

// --- Function Implementations ---

/**
 * @brief Reference scalar GEMM implementation. C = A * B.
 *        This version is single-threaded and used for correctness verification.
 * @param A Pointer to the A matrix (M x K).
 * @param B Pointer to the B matrix (K x N).
 * @param C Pointer to the C matrix (M x N). C must be initialized to zero.
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
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float c_val = 0.0f;
            for (int k = 0; k < K; ++k) {
                c_val += A[m * lda + k] * B[k * ldb + n];
            }
            C[m * ldc + n] = c_val;
        }
    }
}

/**
 * @brief Optimized AVX2+FMA GEMM implementation. C = A * B.
 *        Uses cache blocking, OpenMP for parallelism, and AVX2 intrinsics with FMA for compute.
 *        C must be initialized to zero before calling this function.
 * @param A Pointer to the A matrix (M x K).
 * @param B Pointer to the B matrix (K x N).
 * @param C Pointer to the C matrix (M x N). C must be initialized to zero.
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
#if defined(__AVX2__) && defined(__FMA__)
    // Set the number of threads for OpenMP. Can be overridden by OMP_NUM_THREADS environment variable.
    OMP_SET_NUM_THREADS(NUM_THREADS);

    // Enter a parallel region. Each thread will execute the code inside this block.
    // CRITICAL FIX 1 (from 128:3, corrected 128:5): packed_B_buffer is allocated once per thread inside this region.
    #pragma omp parallel
    {
        // Allocate a temporary buffer for packing blocks of B for the current thread.
        // Allocated once per thread using _mm_malloc for 32-byte alignment.
        // This memory will be freed by the thread when it exits the parallel region.
        // Max size needed is BK * BN floats.
        float* packed_B_buffer = (float*)_mm_malloc(static_cast<size_t>(BK) * BN * sizeof(float), 32);
        if (!packed_B_buffer) {
            std::cerr << "Error: _mm_malloc failed for packed_B_buffer in thread " << omp_get_thread_num() << std::endl;
            assert(false && "Failed to allocate aligned memory for packed_B_buffer!");
        }

        // The outer loops for M and N dimensions are parallelized.
        // `collapse(2)` distributes the (M/BM) * (N/BN) blocks among threads.
        // `schedule(static)` ensures a predictable, balanced distribution of tasks.
        // With smaller BM/BN (e.g., 32x32), this creates more tasks for better thread utilization.
        #pragma omp for collapse(2) schedule(static)
        for (int bm_start = 0; bm_start < M; bm_start += BM) {
            for (int bn_start = 0; bn_start < N; bn_start += BN) {
                // Determine actual block dimensions for the current C tile, handling edges.
                int current_BM = std::min(BM, M - bm_start);
                int current_BN = std::min(BN, N - bn_start);

                // Now, iterate over MR-row tiles of C within the current BM block.
                for (int m_tile_start = bm_start; m_tile_start < bm_start + current_BM; m_tile_start += MR) {
                    int current_MR = std::min(MR, (bm_start + current_BM) - m_tile_start);

                    // Iterate over NR-column (vector-width) tiles of C within the current BN block.
                    // n_tile_start is an offset within current_BN.
                    for (int n_tile_start = 0; n_tile_start < current_BN; n_tile_start += NR) {
                        // C_acc: Accumulator registers for the current MR x NR micro-block of C.
                        // Each `__m256` register holds 8 float results for one row of C.
                        // So `c_acc[r]` accumulates for row `m_tile_start + r`.
                        __m256 c_acc[MR];

                        // CRITICAL FIX 1 (from 128:3, corrected 128:5): Initialize C accumulators ONCE for this MR x NR micro-block.
                        // They will accumulate contributions from ALL K-blocks.
                        for (int r = 0; r < current_MR; ++r) {
                            c_acc[r] = _mm256_setzero_ps();
                        }

                        // Loop over K blocks. This is the crucial loop for data reuse and accumulation.
                        for (int bk_start = 0; bk_start < K; bk_start += BK) {
                            int current_BK = std::min(BK, K - bk_start);

                            // PACK THE CURRENT BK x current_BN BLOCK OF B HERE
                            // This block is packed ONCE per (bk_start, bn_start) iteration by each thread.
                            // It converts the strided B data into a contiguous block in `packed_B_buffer`.
                            // This packed block is then repeatedly accessed by the micro-kernel for all M-tiles.
                            // This significantly improves cache locality for B.
                            for (int k_pack = 0; k_pack < current_BK; ++k_pack) {
                                // Source: Row (bk_start + k_pack) of original B, starting from column bn_start.
                                const float* src_ptr = B + (bk_start + k_pack) * ldb + bn_start;
                                // Destination: Row k_pack in packed_B_buffer (which has width current_BN).
                                float* dest_ptr = packed_B_buffer + k_pack * current_BN;
                                std::memcpy(dest_ptr, src_ptr, current_BN * sizeof(float));
                            }

                            // Innermost K loop: Process `current_BK` elements for the current C micro-block.
                            // This loop is unrolled by `UNROLL_K` to expose ILP and hide FMA latency.
                            for (int k_micro = 0; k_micro < current_BK; k_micro += UNROLL_K) {
                                int current_UNROLL_K = std::min(UNROLL_K, current_BK - k_micro);

                                for (int u = 0; u < current_UNROLL_K; ++u) {
                                    int k_current = k_micro + u; // Current K index within the BK block

                                    // CRITICAL FIX 2 (from 128:3, corrected 128:5): Use masked load for B-vectors in N-tail regions.
                                    // This prevents reading beyond the valid data of current_BN in packed_B_buffer.
                                    __m256 b_vec;
                                    // Calculate how many floats are valid in the current vector slice within current_BN
                                    int num_floats_to_load = std::min(NR, current_BN - n_tile_start);

                                    if (num_floats_to_load == NR) {
                                        // Full vector load, use unaligned load as the actual memory address
                                        // (&packed_B_buffer[k_current * current_BN + n_tile_start]) might not be 32-byte aligned
                                        // due to `current_BN` potentially not being a multiple of `NR`. `_mm256_loadu_ps` is robust.
                                        b_vec = _mm256_loadu_ps(&packed_B_buffer[k_current * current_BN + n_tile_start]);
                                    } else {
                                        // Partial vector load using a mask to only load valid elements.
                                        // Create a mask where active lanes have all bits set (-1 integer) for valid elements.
                                        alignas(32) int mask_array[VEC_FLOATS_F32];
                                        for (int i = 0; i < VEC_FLOATS_F32; ++i) {
                                            mask_array[i] = (i < num_floats_to_load) ? -1 : 0;
                                        }
                                        __m256i mask_vec = _mm256_load_si256((__m256i*)mask_array); // Load aligned mask
                                        // Use masked load for the potentially unaligned and partial vector.
                                        b_vec = _mm256_maskload_ps(&packed_B_buffer[k_current * current_BN + n_tile_start], mask_vec);
                                    }

                                    // Process MR rows of A and accumulate into C accumulators.
                                    // This involves loading a single float from A and broadcasting it.
                                    for (int r = 0; r < current_MR; ++r) {
                                        // Load a single float from A and broadcast it to all lanes of an AVX2 vector.
                                        float a_val = A[(m_tile_start + r) * lda + (bk_start + k_current)];
                                        __m256 a_broadcast = _mm256_set1_ps(a_val);

                                        // Fused Multiply-Add: c_acc[r] = (a_broadcast * b_vec) + c_acc[r].
                                        // This is the core computation, highly efficient on FMA-enabled CPUs.
                                        c_acc[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                    }
                                } // End of UNROLL_K loop
                            } // End of K-micro loop
                        } // End of K-block loop. All K contributions accumulated for this C micro-block.

                        // CRITICAL FIX 1 (from 128:3, corrected 128:5): Store the accumulated C values ONCE after all K contributions are processed.
                        for (int r = 0; r < current_MR; ++r) {
                            float* c_ptr = &C[(m_tile_start + r) * ldc + (bn_start + n_tile_start)];

                            // Handle N-tail: if (bn_start + n_tile_start + NR) exceeds N,
                            // only store the valid portion of the vector.
                            int num_cols_to_store = std::min(NR, N - (bn_start + n_tile_start));

                            if (num_cols_to_store == NR) {
                                // Store full vector (8 floats). `_mm256_storeu_ps` for unaligned store,
                                // as the target C matrix may not be 32-byte aligned.
                                _mm256_storeu_ps(c_ptr, c_acc[r]);
                            } else {
                                // Store partial vector using a mask.
                                // Create a mask where active lanes have all bits set (-1 integer).
                                alignas(32) int mask_array[VEC_FLOATS_F32];
                                for (int i = 0; i < VEC_FLOATS_F32; ++i) {
                                    mask_array[i] = (i < num_cols_to_store) ? -1 : 0;
                                }
                                __m256i mask_vec = _mm256_load_si256((__m256i*)mask_array); // Load aligned mask
                                _mm256_maskstore_ps(c_ptr, mask_vec, c_acc[r]);
                            }
                        }
                    } // End of N-tile loop
                } // End of M-tile loop
            } // End of BN block loop
        } // End of BM block loop

        // CRITICAL FIX 1 (from 128:3, corrected 128:5): Free the per-thread aligned memory when the thread exits the parallel region.
        _mm_free(packed_B_buffer);
    } // End of omp parallel
#else // Fallback if AVX2 or FMA is not available at compile time
    std::cerr << "Warning: AVX2/FMA intrinsics not available or not enabled. Falling back to scalar implementation for gemm_avx2." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif // __AVX2__ && __FMA__
}

// Helper function to write a matrix to a text file.
// Corrected to *not* write dimensions on the first line to comply with the verification script.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    // Write matrix elements
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << std::fixed << std::setprecision(6) << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << std::endl;
    }
    ofs.close();
}

// Main function for demonstration, performance measurement, and correctness checking.
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump_matrices = false;

    // Check for optional --dump-matrices flag
    if (argc > 4 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    std::cout << "GEMM dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    std::cout << "Target CPU: AMD Ryzen 7 6800HS (x86_64, AVX2, FMA)" << std::endl;
    std::cout << "Using AVX2/FMA optimized kernel." << std::endl;
#if defined(_OPENMP)
    // Set default number of threads, can be overridden by OMP_NUM_THREADS environment variable.
    OMP_SET_NUM_THREADS(NUM_THREADS);
    // Query and print the actual number of threads being used by OpenMP.
    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0) {
            std::cout << "OpenMP enabled. Actual threads: " << omp_get_num_threads() << std::endl;
        }
    }
#else
    std::cout << "OpenMP not enabled (compile with -fopenmp to enable)." << std::endl;
#endif

    // Using standard leading dimensions for dense row-major matrices
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Allocate memory for matrices A, B, C (optimized), and C_ref (scalar reference).
    // Using std::vector to manage memory. C matrices are initialized to zero.
    std::vector<float> A_vec(M * K);
    std::vector<float> B_vec(K * N);
    std::vector<float> C_vec_optimized(M * N, 0.0f);
    std::vector<float> C_vec_scalar; // Allocated only if dump_matrices is true

    // Initialize A and B with random float values between 0.0 and 1.0.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) A_vec[i] = dis(gen);
    for (int i = 0; i < K * N; ++i) B_vec[i] = dis(gen);

    if (dump_matrices) {
        std::cout << "--- Test Mode: Dumping matrices and performing correctness check ---" << std::endl;
        // Create 'workspace' directory if it doesn't exist
        (void)system("mkdir -p workspace"); // Fix: Suppress unused result warning

        // Dump input matrices A and B to files
        write_matrix_to_file("workspace/A.txt", A_vec.data(), M, K, lda);
        write_matrix_to_file("workspace/B.txt", B_vec.data(), K, N, ldb);
        std::cout << "Input matrices A.txt and B.txt dumped to 'workspace/'." << std::endl;

        // Compute reference C using the scalar implementation
        C_vec_scalar.resize(M * N, 0.0f); // Initialize C_ref to zero
        std::cout << "Calculating reference GEMM (scalar)..." << std::endl;
        gemm_scalar(A_vec.data(), B_vec.data(), C_vec_scalar.data(), M, N, K, lda, ldb, ldc);
        std::cout << "Reference GEMM done." << std::endl;
    } else {
        std::cout << "--- Performance Mode: Running optimized GEMM only ---" << std::endl;
    }

    // Compute optimized C using the AVX2+FMA implementation
    std::cout << "Calculating optimized GEMM (AVX2)..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm_avx2(A_vec.data(), B_vec.data(), C_vec_optimized.data(), M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized GEMM done." << std::endl;

    double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    double gflops = 2.0 * M * N * K / (duration_ms * 1e6); // 2 ops (mul + add) per element
    std::cout << "Optimized GEMM took " << std::fixed << std::setprecision(3) << duration_ms << " ms, "
              << std::fixed << std::setprecision(2) << gflops << " GFLOPS." << std::endl;

    if (dump_matrices) {
        // Dump optimized C result
        write_matrix_to_file("workspace/C.txt", C_vec_optimized.data(), M, N, ldc);
        std::cout << "Optimized result C.txt dumped to 'workspace/'." << std::endl;

        // Perform correctness check by comparing optimized C with scalar C_ref
        float max_diff = 0.0f;
        int diff_count = 0;
        for (int i = 0; i < M * N; ++i) {
            float diff = std::abs(C_vec_optimized[i] - C_vec_scalar[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > 1e-4f) { // Count elements exceeding tolerance
                diff_count++;
            }
        }

        float epsilon = 1e-4f; // Tolerance for float comparisons
        if (max_diff < epsilon) {
            std::cout << "Internal check: PASSED (Max absolute difference: " << max_diff << ")" << std::endl;
        } else {
            std::cout << "Internal check: FAILED (Max absolute difference: " << max_diff
                      << " which is >= " << epsilon << ". Number of differing elements: " << diff_count << ")" << std::endl;
        }
    }

    return 0;
}