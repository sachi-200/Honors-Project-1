// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2 -std=c++17

#include <immintrin.h> // For AVX2 and FMA intrinsics
#include <iostream>    // For console I/O
#include <vector>      // For std::vector (dynamic arrays)
#include <cstring>     // For std::memcpy
#include <chrono>      // For timing
#include <random>      // For random matrix initialization
#include <cassert>     // For assert
#include <fstream>     // For file I/O
#include <string>      // For std::string
#include <iomanip>     // For std::fixed, std::setprecision
#include <algorithm>   // For std::min

#ifdef _OPENMP
#include <omp.h>       // For OpenMP parallelization
#endif

// --- Autotuning Parameters ---
// These parameters are crucial for cache-aware tiling and micro-kernel optimization.
// They are chosen to favor L1D cache residency for the inner computation blocks.
// Target CPU: AMD Ryzen 7 6800HS (Zen 3/3+), L1d: 32KB/core, L2: 512KB/core.

// Tile sizes for M, N, K dimensions for the outer loops.
// A block of A is BM x BK. A block of B is BK x BN.
// BM = 32, BK = 64 => A_block = 32 * 64 * 4 bytes = 8 KB
// BN = 64, BK = 64 => B_block = 64 * 64 * 4 bytes = 16 KB
// Total data in L1 for one K-block iteration: ~24 KB (fits well within 32KB L1d).
constexpr int BM = 32; // Block size for M dimension
constexpr int BN = 64; // Block size for N dimension
constexpr int BK = 64; // Block size for K dimension

// Micro-kernel register blocking factors.
// MR_AVX2 specifies how many rows of C are computed simultaneously.
// NR_AVX2 specifies how many AVX2 vectors (NR_AVX2 * VEC_WIDTH columns) of C are computed simultaneously.
// This results in MR_AVX2 * NR_AVX2 __m256 accumulators.
// AVX2 has 16 YMM registers. 4x4 = 16 registers. This is an optimal fit, maximizing register reuse.
constexpr int MR_AVX2 = 4; // Register block rows for M
constexpr int NR_AVX2 = 4; // Register block vectors for N (each vector is VEC_WIDTH floats)

// Inner K loop unroll factor.
// Reduces loop overhead and exposes more Instruction-Level Parallelism (ILP) to the CPU.
constexpr int UNROLL_K = 4;

// Vector width for float with AVX2 intrinsics.
constexpr int VEC_WIDTH = 8; // 8 floats per __m256 register

// --- Helper function for matrix I/O ---
// Writes a matrix to a text file for debugging and verification.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << std::fixed << std::setprecision(4); // Format output for readability
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

// --- Scalar GEMM Implementation (Reference) ---
// This is a straightforward, unoptimized triple-nested loop implementation.
// It serves as a correctness reference for the optimized kernels.
// Assumes row-major storage: A[row*lda + col], B[row*ldb + col], C[row*ldc + col].
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

// --- AVX2 Optimized GEMM Implementation ---
// Implements a high-performance GEMM using AVX2 and FMA intrinsics, OpenMP,
// and cache-aware tiling with register blocking.
// Assumes row-major storage: A[row*lda + col], B[row*ldb + col], C[row*ldc + col].
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

#if defined(__AVX2__) && defined(__FMA__)
    // Outer loops for M and N dimensions, iterating over tiles (BM x BN blocks).
    // OpenMP parallelization is applied here to distribute workload across available cores.
    // 'collapse(2)' allows parallelizing both M and N loops, providing ample parallelism.
    // 'schedule(static)' is generally a good choice for uniform workloads like GEMM blocks,
    // ensuring predictable distribution and minimal overhead.
#pragma omp parallel for collapse(2) schedule(static)
    for (int m = 0; m < M; m += BM) {
        for (int n = 0; n < N; n += BN) {
            // Loop for K dimension, iterating over BK-sized blocks.
            // This loop iterates over the depth dimension, accumulating partial results.
            for (int k = 0; k < K; k += BK) {

                // Determine the actual bounds for the current M, N, K blocks.
                // std::min handles cases where dimensions are not multiples of tile sizes.
                int M_block_end = std::min(m + BM, M);
                int N_block_end = std::min(n + BN, N);
                int K_block_end = std::min(k + BK, K);

                // Micro-kernel loops: operate on smaller MR_AVX2 x (NR_AVX2 * VEC_WIDTH) blocks of C.
                // These loops iterate within the current BM x BN block.
                for (int ii = m; ii < M_block_end; ii += MR_AVX2) {
                    // Current number of rows for this micro-block of C, handling M dimension tail.
                    int MR_current = std::min(MR_AVX2, M_block_end - ii);

                    for (int jj = n; jj < N_block_end; jj += NR_AVX2 * VEC_WIDTH) {
                        // Accumulator registers for the C micro-block (MR_AVX2 rows x NR_AVX2 vectors).
                        __m256 c_acc[MR_AVX2][NR_AVX2];

                        // Initialize C accumulators:
                        // If this is the first K-block (k == 0), initialize with zeros.
                        // Otherwise, load existing values from C to accumulate onto.
                        if (k == 0) {
                            for (int r = 0; r < MR_current; ++r) {
                                for (int c_vec_idx = 0; c_vec_idx < NR_AVX2; ++c_vec_idx) {
                                    c_acc[r][c_vec_idx] = _mm256_setzero_ps();
                                }
                            }
                        } else {
                            for (int r = 0; r < MR_current; ++r) {
                                int row_c = ii + r;
                                for (int c_vec_idx = 0; c_vec_idx < NR_AVX2; ++c_vec_idx) {
                                    int col_c = jj + c_vec_idx * VEC_WIDTH;
                                    // Handle partial vectors at the N-dimension tail of the matrix.
                                    if (col_c + VEC_WIDTH <= N_block_end) {
                                        // Full vector load.
                                        c_acc[r][c_vec_idx] = _mm256_loadu_ps(C + row_c * ldc + col_c);
                                    } else if (col_c < N_block_end) {
                                        // Partial vector load using a temporary aligned buffer and memcpy.
                                        // This is a robust way to handle unaligned or partial loads/stores.
                                        alignas(32) float temp_buffer[VEC_WIDTH] = {0.0f};
                                        std::memcpy(temp_buffer, C + row_c * ldc + col_c, (N_block_end - col_c) * sizeof(float));
                                        c_acc[r][c_vec_idx] = _mm256_load_ps(temp_buffer);
                                    } else {
                                        // Should not be reached if jj loop correctly bounds.
                                        c_acc[r][c_vec_idx] = _mm256_setzero_ps();
                                    }
                                }
                            }
                        }

                        // Inner K loop: processes the current K-block, unrolled by UNROLL_K.
                        for (int kk_inner = k; kk_inner < K_block_end; kk_inner += UNROLL_K) {
                            // Unroll loop for K dimension (UNROLL_K factor).
                            // This minimizes loop overhead and allows the compiler to schedule more instructions.
                            for (int u = 0; u < UNROLL_K; ++u) {
                                int current_k = kk_inner + u;
                                if (current_k >= K_block_end) {
                                    break; // Handle K-dimension tail (if K_block_end not multiple of UNROLL_K).
                                }

                                // Load A scalars: One scalar from A[ii+r][current_k] for each row of the micro-block.
                                // Then broadcast it to an AVX2 vector.
                                __m256 a_vals[MR_AVX2];
                                for (int r = 0; r < MR_current; ++r) {
                                    a_vals[r] = _mm256_broadcast_ss(A + (ii + r) * lda + current_k);
                                }

                                // Load B vectors: NR_AVX2 vectors from B[current_k][jj + c_vec_idx * VEC_WIDTH].
                                __m256 b_vecs[NR_AVX2];
                                for (int c_vec_idx = 0; c_vec_idx < NR_AVX2; ++c_vec_idx) {
                                    int col_b = jj + c_vec_idx * VEC_WIDTH;
                                    // Handle partial vectors at the N-dimension tail of the matrix.
                                    if (col_b + VEC_WIDTH <= N_block_end) {
                                        // Full vector load.
                                        b_vecs[c_vec_idx] = _mm256_loadu_ps(B + current_k * ldb + col_b);
                                    } else if (col_b < N_block_end) {
                                        // Partial vector load using a temporary aligned buffer.
                                        alignas(32) float temp_buffer[VEC_WIDTH] = {0.0f};
                                        std::memcpy(temp_buffer, B + current_k * ldb + col_b, (N_block_end - col_b) * sizeof(float));
                                        b_vecs[c_vec_idx] = _mm256_load_ps(temp_buffer);
                                    } else {
                                        b_vecs[c_vec_idx] = _mm256_setzero_ps(); // Beyond N_block_end
                                    }
                                }

                                // Accumulate results using Fused Multiply-Add (FMA) instructions.
                                // FMA combines multiply and add into a single instruction, improving throughput and precision.
                                for (int r = 0; r < MR_current; ++r) {
                                    for (int c_vec_idx = 0; c_vec_idx < NR_AVX2; ++c_vec_idx) {
                                        c_acc[r][c_vec_idx] = _mm256_fmadd_ps(a_vals[r], b_vecs[c_vec_idx], c_acc[r][c_vec_idx]);
                                    }
                                }
                            } // End of UNROLL_K loop
                        } // End of inner K-block loop (kk_inner)

                        // Store C accumulators back to the C matrix.
                        for (int r = 0; r < MR_current; ++r) {
                            int row_c = ii + r;
                            for (int c_vec_idx = 0; c_vec_idx < NR_AVX2; ++c_vec_idx) {
                                int col_c = jj + c_vec_idx * VEC_WIDTH;
                                // Handle partial vectors at the N-dimension tail of the matrix.
                                if (col_c + VEC_WIDTH <= N_block_end) {
                                    // Full vector store.
                                    _mm256_storeu_ps(C + row_c * ldc + col_c, c_acc[r][c_vec_idx]);
                                } else if (col_c < N_block_end) {
                                    // Partial vector store using a temporary aligned buffer.
                                    alignas(32) float temp_buffer[VEC_WIDTH];
                                    _mm256_store_ps(temp_buffer, c_acc[r][c_vec_idx]);
                                    std::memcpy(C + row_c * ldc + col_c, temp_buffer, (N_block_end - col_c) * sizeof(float));
                                }
                                // No 'else' needed as col_c >= N_block_end means this vector slot is out of bounds
                                // and should not be stored.
                            }
                        }
                    } // End of jj loop (micro-kernel columns)
                } // End of ii loop (micro-kernel rows)
            } // End of K-block loop (k)
        } // End of N-block loop (n)
    } // End of M-block loop (m)
#else
    // Fallback to scalar GEMM if AVX2 and/or FMA intrinsics are not enabled at compile time.
    std::cerr << "Error: AVX2 and/or FMA intrinsics not enabled. Please compile with -mavx2 -mfma.\n";
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

// --- Main function for CLI parsing and demonstration ---
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]\n";
        return 1;
    }

    // Parse matrix dimensions M, N, K from command line arguments.
    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump_matrices = false;
    if (argc > 4 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    // Use default leading dimensions (lda=K, ldb=N, ldc=N) for dense row-major matrices.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Allocate matrices using std::vector for automatic memory management.
    std::vector<float> A_vec(static_cast<size_t>(M) * lda);
    std::vector<float> B_vec(static_cast<size_t>(K) * ldb);
    std::vector<float> C_vec(static_cast<size_t>(M) * ldc);
    std::vector<float> C_ref_vec; // Only allocated in dump_matrices mode.

    // Initialize matrices A and B with random float values, C with zeros.
    std::mt19937 gen(42); // Mersenne Twister engine, seeded for reproducibility.
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // Random floats between 0.0 and 1.0.

    for (size_t i = 0; i < M * lda; ++i) A_vec[i] = dist(gen);
    for (size_t i = 0; i < K * ldb; ++i) B_vec[i] = dist(gen);
    for (size_t i = 0; i < M * ldc; ++i) C_vec[i] = 0.0f;

    // Get raw pointers for the GEMM functions.
    const float* A = A_vec.data();
    const float* B = B_vec.data();
    float* C = C_vec.data();

    // --- Conditional Logic based on --dump-matrices flag ---
    if (dump_matrices) { // Test Mode: Run both scalar and optimized, dump files, verify correctness.
        std::cout << "--- Test Mode: DUMP MATRICES & VERIFY ---\n";

        // Create 'workspace' directory for output files.
        system("mkdir -p workspace");

        // Dump input matrices A and B to files.
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        // Allocate C_ref for scalar reference result.
        C_ref_vec.resize(static_cast<size_t>(M) * ldc, 0.0f);
        float* C_ref = C_ref_vec.data();

        // Compute reference result using the scalar GEMM.
        std::cout << "Running gemm_scalar for reference...\n";
        auto start_scalar = std::chrono::high_resolution_clock::now();
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        auto end_scalar = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> scalar_duration = end_scalar - start_scalar;
        std::cout << "Scalar GEMM took: " << scalar_duration.count() * 1000 << " ms\n";

        // Compute optimized result using gemm_avx2.
        std::cout << "Running gemm_avx2...\n";
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_duration = end_optimized - start_optimized;
        std::cout << "Optimized GEMM took: " << optimized_duration.count() * 1000 << " ms\n";

        // Dump optimized result C to a file.
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

        // Perform correctness check by comparing optimized C with reference C_ref.
        float max_diff = 0.0f;
        // Epsilon based on typical float precision and accumulation of M*K products.
        float epsilon = 1e-4f * M * K; 
        int errors = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (diff > epsilon) {
                    errors++;
                    // Optional: Print first few errors for debugging.
                    // if (errors < 10) {
                    //     std::cerr << "Mismatch at C[" << i << "][" << j << "]: Optimized=" << C[i * ldc + j]
                    //               << ", Scalar=" << C_ref[i * ldc + j] << ", Diff=" << diff << std::endl;
                    // }
                }
            }
        }

        if (errors == 0) {
            std::cout << "Internal check: PASSED (Max Diff: " << max_diff << ")\n";
        } else {
            std::cout << "Internal check: FAILED (" << errors << " errors, Max Diff: " << max_diff << ")\n";
        }

    } else { // Performance Mode: Only run optimized kernel, no file I/O or verification.
        std::cout << "--- Performance Mode: BENCHMARK gemm_avx2 ---\n";

        // Ensure OpenMP is configured correctly (usually via OMP_NUM_THREADS env var).
#ifdef _OPENMP
        std::cout << "Using " << omp_get_max_threads() << " OpenMP threads.\n";
#endif

        // Run the optimized GEMM and measure its execution time.
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_duration = end_optimized - start_optimized;

        // Calculate and print GFLOPS (Giga Floating Point Operations Per Second).
        // Each multiply-add (A*B+C) counts as 2 FLOPS.
        double gflops = 2.0 * static_cast<double>(M) * N * K / (optimized_duration.count() * 1e9);
        std::cout << "Optimized GEMM took: " << optimized_duration.count() * 1000 << " ms, GFLOPS: " << gflops << "\n";
    }

    return 0;
}