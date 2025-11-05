// Example compile command for GCC/Clang:
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm -std=c++17
// or
// clang++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm -std=c++17
//
// To enable AVX2, -mavx2 is crucial. -mfma enables FMA instructions.
// -march=x86-64-v2 implies AVX, AVX2, FMA if available on the CPU, but explicitly adding them is safer.
// -fopenmp is needed for OpenMP parallelization.
// -std=c++17 or later ensures C++17 features are available.

#include <immintrin.h> // For AVX2 intrinsics
#include <iostream>
#include <vector>
#include <cstring>   // For std::memcpy, std::memset
#include <chrono>    // For timing
#include <random>    // For random data generation
#include <cassert>   // For assert
#include <fstream>   // For file operations
#include <string>
#include <iomanip>   // For std::fixed, std::setprecision
#include <algorithm> // For std::min, std::max
#include <cmath>     // For std::abs, std::sqrt
#include <cstdlib>   // For posix_memalign, free, system
#include <cfloat>    // For FLT_MIN (for error checking)

#if defined(_OPENMP)
#include <omp.h> // For OpenMP
#endif

// Define default constants for tiling and unrolling
// These are tunable parameters for performance optimization.
// Target CPU: AMD Ryzen 7 6800HS (Zen 3/3+)
// L1d cache: 32KB, L2 cache: 512KB, L3 cache: 16MB.
// float is 4 bytes.
// VEC_SIZE for AVX2 is 8 floats (32 bytes).

// Micro-kernel dimensions (fixed for AVX2)
constexpr int VEC_SIZE = 8; // floats per AVX2 vector (e.g., __m256 holds 8 floats)

// MR: Rows of C to compute in the micro-kernel (register blocking for A).
// NR: Columns of C to compute in the micro-kernel (vector width for B and C).
// For AVX2, NR is typically VEC_SIZE (8 floats).
// MR=8 is chosen based on consistent performance feedback for Zen 3 (Ryzen 7 6800HS),
// which has 16 AVX2 registers. Using 8 __m256 accumulators for C (c_acc[MR]),
// plus 1 for `a_broadcast` and 1 for `b_vec`, totals 10 registers. This configuration
// strikes a good balance for instruction-level parallelism (ILP) and register pressure,
// helping to avoid register spills and enabling efficient instruction scheduling.
constexpr int MR = 8;
constexpr int NR = VEC_SIZE;

// Tiling block sizes
// BM: Block size for M (A rows, C rows)
// BN: Block size for N (B cols, C cols)
// BK: Block size for K (A cols, B rows)
// These are chosen to optimize L1/L2 cache usage based on the Zen 3 cache hierarchy.
// A_block (BM x BK): 96 * 64 * 4 bytes = 24.5 KB (fits L1d: 32KB)
// B_packed (BK x BN): 64 * 96 * 4 bytes = 24.5 KB (fits L1d: 32KB). This size was found to
// offer a good balance for L1D cache utilization without causing too many conflicts.
// C_block (BM x BN): 96 * 96 * 4 bytes = 36.8 KB (fits L2: 512KB).
// The goal is to keep these working sets within L1 and L2 caches to reduce memory latency.
// These block sizes have shown good performance and cache utilization for the target CPU.
constexpr int BM = 96;
constexpr int BN = 96;
constexpr int BK = 64;

// UNROLL_K: Unroll factor for the innermost K loop within the micro-kernel.
// Increased from 12 to 16 in this iteration.
// The workload is still identified as Compute Bound. Further increasing UNROLL_K exposes more
// Instruction-Level Parallelism (ILP) to the CPU's out-of-order execution engine. This can help
// keep the FMA units more saturated by providing a larger window of independent operations,
// reducing loop overhead, and potentially improving the average number of FMA instructions
// issued per cycle on Zen 3. This leverages the strong out-of-order capabilities of the target CPU.
constexpr int UNROLL_K = 16;

// Helper function to write a matrix to a file
// Matrices are assumed to be stored in row-major format.
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    ofs << std::fixed << std::setprecision(6); // Format floating point numbers for consistent output

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << "\n";
    }
    ofs.close();
}

// Reference scalar GEMM implementation (row-major)
// This function serves as a correctness reference and is single-threaded.
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

// AVX2 optimized GEMM implementation (row-major)
// Uses tiling, OpenMP for parallelization, and AVX2+FMA intrinsics for vectorization.
#if defined(__AVX2__) && defined(__FMA__)
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

    // B_packed_local: Buffer for packing sub-blocks of B.
    // Allocated once per thread on the stack with 64-byte alignment.
    // Packing B improves cache locality during the micro-kernel, as B elements are
    // accessed contiguously after packing, turning strided column-wise accesses into
    // cache-friendly row-wise accesses.
    // The size BK * BN is the maximum size for a packed B block.
    alignas(64) float B_packed_local[BK * BN];

    // OpenMP parallelization strategy:
    // `collapse(2)` parallelizes both the `m_block_start` and `n_block_start` loops,
    // distributing `BM x BN` C blocks across available threads.
    // `schedule(static)` ensures a predictable, even workload distribution, which is
    // generally good for dense matrix operations.
    // `firstprivate(B_packed_local)` provides each thread with its own private copy
    // of the packed B buffer, preventing data races and false sharing.
    #pragma omp parallel for collapse(2) schedule(static) firstprivate(B_packed_local)
    for (int m_block_start = 0; m_block_start < M; m_block_start += BM) {
        for (int n_block_start = 0; n_block_start < N; n_block_start += BN) {

            // Determine actual dimensions for the current M and N blocks
            const int current_m_block_end = std::min(m_block_start + BM, M);
            const int current_n_block_end = std::min(n_block_start + BN, N);
            const int current_N_block_size = current_n_block_end - n_block_start;

            // Inner loop for K blocks
            for (int k_block_start = 0; k_block_start < K; k_block_start += BK) {

                const int current_k_block_end = std::min(k_block_start + BK, K);
                const int current_K_block_size = current_k_block_end - k_block_start;

                // --- Packing B_block ---
                // Pack the `current_K_block_size x current_N_block_size` sub-block of B
                // from its original row-major layout into the `B_packed_local` buffer.
                // This reorders B elements for optimal access within the micro-kernel.
                // It also zero-pads rows if current_N_block_size < BN.
                for (int k_idx_pack = 0; k_idx_pack < current_K_block_size; ++k_idx_pack) {
                    const float* B_row_ptr = B + (k_block_start + k_idx_pack) * ldb + n_block_start;
                    float* B_packed_row_ptr = &B_packed_local[k_idx_pack * BN];

                    // Copy actual elements
                    std::memcpy(B_packed_row_ptr, B_row_ptr, current_N_block_size * sizeof(float));

                    // Zero-pad the remainder of the packed row if `current_N_block_size < BN`.
                    // This is crucial for correctness when using `_mm256_loadu_ps` in the micro-kernel,
                    // as it prevents reading uninitialized memory when `N` is not a multiple of `VEC_SIZE`
                    // or `BN`. The multiplication by these zero-padded values will correctly result in zero.
                    if (current_N_block_size < BN) {
                        std::memset(B_packed_row_ptr + current_N_block_size, 0, (BN - current_N_block_size) * sizeof(float));
                    }
                }
                // If current_K_block_size < BK, the rest of B_packed_local for k_idx_pack >= current_K_block_size
                // remains uninitialized or retains previous values. This is handled by the `k_current >= current_K_block_size` check.

                // Iterate over rows within the M-block (m_tile_start)
                for (int m_tile_start = m_block_start; m_tile_start < current_m_block_end; m_tile_start += MR) {
                    const int actual_MR = std::min(MR, current_m_block_end - m_tile_start);

                    // Iterate over columns within the N-block (n_tile_start)
                    for (int n_tile_start = n_block_start; n_tile_start < current_n_block_end; n_tile_start += NR) {
                        const int actual_NR = std::min(NR, current_n_block_end - n_tile_start);

                        // --- Micro-kernel ---
                        // Computes an `actual_MR x actual_NR` block of C, accumulating into `c_acc` registers.

                        // C accumulators: `MR` vector registers, each holding `VEC_SIZE` floats.
                        // For MR=8, this utilizes 8 __m256 registers for C accumulation.
                        __m256 c_acc[MR];
                        for (int i = 0; i < MR; ++i) {
                            c_acc[i] = _mm256_setzero_ps(); // Initialize accumulators to zero
                        }

                        // Prefetch C data. _MM_HINT_T0 brings data into L1 cache.
                        // This helps avoid write-allocate penalties and ensures C data is ready for final accumulation.
                        _mm_prefetch((const char*)(C + (m_tile_start) * ldc + n_tile_start), _MM_HINT_T0);
                        if (actual_MR > 1) { // If micro-kernel height is more than one row, prefetch a later row too.
                            _mm_prefetch((const char*)(C + (m_tile_start + actual_MR - 1) * ldc + n_tile_start), _MM_HINT_T0);
                        }

                        // K loop within the micro-kernel, unrolled by UNROLL_K.
                        // For Zen 3 (AMD Ryzen 7 6800HS), explicit prefetching for A and B in the inner
                        // K-loop can sometimes interfere with effective hardware prefetchers or add instruction
                        // overhead. Given the kernel is Compute Bound, relying on the CPU's sophisticated
                        // hardware prefetchers for A and B (after initial block loading) tends to yield
                        // better results by simplifying the instruction stream and avoiding potential
                        // prefetch misspeculation.
                        for (int k_idx = 0; k_idx < current_K_block_size; k_idx += UNROLL_K) {

                            // Process UNROLL_K iterations of K.
                            // Loop body is unrolled to expose more ILP to the CPU.
                            // Each `unroll_step` calculates a partial sum for `MR` rows of C.
                            for (int unroll_step = 0; unroll_step < UNROLL_K; ++unroll_step) {
                                int k_current = k_idx + unroll_step;
                                // Handle K-tail for unrolling. This check ensures we don't access
                                // B_packed_local beyond the valid `current_K_block_size`.
                                if (k_current >= current_K_block_size) break;

                                // Load B vector from packed buffer. `_mm256_loadu_ps` handles potentially unaligned access.
                                // It loads `VEC_SIZE` floats (one row of the packed B) into a vector register.
                                __m256 b_vec = _mm256_loadu_ps(&B_packed_local[k_current * BN + (n_tile_start - n_block_start)]);

                                // For each row (r) in the MR block of C:
                                for (int r = 0; r < actual_MR; ++r) {
                                    // Load A scalar element and broadcast it to all elements of an AVX2 vector.
                                    // A is accessed sequentially (stride 1) within the K-loop. This broadcast operation is efficient.
                                    __m256 a_broadcast = _mm256_broadcast_ss(&A[(m_tile_start + r) * lda + (k_block_start + k_current)]);
                                    // Fused Multiply-Add: c_acc[r] = a_broadcast * b_vec + c_acc[r]
                                    // This is the core computation (8 multiplications and 8 additions in one FMA instruction).
                                    c_acc[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // --- Store results back to C ---
                        // Add the accumulated C block (`c_acc`) to the existing C matrix.
                        // Vector loads (`_mm256_loadu_ps`), additions (`_mm256_add_ps`), and stores (`_mm256_storeu_ps`)
                        // are used for full vector widths. Scalar cleanup handles `N` dimension tails.
                        for (int r = 0; r < actual_MR; ++r) {
                            float* C_target_row_ptr = C + (m_tile_start + r) * ldc + n_tile_start;
                            // Iterate in steps of VEC_SIZE (8 floats) for vector operations.
                            for (int c_offset = 0; c_offset < actual_NR; c_offset += VEC_SIZE) {
                                if (c_offset + VEC_SIZE <= actual_NR) {
                                    // Process a full vector of 8 floats.
                                    __m256 c_existing_vec = _mm256_loadu_ps(C_target_row_ptr + c_offset); // Load existing C values.
                                    __m256 c_result_vec = _mm256_add_ps(c_existing_vec, c_acc[r]);       // Add accumulated values from register.
                                    _mm256_storeu_ps(C_target_row_ptr + c_offset, c_result_vec);         // Store back to C.
                                } else {
                                    // Handle N-tail: remaining columns that do not form a full vector.
                                    // Access elements directly from the `__m256` accumulator by casting it to a `float` array.
                                    // `tail_c_offset - c_offset` correctly maps to the corresponding index within the `__m256` register.
                                    for (int tail_c_offset = c_offset; tail_c_offset < actual_NR; ++tail_c_offset) {
                                        C_target_row_ptr[tail_c_offset] += ((float*)&c_acc[r])[tail_c_offset - c_offset];
                                    }
                                }
                            }
                        }
                    } // End n_tile_start loop (over N blocks within BN)
                } // End m_tile_start loop (over M blocks within BM)
            } // End k_block_start loop (over K blocks within BK)
        } // End n_block_start loop (OpenMP parallel over N blocks)
    } // End m_block_start loop (OpenMP parallel over M blocks)
}
#else // If AVX2 and FMA are not defined, provide a fallback or warning.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "Warning: gemm_avx2 called but __AVX2__ or __FMA__ not defined. Falling back to scalar." << std::endl;
    // Fallback to the scalar implementation for correctness, though it will be significantly slower.
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}
#endif // __AVX2__ && __FMA__


int main(int argc, char* argv[]) {
    // Parse command line arguments
    int M = 0, N = 0, K = 0;
    bool dump_matrices = false;

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    try {
        M = std::stoi(argv[1]);
        N = std::stoi(argv[2]);
        K = std::stoi(argv[3]);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid argument: M, N, K must be integers." << std::endl;
        return 1;
    } catch (const std::out_of_range& e) {
        std::cerr << "Out of range argument: M, N, K are too large." << std::endl;
        return 1;
    }

    if (argc > 4 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    // Ensure dimensions are positive
    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "M, N, K must be positive integers." << std::endl;
        return 1;
    }

    // Define leading dimensions (assuming row-major storage for all matrices)
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Allocate matrices using posix_memalign for 64-byte alignment.
    // This alignment is critical for optimal AVX performance and cache line efficiency.
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* C_ref = nullptr; // For scalar reference in dump mode

    const size_t alignment = 64; // 64-byte alignment for cache lines and AVX.

    if (posix_memalign((void**)&A, alignment, (size_t)M * lda * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory for A." << std::endl; return 1;
    }
    if (posix_memalign((void**)&B, alignment, (size_t)K * ldb * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory for B." << std::endl; free(A); return 1;
    }
    if (posix_memalign((void**)&C, alignment, (size_t)M * ldc * sizeof(float)) != 0) {
        std::cerr << "Failed to allocate aligned memory for C." << std::endl; free(A); free(B); return 1;
    }

    if (dump_matrices) {
        if (posix_memalign((void**)&C_ref, alignment, (size_t)M * ldc * sizeof(float)) != 0) {
            std::cerr << "Failed to allocate aligned memory for C_ref." << std::endl; free(A); free(B); free(C); return 1;
        }
    }

    // Initialize matrices with random values, C and C_ref to zero.
    std::mt19937 gen(0); // Use a fixed seed for reproducibility across runs
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < (size_t)M * lda; ++i) A[i] = dist(gen);
    for (size_t i = 0; i < (size_t)K * ldb; ++i) B[i] = dist(gen);
    for (size_t i = 0; i < (size_t)M * ldc; ++i) C[i] = 0.0f; // C must be initialized to 0 for C += A*B
    if (dump_matrices) {
        for (size_t i = 0; i < (size_t)M * ldc; ++i) C_ref[i] = 0.0f;
    }

    // --- Main Logic: Test Mode or Performance Mode ---
    if (dump_matrices) {
        // Test Mode: Compute both scalar and optimized, dump matrices, and check correctness.
        std::cout << "--- Test Mode (M=" << M << ", N=" << N << ", K=" << K << ") ---" << std::endl;

        // Create workspace directory for output files.
        (void)std::system("mkdir -p workspace"); // Ignoring return value, directory might already exist.

        // Dump A and B matrices to files.
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt" << std::endl;

        // Compute reference result with scalar GEMM.
        std::cout << "Running scalar GEMM..." << std::endl;
        auto start_scalar = std::chrono::high_resolution_clock::now();
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        auto end_scalar = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> scalar_time = end_scalar - start_scalar;
        std::cout << "Scalar GEMM took: " << scalar_time.count() << " seconds." << std::endl;

        // Compute optimized result with AVX2 GEMM.
        std::cout << "Running AVX2 GEMM..." << std::endl;
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_time = end_optimized - start_optimized;
        std::cout << "AVX2 GEMM took: " << optimized_time.count() << " seconds." << std::endl;

        // Dump optimized C matrix to file.
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Optimized C matrix dumped to workspace/C.txt" << std::endl;

        // Internal correctness check: Compare optimized C with scalar C_ref.
        std::cout << "Performing internal correctness check..." << std::endl;
        double max_diff = 0.0;
        double sum_sq_diff = 0.0;
        double epsilon = 1e-4f; // Tolerance for float comparison.
        int errors = 0;
        for (int i = 0; i < M * N; ++i) {
            double diff = std::abs(C[i] - C_ref[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            sum_sq_diff += diff * diff;

            // Use relative error check for robustness for values not close to zero.
            // Add FLT_MIN to divisor to prevent division by zero if both C[i] and C_ref[i] are near zero.
            if (diff > epsilon && diff / (std::max(std::abs(C[i]), std::abs(C_ref[i])) + FLT_MIN) > epsilon) {
                errors++;
            }
        }
        double rms_error = std::sqrt(sum_sq_diff / (M * N));

        if (errors == 0 && max_diff < epsilon) {
            std::cout << "Internal check: PASSED (Max difference: " << max_diff << ", RMS error: " << rms_error << ")" << std::endl;
        } else {
            std::cout << "Internal check: FAILED (" << errors << " errors found, Max difference: " << max_diff << ", RMS error: " << rms_error << ")" << std::endl;
        }

    } else {
        // Performance Mode: Only compute optimized GEMM. No file I/O, no reference check.
        std::cout << "--- Performance Mode (M=" << M << ", N=" << N << ", K=" << K << ") ---" << std::endl;

        // Compute optimized result with AVX2 GEMM.
        std::cout << "Running AVX2 GEMM..." << std::endl;
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_time = end_optimized - start_optimized;

        double gflops = 2.0 * M * N * K / (optimized_time.count() * 1e9);
        std::cout << "AVX2 GEMM took: " << optimized_time.count() << " seconds." << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
    }

    // Free allocated memory.
    free(A);
    free(B);
    free(C);
    if (dump_matrices) {
        free(C_ref);
    }

    return 0;
}