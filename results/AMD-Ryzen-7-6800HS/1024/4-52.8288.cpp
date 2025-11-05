// Example compile command for GCC/Clang:
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm -std=c++17
// or
// clang++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm -std=c++17
//
// To enable AVX2, -mavx2 is crucial. -mfma enables FMA instructions.
// -march=x86-64-v2 implies AVX, AVX2, FMA if available on the CPU, but explicitly adding them is safer.
// -fopenmp is needed for OpenMP parallelization.
// -std=c++17 or later ensures C++17 features are available.

#include <immintrin.h> // For AVX2 intrinsics and _mm_prefetch
#include <iostream>
#include <vector>
#include <cstring>   // For std::memcpy, std::memset
#include <chrono>    // For timing
#include <random>    // For random data generation
#include <cassert>   // For assert
#include <fstream>   // For file operations
#include <string>
#include <iomanip>   // For std::fixed, std::setprecision
#include <algorithm> // For std::min
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
// For AVX2, NR is typically VEC_SIZE. MR=8 means 8 __m256 registers are used as C accumulators.
// This 8x8 micro-kernel is common for AVX2, balancing register pressure and ILP.
constexpr int MR = 8;
constexpr int NR = VEC_SIZE;

// Tiling block sizes
// BM: Block size for M (A rows, C rows)
// BN: Block size for N (B cols, C cols)
// BK: Block size for K (A cols, B rows)
// These are chosen to optimize L1/L2 cache usage.
// Current settings aim to keep sub-blocks of A and packed B in L1D cache.
// A_block (BM x BK): 96 * 64 * 4 bytes = 24.5 KB (fits L1d: 32KB)
// B_packed (BK x BN): 64 * 96 * 4 bytes = 24.5 KB (fits L1d: 32KB, reduced from 128 for better L1d fit)
// C_block (BM x BN): 96 * 96 * 4 bytes = 36.8 KB (fits L2: 512KB)
// Reducing BN from 128 to 96 (a multiple of VEC_SIZE) helps alleviate L1d pressure for B_packed_local,
// potentially reducing L1d cache misses for the B matrix.
constexpr int BM = 96;
constexpr int BN = 96; // Adjusted for better L1D cache fit
constexpr int BK = 64;

// UNROLL_K: Unroll factor for the innermost K loop within the micro-kernel.
// Reverted to 4. Previous iteration showed regression when UNROLL_K was 8,
// suggesting that 8 was too aggressive, likely due to increased register pressure
// or instruction cache/decode issues for the Zen 3 architecture.
// UNROLL_K=4 balances ILP and resource usage.
constexpr int UNROLL_K = 4; // Reverted for better performance based on feedback

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
    // Allocated once per thread on the stack. For current settings (BK=64, BN=96),
    // max size is 64 * 96 * 4 bytes = 24.5 KB. This fits comfortably on the stack and L1d.
    // Alignment to 64 bytes is crucial for AVX operations and cache lines.
    // The `alignas(64)` keyword ensures this buffer is cache-line aligned.
    alignas(64) float B_packed_local[BK * BN];

    // OpenMP parallelization for outer M and N blocks.
    // `collapse(2)` parallelizes both loops, creating tasks for `BM x BN` C blocks.
    // `schedule(static)` ensures a predictable workload distribution (good for load balancing).
    // `firstprivate(B_packed_local)` ensures each thread gets its own copy of the B_packed_local buffer,
    // avoiding data races and false sharing.
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
                // Pack the current `current_K_block_size x current_N_block_size` sub-block of B
                // into contiguous memory (B_packed_local).
                // This improves cache locality for B access within the micro-kernel, as B is accessed
                // column-wise in the original memory layout but row-wise here after packing.
                // The packed B is effectively row-major (`BK` logical rows, each `BN` elements wide).
                // It's important to zero-pad packed B when current_N_block_size < BN for safe vector loads.
                for (int k_idx_pack = 0; k_idx_pack < current_K_block_size; ++k_idx_pack) {
                    const float* B_row_ptr = B + (k_block_start + k_idx_pack) * ldb + n_block_start;
                    float* B_packed_row_ptr = &B_packed_local[k_idx_pack * BN]; // Offset into packed buffer

                    // Copy elements of the B row into B_packed_local
                    std::memcpy(B_packed_row_ptr, B_row_ptr, current_N_block_size * sizeof(float));

                    // Zero-pad if `current_N_block_size < BN`. This is crucial for safe `_mm256_loadu_ps`
                    // when `N` is not a multiple of `BN` or `VEC_SIZE`, preventing garbage reads.
                    // The padding ensures that vector loads within the micro-kernel don't read garbage
                    // beyond the actual N dimension if `BN` is larger than the actual N block size.
                    if (current_N_block_size < BN) {
                        std::memset(B_packed_row_ptr + current_N_block_size, 0, (BN - current_N_block_size) * sizeof(float));
                    }
                }

                // Iterate over rows within the M-block (m_tile_start)
                for (int m_tile_start = m_block_start; m_tile_start < current_m_block_end; m_tile_start += MR) {
                    const int actual_MR = std::min(MR, current_m_block_end - m_tile_start);

                    // Iterate over columns within the N-block (n_tile_start)
                    for (int n_tile_start = n_block_start; n_tile_start < current_n_block_end; n_tile_start += NR) {
                        const int actual_NR = std::min(NR, current_n_block_end - n_tile_start);

                        // --- Micro-kernel ---
                        // Computes an `actual_MR x actual_NR` block of C.

                        // C accumulators: `MR` vector registers, each holding `VEC_SIZE` floats.
                        // For MR=8, this means 8 __m256 registers.
                        __m256 c_acc[MR];
                        for (int i = 0; i < MR; ++i) {
                            c_acc[i] = _mm256_setzero_ps(); // Initialize accumulators to zero
                        }

                        // Prefetch C data that will be written to (for the current MR x NR block).
                        // Hints to load into L1 cache, preventing write-allocate penalties.
                        _mm_prefetch((const char*)(C + (m_tile_start) * ldc + n_tile_start), _MM_HINT_T0);
                        if (actual_MR > 1) { // Prefetch a later row if micro-kernel is taller than 1 row
                            _mm_prefetch((const char*)(C + (m_tile_start + actual_MR - 1) * ldc + n_tile_start), _MM_HINT_T0);
                        }


                        // K loop within the micro-kernel, unrolled by UNROLL_K.
                        for (int k_idx = 0; k_idx < current_K_block_size; k_idx += UNROLL_K) {
                            // Prefetch A and B data for the next UNROLL_K block in K.
                            // These hints aim to bring data into cache just before it's needed,
                            // overlapping memory accesses with computation.
                            // Prefetch for A: next K elements for current MR rows.
                            _mm_prefetch((const char*)(A + (m_tile_start) * lda + (k_block_start + k_idx + UNROLL_K)), _MM_HINT_T0);
                            if (actual_MR > 1) { // Prefetch a later row of A as well
                                _mm_prefetch((const char*)(A + (m_tile_start + actual_MR - 1) * lda + (k_block_start + k_idx + UNROLL_K)), _MM_HINT_T0);
                            }
                            // Prefetch for B: next K rows for current NR columns (from packed buffer).
                            _mm_prefetch((const char*)(B_packed_local + (k_idx + UNROLL_K) * BN + (n_tile_start - n_block_start)), _MM_HINT_T0);

                            // Process UNROLL_K iterations of K.
                            for (int unroll_step = 0; unroll_step < UNROLL_K; ++unroll_step) {
                                int k_current = k_idx + unroll_step;
                                if (k_current >= current_K_block_size) break; // Handle K-tail for unrolling.

                                // Load B vector from packed buffer. `_mm256_loadu_ps` handles unaligned access,
                                // which is fine as B_packed_local is 64-byte aligned, but the offset `(n_tile_start - n_block_start)`
                                // might not be 32-byte aligned if N is not a multiple of VEC_SIZE.
                                // It loads `VEC_SIZE` floats from the packed B row.
                                __m256 b_vec = _mm256_loadu_ps(&B_packed_local[k_current * BN + (n_tile_start - n_block_start)]);

                                // For each row in the MR block of C
                                for (int r = 0; r < actual_MR; ++r) {
                                    // Load A scalar and broadcast it to all elements of an AVX2 vector.
                                    // This is efficient as A is accessed sequentially row-wise within the K-loop.
                                    __m256 a_broadcast = _mm256_broadcast_ss(&A[(m_tile_start + r) * lda + (k_block_start + k_current)]);
                                    // Fused Multiply-Add: c_acc[r] = a_broadcast * b_vec + c_acc[r]
                                    // This is the core computation, performing 8 multiplications and 8 additions in one instruction.
                                    c_acc[r] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r]);
                                }
                            }
                        }

                        // --- Store results back to C ---
                        // Add the accumulated C block (`c_acc`) to the existing C matrix.
                        // This uses `_mm256_loadu_ps`, `_mm256_add_ps`, `_mm256_storeu_ps` for full vectors.
                        // Scalar cleanup is used for `N` dimension tails.
                        for (int r = 0; r < actual_MR; ++r) {
                            float* C_target_row_ptr = C + (m_tile_start + r) * ldc + n_tile_start;
                            // Iterate in steps of VEC_SIZE (8 floats)
                            for (int c_offset = 0; c_offset < actual_NR; c_offset += VEC_SIZE) {
                                if (c_offset + VEC_SIZE <= actual_NR) {
                                    // Process a full vector of 8 floats
                                    __m256 c_existing_vec = _mm256_loadu_ps(C_target_row_ptr + c_offset); // Load existing C values
                                    __m256 c_result_vec = _mm256_add_ps(c_existing_vec, c_acc[r]);       // Add accumulated values
                                    _mm256_storeu_ps(C_target_row_ptr + c_offset, c_result_vec);         // Store back to C
                                } else {
                                    // Handle N-tail (remaining columns that don't form a full vector)
                                    // Access elements directly from the __m256 accumulator by converting it to a float array.
                                    // `tail_c_offset - c_offset` correctly maps the tail elements to indices 0, 1, ...
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
#else // If AVX2 and FMA are not defined, provide a fallback or warning
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    std::cerr << "Warning: gemm_avx2 called but __AVX2__ or __FMA__ not defined. Falling back to scalar." << std::endl;
    // Fallback to the scalar implementation for correctness, though it will be slow.
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

    // Allocate matrices using posix_memalign for 64-byte alignment
    // This is required for optimal AVX performance (e.g., non-unaligned loads)
    // and cache line alignment.
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* C_ref = nullptr; // For scalar reference in dump mode

    // Alignment for AVX instructions is typically 32 bytes for __m256,
    // but 64 bytes is often preferred for cache line alignment on x86-64.
    const size_t alignment = 64;

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

        // Create workspace directory for output files
        // Ignoring return value as the system call might fail if dir exists, but behavior is usually fine.
        (void)std::system("mkdir -p workspace");

        // Dump A and B matrices to files
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B dumped to workspace/A.txt and workspace/B.txt" << std::endl;

        // Compute reference result with scalar GEMM
        std::cout << "Running scalar GEMM..." << std::endl;
        auto start_scalar = std::chrono::high_resolution_clock::now();
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        auto end_scalar = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> scalar_time = end_scalar - start_scalar;
        std::cout << "Scalar GEMM took: " << scalar_time.count() << " seconds." << std::endl;

        // Compute optimized result with AVX2 GEMM
        std::cout << "Running AVX2 GEMM..." << std::endl;
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_time = end_optimized - start_optimized;
        std::cout << "AVX2 GEMM took: " << optimized_time.count() << " seconds." << std::endl;

        // Dump optimized C matrix to file
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Optimized C matrix dumped to workspace/C.txt" << std::endl;

        // Internal correctness check: Compare optimized C with scalar C_ref
        std::cout << "Performing internal correctness check..." << std::endl;
        double max_diff = 0.0;
        double sum_sq_diff = 0.0;
        double epsilon = 1e-4f; // Tolerance for float comparison
        int errors = 0;
        for (int i = 0; i < M * N; ++i) {
            double diff = std::abs(C[i] - C_ref[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            sum_sq_diff += diff * diff;

            // Use relative error check for robustness for values not close to zero
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

        // Compute optimized result with AVX2 GEMM
        std::cout << "Running AVX2 GEMM..." << std::endl;
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_time = end_optimized - start_optimized;

        double gflops = 2.0 * M * N * K / (optimized_time.count() * 1e9);
        std::cout << "AVX2 GEMM took: " << optimized_time.count() << " seconds." << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
    }

    // Free allocated memory
    free(A);
    free(B);
    free(C);
    if (dump_matrices) {
        free(C_ref);
    }

    return 0;
}