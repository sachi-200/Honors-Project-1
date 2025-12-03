// Compile with g++ (or clang++):
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm
// Note: -march=x86-64-v2 implies AVX, AVX2, FMA, but explicitly adding -mavx2 -mfma
// ensures the compiler uses these instruction sets.
// For older GCC/Clang or systems without AVX2:
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm
// (This would compile the scalar version or a less optimized path if __AVX2__ is not defined.)

#include <immintrin.h> // For AVX2 intrinsics
#include <iostream>    // For console output
#include <vector>      // For std::vector
#include <cstring>     // For std::memcpy, std::memset
#include <chrono>      // For timing
#include <random>      // For random number generation
#include <cassert>     // For assert
#include <fstream>     // For file operations
#include <string>      // For std::string
#include <iomanip>     // For std::setprecision, std::fixed
#include <numeric>     // For std::iota (optional, if needed)
#include <filesystem>  // For creating directories (C++17)

#ifdef _OPENMP
#include <omp.h>       // For OpenMP parallelism
#endif

// --- Autotuning Parameters ---
// AVX2 vector width for floats
constexpr int AVX_WIDTH = 8; // __m256 holds 8 floats

// Micro-kernel dimensions (C_MR x C_NR_SIMD*AVX_WIDTH)
// C_MR: number of rows of C processed in the micro-kernel (from A's contribution)
// C_NR_SIMD: number of AVX vectors for C's columns (from B's contribution)
// Total C columns processed in micro-kernel = C_NR_SIMD * AVX_WIDTH
// This creates a C_MR x (C_NR_SIMD * AVX_WIDTH) block of C accumulators held in registers.
// C_MR * C_NR_SIMD = 16, to optimally fit into 16 YMM registers on modern x86-64 CPUs.
constexpr int C_MR = 4;
constexpr int C_NR_SIMD = 4;
constexpr int C_NR = C_NR_SIMD * AVX_WIDTH; // Total actual columns (32 floats)

// K-loop unroll factor within the micro-kernel.
// This processes UNROLL_K steps of the K dimension before updating the next C block.
// Reverted from 6 back to 4. Previous attempts to increase UNROLL_K to 6 or 8
// consistently led to performance regressions, suggesting 4 is currently optimal
// for this micro-kernel structure and target architecture.
constexpr int UNROLL_K = 4;

// Block sizes for cache-aware tiling.
// BM, BN, BK are chosen to promote data reuse in L2/L3 caches.
// These values should be tuned based on specific CPU cache sizes and associativity.
// For AMD Ryzen 7 6800HS, L1=32KB, L2=512KB/core, L3=16MB shared.
// BM * BK (for A block) and BK * BN (for B block) should ideally fit in L2/L3.
// The current settings (128*256*4 bytes = 128KB for A, 256*128*4 bytes = 128KB for B)
// fit well within the 512KB L2 cache per core.
constexpr int BM = 128; // Block size for M (rows of A/C)
constexpr int BN = 128; // Block size for N (columns of B/C)
constexpr int BK = 256; // Block size for K (inner dimension)

// --- Helper for aligned memory allocation (for AVX2 intrinsics) ---
// AVX2 intrinsics like _mm256_load_ps and _mm256_store_ps require 32-byte alignment.
// While _mm256_loadu_ps and _mm256_storeu_ps (unaligned) are used below for robustness
// against arbitrary `ld` values and block starts, aligned allocation is still good practice
// and can provide performance benefits.
constexpr size_t ALIGNMENT = 32; // Defined globally for consistency (32 bytes for AVX2)

void* aligned_malloc(size_t size, size_t alignment) {
    void* ptr = nullptr;
#ifdef _MSC_VER
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}

void aligned_free(void* ptr) {
#ifdef _MSC_VER
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// --- Helper function for matrix dumping to file ---
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    file << std::fixed << std::setprecision(6);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

// --- Reference Scalar GEMM Implementation ---
// Implemented as a simple triple-nested loop (ijk order).
// Assumes row-major storage for A, B, C using lda, ldb, ldc.
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

#ifdef _OPENMP
// Thread-local storage for B_packed buffer. Each OpenMP thread gets its own vector.
// This avoids repeated allocations/deallocations and false sharing between threads.
thread_local std::vector<float> b_packed_thread_local_storage;
#endif

// --- Optimized AVX2+FMA GEMM Implementation ---
// This function implements GEMM (C = A * B + C) using AVX2 intrinsics, FMA,
// OpenMP for parallelization, and cache-aware tiling.
// Assumes row-major storage for A, B, C.
//
// Optimization Strategy:
// 1. **Blocking/Tiling:** Outer loops tile M, N, K dimensions (`BM`, `BN`, `BK`)
//    to maximize data reuse in L2/L3 caches.
// 2. **OpenMP Parallelism:** The outer M and N block loops are parallelized using OpenMP
//    to distribute work across available CPU cores. `schedule(static)` is used,
//    reverting from `schedule(static, 1)` as the latter showed a performance regression.
//    `static` with default chunking provides stable and often optimal load balancing
//    for uniform GEMM workloads.
// 3. **Register Blocking (Micro-kernel):** The innermost loops process a `C_MR` x `C_NR`
//    block of the C matrix. This block is accumulated in `__m256` registers, maximizing
//    register reuse and instruction-level parallelism (ILP). `C_MR * C_NR_SIMD = 16`
//    is chosen to perfectly fit into the 16 YMM registers available on x86-64,
//    minimizing register spilling.
// 4. **Data Packing (B Matrix):** A `BK x BN` block of the B matrix is packed into a
//    contiguous temporary buffer (`B_packed_ptr`) before the micro-kernel. This ensures
//    that B accesses within the micro-kernel are contiguous, improving cache line utilization
//    and reducing memory-bound bottlenecks. Each thread has its own `B_packed` buffer
//    (managed via `thread_local` storage).
// 5. **FMA (Fused Multiply-Add):** `_mm256_fmadd_ps` is used for the core `C += A*B`
//    operation, which performs a multiply and an add in a single instruction, improving
//    throughput and precision.
// 6. **K-Loop Unrolling:** The inner K-loop within the micro-kernel is unrolled by `UNROLL_K=4`
//    to reduce loop overhead and expose ILP. This value has been found to be the most performant
//    after experimenting with higher unroll factors which caused regressions.
// 7. **Prefetching:** `_mm_prefetch` is used strategically to bring future data (next A and B blocks)
//    into cache proactively, aiming to hide memory latency. The prefetching for `A` has been reverted
//    to target `(i + actual_mr - 1) * lda` (last row of current C_MR block), as this strategy
//    previously showed a higher and more effective prefetch percentage.
// 8. **Tail Handling (Correctness):** Robust tail handling for the N-dimension (columns)
//    is implemented using a temporary `temp_tail_buffer`. This buffer, aligned to 32 bytes,
//    is used to correctly load/store partial AVX vectors (when `N` is not a multiple of `AVX_WIDTH`).
//    Valid data is copied into the buffer, the remainder is zero-padded, and then a full
//    AVX vector operation is performed, preventing out-of-bounds memory accesses and ensuring correctness.
//    M and K dimension tails are handled by adjusting loop bounds and `break` statements.
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

#if defined(__AVX2__) && defined(__FMA__) // Ensure AVX2 and FMA are available

    // Temporary buffer for handling tail columns in N dimension (must be 32-byte aligned)
    alignas(ALIGNMENT) float temp_tail_buffer[AVX_WIDTH];

    // OpenMP parallelization over M and N blocks.
    // 'collapse(2)' allows OpenMP to parallelize both loops combined.
    // 'schedule(static)' is used for predictable load balancing and lower overhead.
    // 'firstprivate(temp_tail_buffer)' gives each thread its own copy of the stack buffer.
    // 'default(shared)' means most variables are shared, but `B_packed_ptr` is managed per thread.
    #pragma omp parallel for collapse(2) schedule(static) \
        firstprivate(temp_tail_buffer) default(shared)
    for (int i_block_base = 0; i_block_base < M; i_block_base += BM) {
        for (int j_block_base = 0; j_block_base < N; j_block_base += BN) {
            // Determine actual size of current M and N blocks, handling matrix edges
            const int M_block_end = std::min(i_block_base + BM, M);
            const int N_block_end = std::min(j_block_base + BN, N);
            const int BN_actual = N_block_end - j_block_base; // Actual width of current N block

            // Pointer to the thread-local B_packed buffer.
            // In OpenMP, b_packed_thread_local_storage is a thread_local variable,
            // so each thread has its own instance.
            float* B_packed_ptr;

            #ifdef _OPENMP
                // Ensure thread-local storage is resized to maximum needed size for a BK x BN block
                // (BK * BN floats). It will only reallocate if the maximum size seen changes.
                if (b_packed_thread_local_storage.size() < (size_t)BK * BN) {
                    b_packed_thread_local_storage.resize((size_t)BK * BN);
                }
                B_packed_ptr = b_packed_thread_local_storage.data();
            #else
                // For single-threaded execution, use a static local vector.
                static std::vector<float> b_packed_local_storage;
                if (b_packed_local_storage.size() < (size_t)BK * BN) {
                    b_packed_local_storage.resize((size_t)BK * BN);
                }
                B_packed_ptr = b_packed_local_storage.data();
            #endif

            // Iterate over K dimension in blocks
            for (int k_block_base = 0; k_block_base < K; k_block_base += BK) {
                const int K_block_end = std::min(k_block_base + BK, K);
                const int BK_actual = K_block_end - k_block_base; // Actual height of current K block

                // --- Pack B block into B_packed_ptr ---
                // Copy B[k_block_base:K_block_end, j_block_base:N_block_end] into B_packed_ptr.
                // B_packed_ptr will be BK_actual x BN_actual (row-major within the packed block).
                // This makes accesses to B in the micro-kernel contiguous.
                for (int k_pack_row = 0; k_pack_row < BK_actual; ++k_pack_row) {
                    std::memcpy(B_packed_ptr + k_pack_row * BN_actual,               // Destination in packed buffer
                                &B[(k_block_base + k_pack_row) * ldb + j_block_base], // Source in original B
                                BN_actual * sizeof(float));                          // Size to copy
                }

                // --- Micro-kernel for C_MR x C_NR block ---
                // Iterating over rows of the current C M-block
                for (int i = i_block_base; i < M_block_end; i += C_MR) {
                    const int actual_mr = std::min(C_MR, M_block_end - i);

                    // Iterating over columns of the current C N-block
                    for (int j_current_block_start = j_block_base; j_current_block_start < N_block_end; j_current_block_start += C_NR) {
                        const int current_j_block_width = std::min(C_NR, N_block_end - j_current_block_start); // Actual width of this C_NR chunk
                        const int num_full_vectors = current_j_block_width / AVX_WIDTH;
                        const int tail_cols = current_j_block_width % AVX_WIDTH;

                        // C accumulators for the C_MR x C_NR block, held in YMM registers.
                        __m256 c_acc[C_MR][C_NR_SIMD];

                        // Load initial C values into accumulators for the current C block (C_MR x current_j_block_width)
                        // This step is critical for C = C + A*B accumulation across K-blocks.
                        for (int r = 0; r < actual_mr; ++r) {
                            int c_simd = 0;
                            // Load full vectors of C using unaligned load for flexibility
                            for (; c_simd < num_full_vectors; ++c_simd) {
                                c_acc[r][c_simd] = _mm256_loadu_ps(&C[(i + r) * ldc + j_current_block_start + c_simd * AVX_WIDTH]);
                            }
                            // Handle the tail vector for C, if any (robustly using temp_tail_buffer)
                            if (tail_cols > 0) {
                                // Copy valid C elements into temp_tail_buffer
                                std::memcpy(temp_tail_buffer, &C[(i + r) * ldc + j_current_block_start + c_simd * AVX_WIDTH], tail_cols * sizeof(float));
                                // Zero-pad the rest of the vector to avoid processing garbage values
                                std::memset(temp_tail_buffer + tail_cols, 0, (AVX_WIDTH - tail_cols) * sizeof(float));
                                c_acc[r][c_simd] = _mm256_load_ps(temp_tail_buffer); // Load from aligned temp buffer
                                c_simd++; // Increment to account for the tail vector
                            }
                            // Zero out any remaining accumulators that are declared but not used in this current_j_block_width
                            for (; c_simd < C_NR_SIMD; ++c_simd) {
                                c_acc[r][c_simd] = _mm256_setzero_ps();
                            }
                        }
                        // Clear unused accumulators for safety in case actual_mr < C_MR
                        for (int r = actual_mr; r < C_MR; ++r) {
                            for (int c_simd = 0; c_simd < C_NR_SIMD; ++c_simd) {
                                c_acc[r][c_simd] = _mm256_setzero_ps();
                            }
                        }

                        // Inner K loop: accumulate products of A and B into C_acc
                        for (int k_local = 0; k_local < BK_actual; k_local += UNROLL_K) {
                            // Prefetching: Hint to the CPU to load data into cache for future use.
                            // Prefetch `A` values for the next `UNROLL_K` steps for the last row of the current `C_MR` block.
                            // Reverted A prefetch to the previously effective strategy, which showed better prefetch percentage.
                            _mm_prefetch((const char*)&A[(i + actual_mr - 1) * lda + k_block_base + k_local + UNROLL_K], _MM_HINT_T0);
                            // Prefetch `B_packed` values for the next `UNROLL_K` rows.
                            _mm_prefetch((const char*)&B_packed_ptr[(k_local + UNROLL_K) * BN_actual], _MM_HINT_T0);

                            // Unroll the K loop to process multiple K steps per micro-kernel iteration
                            for (int k_unroll = 0; k_unroll < UNROLL_K; ++k_unroll) {
                                int current_k_local = k_local + k_unroll;
                                if (current_k_local >= BK_actual) break; // K-tail handling for unrolled loop

                                // For each row of A/C in the micro-kernel (MR rows)
                                for (int r = 0; r < actual_mr; ++r) {
                                    // Load scalar A value: A[current_row_idx][current_k]
                                    // A is row-major: A[row * lda + col]
                                    // Broadcast the scalar A value across an entire AVX2 vector.
                                    __m256 a_broadcast = _mm256_broadcast_ss(&A[(i + r) * lda + k_block_base + current_k_local]);

                                    // For each column vector of B/C in the micro-kernel (NR_SIMD vectors)
                                    int c_simd = 0;
                                    // Process full vectors for B from B_packed
                                    for (; c_simd < num_full_vectors; ++c_simd) {
                                        // Load B vector from the packed buffer.
                                        // The column offset `(j_current_block_start - j_block_base)` adjusts for the start of the C_NR block within BN_actual.
                                        __m256 b_vec = _mm256_loadu_ps(&B_packed_ptr[current_k_local * BN_actual + (j_current_block_start - j_block_base) + c_simd * AVX_WIDTH]);

                                        // Fused multiply-add: C_acc = (a_broadcast * b_vec) + C_acc
                                        c_acc[r][c_simd] = _mm256_fmadd_ps(a_broadcast, b_vec, c_acc[r][c_simd]);
                                    }
                                    // Handle the tail vector for B from B_packed, if any (robustly using temp_tail_buffer)
                                    if (tail_cols > 0) {
                                        // Copy valid B elements into temp_tail_buffer, zero-pad
                                        std::memcpy(temp_tail_buffer, &B_packed_ptr[current_k_local * BN_actual + (j_current_block_start - j_block_base) + c_simd * AVX_WIDTH], tail_cols * sizeof(float));
                                        std::memset(temp_tail_buffer + tail_cols, 0, (AVX_WIDTH - tail_cols) * sizeof(float));
                                        __m256 b_vec_tail = _mm256_load_ps(temp_tail_buffer); // Load from aligned temp buffer
                                        c_acc[r][c_simd] = _mm256_fmadd_ps(a_broadcast, b_vec_tail, c_acc[r][c_simd]);
                                    }
                                }
                            }
                        }

                        // Store accumulated results from registers back to C matrix
                        for (int r = 0; r < actual_mr; ++r) {
                            int c_simd = 0;
                            // Store full vectors using unaligned store for flexibility
                            for (; c_simd < num_full_vectors; ++c_simd) {
                                _mm256_storeu_ps(&C[(i + r) * ldc + j_current_block_start + c_simd * AVX_WIDTH], c_acc[r][c_simd]);
                            }
                            // Store tail vector for C, if any (robustly using temp_tail_buffer)
                            if (tail_cols > 0) {
                                _mm256_store_ps(temp_tail_buffer, c_acc[r][c_simd]); // Store to aligned temp buffer
                                // Copy only the relevant part back to C
                                std::memcpy(&C[(i + r) * ldc + j_current_block_start + c_simd * AVX_WIDTH], temp_tail_buffer, tail_cols * sizeof(float));
                            }
                        }
                    }
                }
            }
        }
    }
#else
    // Fallback if AVX2 or FMA is not available at compile time.
    // This provides a gracefully degrading behavior and ensures the code compiles
    // even on systems without the target ISA, though performance will be poor.
    std::cerr << "Warning: gemm_avx2 called but AVX2/FMA intrinsics not enabled."
              << " Falling back to scalar implementation." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}


// --- Main function for CLI parsing and demonstration ---
int main(int argc, char* argv[]) {
    // Basic CLI argument parsing
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    int M = std::stoi(argv[1]);
    int N = std::stoi(argv[2]);
    int K = std::stoi(argv[3]);
    bool dump_matrices = false;

    if (argc > 4 && std::string(argv[4]) == "--dump-matrices") {
        dump_matrices = true;
    }

    // Define leading dimensions. For simplicity, assuming dense row-major allocation
    // where leading dimension matches the logical dimension.
    int lda = K; // For A (M x K matrix), lda is K
    int ldb = N; // For B (K x N matrix), ldb is N
    int ldc = N; // For C (M x N matrix), ldc is N

    // Allocate matrices using aligned_malloc
    float* A = (float*)aligned_malloc(M * lda * sizeof(float), ALIGNMENT);
    float* B = (float*)aligned_malloc(K * ldb * sizeof(float), ALIGNMENT);
    float* C = (float*)aligned_malloc(M * ldc * sizeof(float), ALIGNMENT);
    float* C_ref = nullptr; // Allocated only if --dump-matrices is present

    if (!A || !B || !C) {
        std::cerr << "Error: Failed to allocate matrices." << std::endl;
        aligned_free(A); aligned_free(B); aligned_free(C); // Free any successfully allocated memory
        return 1;
    }

    if (dump_matrices) {
        C_ref = (float*)aligned_malloc(M * ldc * sizeof(float), ALIGNMENT);
        if (!C_ref) {
            std::cerr << "Error: Failed to allocate C_ref." << std::endl;
            aligned_free(A); aligned_free(B); aligned_free(C);
            return 1;
        }
    }

    // Initialize matrices with random data and C/C_ref with zeros
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f); // Values between -1.0 and 1.0

    for (int i = 0; i < M * lda; ++i) A[i] = dis(gen);
    for (int i = 0; i < K * ldb; ++i) B[i] = dis(gen);
    std::memset(C, 0, M * ldc * sizeof(float)); // Initialize C to zeros
    if (dump_matrices) {
        std::memset(C_ref, 0, M * ldc * sizeof(float)); // Initialize C_ref to zeros
    }

    if (dump_matrices) {
        // --- Test Mode: Dump matrices, compute reference, compute optimized, verify ---
        std::cout << "Running in Test Mode (--dump-matrices specified)" << std::endl;
        std::filesystem::create_directory("workspace"); // Ensure workspace directory exists

        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);

        std::cout << "Computing reference GEMM (scalar) for correctness check..." << std::endl;
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

        std::cout << "Computing optimized GEMM (AVX2)..." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);

        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);

        // Verify correctness by comparing optimized C with reference C_ref
        float max_diff = 0.0f;
        float epsilon = 1e-4f; // Tolerance for floating-point comparisons
        bool passed = true;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
                if (diff > epsilon) {
                    passed = false;
                    // For debugging, uncomment to see first few discrepancies:
                    // std::cerr << "Mismatch at C[" << i << "][" << j << "]: "
                    //           << "Optimized=" << C[i * ldc + j] << ", Reference=" << C_ref[i * ldc + j]
                    //           << ", Diff=" << diff << std::endl;
                }
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        if (passed) {
            std::cout << "Internal check: PASSED. Max difference: " << max_diff << std::endl;
        } else {
            std::cout << "Internal check: FAILED. Max difference: " << max_diff << std::endl;
        }

    } else {
        // --- Performance Mode: Only run optimized GEMM and optionally time it ---
        std::cout << "Running in Perf Mode (no --dump-matrices)" << std::endl;
        std::cout << "Computing optimized GEMM (AVX2) for M=" << M << ", N=" << N << ", K=" << K << "..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end = std::chrono::high_resolution_clock::now();

        double duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // Calculate GFLOPS: 2 * M * N * K floating-point operations
        // (one multiply and one add per element for each K iteration)
        double gflops = (2.0 * M * N * K) / (duration_ms * 1e6);

        std::cout << "Execution time: " << duration_ms << " ms" << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;
    }

    // Cleanup allocated memory
    aligned_free(A);
    aligned_free(B);
    aligned_free(C);
    if (C_ref) {
        aligned_free(C_ref);
    }

    return 0;
}