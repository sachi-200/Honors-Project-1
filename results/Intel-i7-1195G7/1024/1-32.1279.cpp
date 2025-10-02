#ifndef _GNU_SOURCE
#define _GNU_SOURCE // Required for __builtin_cpu_supports
#endif

#include <iostream>
#include <vector>
#include <cstring> // For memset
#include <chrono>
#include <random>
#include <cassert>
#include <immintrin.h> // For SIMD intrinsics
#include <cmath> // For std::fabs
#include <algorithm> // For std::min
#include <string>
#include <fstream>
#include <filesystem>
#include <tuple> // For autotuning parameter tuples

#ifdef _OPENMP
#include <omp.h>
#else
// Define dummy OpenMP functions if OpenMP is not available
int omp_get_max_threads() { return 1; }
int omp_get_thread_num() { return 0; }
#endif

// --- Compile Instructions ---
// For AVX-512 (Intel 11th Gen supports this):
// g++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
// clang++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//
// For AVX2 fallback (e.g., older Intel/AMD or if AVX-512 is not desired):
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
// clang++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// For portable compilation (uses -march=native for best performance on current CPU, might error if no SIMD):
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native
// clang++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native
//
// Using -march=x86-64-v3 implies AVX2 and FMA, but -mavx512f explicitly enables AVX-512.
// -march=x86-64-v2 implies AVX, AVX2, FMA.

// --- Tunable Parameters (Autotuning) ---
// These are default values. The autotuner can suggest better ones.
// Blocking factors for M, N, K dimensions.
// Aim: Keep active data (A_block, B_block, C_block) in L1/L2 cache.
// A_block (BM x BK), B_block (BK x BN), C_block (BM x BN)
// Example: BM=128, BN=128, BK=96 -> A: 128*96*4B = 49KB, B: 96*128*4B = 49KB, C: 128*128*4B = 65KB
// L1D is 48KB on i7-1195G7. L2 is 1.25MB per core. These blocks should fit well within L2.
//
// For the inner micro-kernel, we will use register blocking:
// AVX-512: MR_ACC = 6, NR_ACC = 16 (for 6x16 block of C using 6 ZMM registers)
// AVX2:    MR_ACC = 6, NR_ACC = 8  (for 6x8 block of C using 6 YMM registers)
//
// These are not `constexpr` because they can be changed by the autotuner (if we were to dynamically apply them)
// For this problem, they are `#define` to act as compile-time constants as requested.
// The autotuner will output suggestions for these values.
#define DEFAULT_BM 128
#define DEFAULT_BN 128
#define DEFAULT_BK 96
#define DEFAULT_UNROLL_K 4 // Unroll factor for the innermost K loop within the micro-kernel.

// Global variables for tunable parameters, allowing them to be set by CLI/autotuner
// Default values will be used if not overridden.
int BM = DEFAULT_BM;
int BN = DEFAULT_BN;
int BK = DEFAULT_BK;
int UNROLL_K = DEFAULT_UNROLL_K;

// --- Function Declarations (CRITICAL: exact signatures) ---
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc);

void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc);

void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc);

void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc);

// --- Helper for matrix I/O ---
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        file << "\n";
    }
    file.close();
}

// --- Scalar Reference Implementation ---
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

// --- Micro-Kernel for AVX-512 ---
// Computes C_block += A_block * B_block using AVX-512 intrinsics and register blocking.
// A_block is (m_block_size x k_block_size), B_block is (k_block_size x n_block_size), C_block is (m_block_size x n_block_size)
// All pointers are already offset to the start of their respective blocks.
// lda, ldb, ldc are the original full matrix leading dimensions.
// is_first_k_block: true if this is the first K-block contributing to the current C_block.
void micro_kernel_avx512(const float* A_ptr, const float* B_ptr, float* C_ptr,
                         int m_block_size, int n_block_size, int k_block_size,
                         int lda, int ldb, int ldc, bool is_first_k_block) {
#if defined(__AVX512F__) && defined(__FMA__)
    // Register blocking constants
    constexpr int MR_ACC = 6;  // Rows of C to compute concurrently
    constexpr int NR_ACC = 16; // Columns of C to compute concurrently (AVX-512 vector width)
    constexpr int SIMD_WIDTH = 16; // float elements in __m512

    // Loop over C rows in MR_ACC chunks
    for (int mr_idx = 0; mr_idx < m_block_size; mr_idx += MR_ACC) {
        int mr_len = std::min(MR_ACC, m_block_size - mr_idx);

        // Loop over C columns in NR_ACC chunks
        for (int nr_idx = 0; nr_idx < n_block_size; nr_idx += NR_ACC) {
            int nr_len = std::min(NR_ACC, n_block_size - nr_idx);

            // C accumulators for the MR_ACC x NR_ACC block
            __m512 c_acc[MR_ACC];

            // Initialize accumulators: zero for first K-block, load from C otherwise
            if (is_first_k_block) {
                for (int i = 0; i < MR_ACC; ++i) {
                    c_acc[i] = _mm512_setzero_ps();
                }
            } else {
                for (int i = 0; i < mr_len; ++i) {
                    // Load existing C values into accumulators
                    // C_ptr + (mr_idx + i) * ldc + nr_idx points to the start of current C row in the block
                    __mmask16 n_mask = (__mmask16)((1 << nr_len) - 1); // Mask for N-tail
                    c_acc[i] = _mm512_maskz_loadu_ps(n_mask, &C_ptr[(mr_idx + i) * ldc + nr_idx]);
                }
                for (int i = mr_len; i < MR_ACC; ++i) { // Zero out unused accumulators if mr_len < MR_ACC
                    c_acc[i] = _mm512_setzero_ps();
                }
            }

            // K loop for inner product computation within the micro-kernel
            // This loop iterates K_block_size times, accumulating into C_acc.
            for (int k_s = 0; k_s < k_block_size; ++k_s) {
                // Prefetch B's next column (or part of it)
                _mm_prefetch((const char*)&B_ptr[k_s * ldb + (nr_idx + NR_ACC)], _MM_HINT_T0);
                // Prefetch A's next row element
                _mm_prefetch((const char*)&A_ptr[(mr_idx + MR_ACC) * lda + k_s], _MM_HINT_T0);

                // Load B_vec: NR_ACC elements from B, for current k_s and nr_idx columns.
                // B_ptr + k_s * ldb + nr_idx points to current B row in the block.
                __mmask16 n_mask = (__mmask16)((1 << nr_len) - 1); // Mask for N-tail
                __m512 b_vec = _mm512_maskz_loadu_ps(n_mask, &B_ptr[k_s * ldb + nr_idx]);

                // For each of MR_ACC rows of A:
                for (int m_i = 0; m_i < mr_len; ++m_i) {
                    // Load A_scalar: one element from A, for current m_r + m_i row and k_s column.
                    // A_ptr + (mr_idx + m_i) * lda + k_s points to current A element in the block.
                    float a_scalar = A_ptr[(mr_idx + m_i) * lda + k_s];
                    __m512 a_bcast = _mm512_set1_ps(a_scalar); // Broadcast scalar to all lanes

                    // FMA: c_acc[m_i] = a_bcast * b_vec + c_acc[m_i]
                    c_acc[m_i] = _mm512_fmadd_ps(a_bcast, b_vec, c_acc[m_i]);
                }
            }

            // Store accumulated results back to C
            for (int i = 0; i < mr_len; ++i) {
                // C_ptr + (mr_idx + i) * ldc + nr_idx points to the start of current C row in the block
                __mmask16 n_mask = (__mmask16)((1 << nr_len) - 1); // Mask for N-tail
                _mm512_mask_storeu_ps(&C_ptr[(mr_idx + i) * ldc + nr_idx], n_mask, c_acc[i]);
            }
        }
    }
#else
    // Fallback to scalar for compilation without AVX-512 support
    // This part should ideally not be reached if dispatch is correct,
    // but serves as a placeholder for safe compilation.
    (void)A_ptr; (void)B_ptr; (void)C_ptr;
    (void)m_block_size; (void)n_block_size; (void)k_block_size;
    (void)lda; (void)ldb; (void)ldc; (void)is_first_k_block;
#endif
}

// --- Micro-Kernel for AVX2 ---
// Computes C_block += A_block * B_block using AVX2 intrinsics and register blocking.
// Same logic as AVX-512 kernel, but with __m256 (8 floats) instead of __m512 (16 floats).
void micro_kernel_avx2(const float* A_ptr, const float* B_ptr, float* C_ptr,
                       int m_block_size, int n_block_size, int k_block_size,
                       int lda, int ldb, int ldc, bool is_first_k_block) {
#if defined(__AVX2__) && defined(__FMA__)
    // Register blocking constants
    constexpr int MR_ACC = 6;  // Rows of C to compute concurrently
    constexpr int NR_ACC = 8;  // Columns of C to compute concurrently (AVX2 vector width)
    constexpr int SIMD_WIDTH = 8; // float elements in __m256

    // Loop over C rows in MR_ACC chunks
    for (int mr_idx = 0; mr_idx < m_block_size; mr_idx += MR_ACC) {
        int mr_len = std::min(MR_ACC, m_block_size - mr_idx);

        // Loop over C columns in NR_ACC chunks
        for (int nr_idx = 0; nr_idx < n_block_size; nr_idx += NR_ACC) {
            int nr_len = std::min(NR_ACC, n_block_size - nr_idx);

            // C accumulators for the MR_ACC x NR_ACC block
            __m256 c_acc[MR_ACC];

            // Initialize accumulators: zero for first K-block, load from C otherwise
            if (is_first_k_block) {
                for (int i = 0; i < MR_ACC; ++i) {
                    c_acc[i] = _mm256_setzero_ps();
                }
            } else {
                for (int i = 0; i < mr_len; ++i) {
                    // Handle N-tail for loading C: load full vector or scalar fallback
                    if (nr_len == SIMD_WIDTH) {
                        c_acc[i] = _mm256_loadu_ps(&C_ptr[(mr_idx + i) * ldc + nr_idx]);
                    } else {
                        // For AVX2, masked loads are often slow. Use a temporary buffer and scalar copy.
                        alignas(32) float temp_buf[SIMD_WIDTH];
                        for (int j = 0; j < nr_len; ++j) {
                            temp_buf[j] = C_ptr[(mr_idx + i) * ldc + nr_idx + j];
                        }
                        for (int j = nr_len; j < SIMD_WIDTH; ++j) { // Pad remaining with zeros
                            temp_buf[j] = 0.0f;
                        }
                        c_acc[i] = _mm256_load_ps(temp_buf); // Load from aligned buffer
                    }
                }
                for (int i = mr_len; i < MR_ACC; ++i) { // Zero out unused accumulators
                    c_acc[i] = _mm256_setzero_ps();
                }
            }

            // K loop for inner product computation within the micro-kernel
            for (int k_s = 0; k_s < k_block_size; ++k_s) {
                // Prefetch B and A. No specific AVX2 intrinsics, rely on generic prefetch.
                _mm_prefetch((const char*)&B_ptr[k_s * ldb + (nr_idx + NR_ACC)], _MM_HINT_T0);
                _mm_prefetch((const char*)&A_ptr[(mr_idx + MR_ACC) * lda + k_s], _MM_HINT_T0);

                // Load B_vec
                __m256 b_vec;
                if (nr_len == SIMD_WIDTH) {
                    b_vec = _mm256_loadu_ps(&B_ptr[k_s * ldb + nr_idx]);
                } else {
                    alignas(32) float temp_buf[SIMD_WIDTH];
                    for (int j = 0; j < nr_len; ++j) {
                        temp_buf[j] = B_ptr[k_s * ldb + nr_idx + j];
                    }
                    for (int j = nr_len; j < SIMD_WIDTH; ++j) {
                        temp_buf[j] = 0.0f;
                    }
                    b_vec = _mm256_load_ps(temp_buf);
                }

                // For each of MR_ACC rows of A:
                for (int m_i = 0; m_i < mr_len; ++m_i) {
                    // Load A_scalar and broadcast
                    float a_scalar = A_ptr[(mr_idx + m_i) * lda + k_s];
                    __m256 a_bcast = _mm256_set1_ps(a_scalar);

                    // FMA
                    c_acc[m_i] = _mm256_fmadd_ps(a_bcast, b_vec, c_acc[m_i]);
                }
            }

            // Store accumulated results back to C
            for (int i = 0; i < mr_len; ++i) {
                if (nr_len == SIMD_WIDTH) {
                    _mm256_storeu_ps(&C_ptr[(mr_idx + i) * ldc + nr_idx], c_acc[i]);
                } else {
                    alignas(32) float temp_buf[SIMD_WIDTH];
                    _mm256_store_ps(temp_buf, c_acc[i]);
                    for (int j = 0; j < nr_len; ++j) {
                        C_ptr[(mr_idx + i) * ldc + nr_idx + j] = temp_buf[j];
                    }
                }
            }
        }
    }
#else
    (void)A_ptr; (void)B_ptr; (void)C_ptr;
    (void)m_block_size; (void)n_block_size; (void)k_block_size;
    (void)lda; (void)ldb; (void)ldc; (void)is_first_k_block;
#endif
}

// --- AVX-512 GEMM Implementation ---
void gemm_avx512(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Outer loops for M and N dimensions are parallelized using OpenMP
    // schedule(dynamic) is chosen to handle varying block sizes at the edges and
    // potentially imbalanced workloads across threads.
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m_start = 0; m_start < M; m_start += BM) {
        for (int n_start = 0; n_start < N; n_start += BN) {
            int m_block_size = std::min(BM, M - m_start);
            int n_block_size = std::min(BN, N - n_start);

            // Innermost K-loop for a single C[M_block x N_block] tile
            for (int k_start = 0; k_start < K; k_start += BK) {
                int k_block_size = std::min(BK, K - k_start);

                // Pointers to the current sub-blocks of A, B, C
                const float* A_block_ptr = A + m_start * lda + k_start;
                const float* B_block_ptr = B + k_start * ldb + n_start;
                float* C_block_ptr = C + m_start * ldc + n_start;

                // Call the AVX-512 micro-kernel
                micro_kernel_avx512(A_block_ptr, B_block_ptr, C_block_ptr,
                                    m_block_size, n_block_size, k_block_size,
                                    lda, ldb, ldc, k_start == 0);
            }
        }
    }
}

// --- AVX2 GEMM Implementation ---
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int m_start = 0; m_start < M; m_start += BM) {
        for (int n_start = 0; n_start < N; n_start += BN) {
            int m_block_size = std::min(BM, M - m_start);
            int n_block_size = std::min(BN, N - n_start);

            for (int k_start = 0; k_start < K; k_start += BK) {
                int k_block_size = std::min(BK, K - k_start);

                const float* A_block_ptr = A + m_start * lda + k_start;
                const float* B_block_ptr = B + k_start * ldb + n_start;
                float* C_block_ptr = C + m_start * ldc + n_start;

                micro_kernel_avx2(A_block_ptr, B_block_ptr, C_block_ptr,
                                  m_block_size, n_block_size, k_block_size,
                                  lda, ldb, ldc, k_start == 0);
            }
        }
    }
}

// --- Top-level GEMM Dispatcher ---
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    // Runtime dispatch based on CPU features
    // __builtin_cpu_supports is a GCC/Clang extension
#ifdef __AVX512F__
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("fma")) {
        // std::cout << "Using AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif
#ifdef __AVX2__
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        // std::cout << "Using AVX2 kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        return;
    }
#endif
    // Fallback to scalar
    // std::cout << "Using scalar kernel." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
}

// --- Main Function for Demo and Benchmarking ---
int main(int argc, char* argv[]) {
    // Default matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int seed = 42;
    int num_threads_override = 0; // 0 means use OpenMP default
    bool dump_matrices = false;
    bool enable_autotune = false;
    bool check_correctness = false; // Only check correctness for smaller matrices by default

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-M" && i + 1 < argc) { M = std::stoi(argv[++i]); }
        else if (arg == "-N" && i + 1 < argc) { N = std::stoi(argv[++i]); }
        else if (arg == "-K" && i + 1 < argc) { K = std::stoi(argv[++i]); }
        else if (arg == "-s" && i + 1 < argc) { seed = std::stoi(argv[++i]); }
        else if (arg == "-t" && i + 1 < argc) { num_threads_override = std::stoi(argv[++i]); }
        else if (arg == "--dump-matrices") { dump_matrices = true; }
        else if (arg == "--autotune") { enable_autotune = true; }
        else if (arg == "--check-correctness") { check_correctness = true; }
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [-M M_dim] [-N N_dim] [-K K_dim] [-s seed] [-t threads] [--dump-matrices] [--autotune] [--check-correctness] [--help]\n";
            return 0;
        }
    }

    // Override number of threads if specified
    if (num_threads_override > 0) {
        omp_set_num_threads(num_threads_override);
    }
    int actual_num_threads = omp_get_max_threads();
    std::cout << "Running GEMM with M=" << M << ", N=" << N << ", K=" << K
              << ", Threads=" << actual_num_threads << std::endl;

    // Create workspace directory if dumping matrices
    if (dump_matrices) {
        std::filesystem::path workspace_dir("workspace");
        if (!std::filesystem::exists(workspace_dir)) {
            std::filesystem::create_directory(workspace_dir);
        }
    }

    // --- Autotuning Harness (optional, prints suggestions) ---
    if (enable_autotune) {
        std::cout << "\n--- Starting Autotuning (suggested block sizes) ---\n";
        // Use a smaller problem size for autotuning to save time
        const int AT_M = 256, AT_N = 256, AT_K = 256;
        int lda_at = AT_K, ldb_at = AT_N, ldc_at = AT_N;

        std::vector<std::tuple<int, int, int>> candidates = {
            {64, 64, 64}, {96, 96, 96}, {128, 128, 128},
            {64, 128, 96}, {128, 64, 96}, {192, 192, 128}
        };

        std::vector<float> A_at(AT_M * lda_at);
        std::vector<float> B_at(AT_K * ldb_at);
        std::vector<float> C_at(AT_M * ldc_at);

        std::mt19937 rng_at(seed);
        std::uniform_real_distribution<float> dist_at(-1.0f, 1.0f);
        for (size_t i = 0; i < A_at.size(); ++i) A_at[i] = dist_at(rng_at);
        for (size_t i = 0; i < B_at.size(); ++i) B_at[i] = dist_at(rng_at);

        double best_time_ms = std::numeric_limits<double>::max();
        std::tuple<int, int, int> best_params = {BM, BN, BK};

        for (const auto& params : candidates) {
            BM = std::get<0>(params);
            BN = std::get<1>(params);
            BK = std::get<2>(params);

            // Warm-up and measure
            const int WARMUP_RUNS = 2;
            const int MEASURE_RUNS = 5;
            double current_total_time_ms = 0.0;

            for (int run = 0; run < WARMUP_RUNS + MEASURE_RUNS; ++run) {
                std::fill(C_at.begin(), C_at.end(), 0.0f); // Zero C for each run

                auto start = std::chrono::high_resolution_clock::now();
                gemm(A_at.data(), B_at.data(), C_at.data(), AT_M, AT_N, AT_K, lda_at, ldb_at, ldc_at);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> duration = end - start;

                if (run >= WARMUP_RUNS) {
                    current_total_time_ms += duration.count();
                }
            }
            double average_time_ms = current_total_time_ms / MEASURE_RUNS;
            double gflops = 2.0 * AT_M * AT_N * AT_K / (average_time_ms * 1e6);

            std::cout << "  (BM=" << BM << ", BN=" << BN << ", BK=" << BK << ") -> "
                      << "Time: " << average_time_ms << " ms, GFLOPS: " << gflops << "\n";

            if (average_time_ms < best_time_ms) {
                best_time_ms = average_time_ms;
                best_params = params;
            }
        }
        std::cout << "\n--- Autotuning Complete ---\n";
        std::cout << "Suggested parameters: BM=" << std::get<0>(best_params)
                  << ", BN=" << std::get<1>(best_params)
                  << ", BK=" << std::get<2>(best_params) << "\n";
        std::cout << "Current parameters: BM=" << DEFAULT_BM << ", BN=" << DEFAULT_BN << ", BK=" << DEFAULT_BK << "\n";
        std::cout << "Consider updating #define DEFAULT_BM, DEFAULT_BN, DEFAULT_BK at the top of the file and recompiling.\n";

        // Reset BM, BN, BK to original defaults or CLI-set values for the main run
        BM = DEFAULT_BM;
        BN = DEFAULT_BN;
        BK = DEFAULT_BK;
    }

    // Allocate and initialize matrices
    // Using simple row-major layout, lda = K, ldb = N, ldc = N
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Use aligned memory for SIMD operations if possible
    // (std::vector is not guaranteed to be aligned, so manual alignment for raw arrays)
    // For simplicity with std::vector, we rely on _mm*_loadu_ps (unaligned loads)
    // or use custom aligned allocators for vectors if maximum performance on aligned data is required.
    // For this problem, _mm*_loadu_ps is generally fine.
    std::vector<float> A(M * lda);
    std::vector<float> B(K * ldb);
    std::vector<float> C(M * ldc);
    std::vector<float> C_ref(M * ldc); // For correctness check

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < A.size(); ++i) A[i] = dist(rng);
    for (size_t i = 0; i < B.size(); ++i) B[i] = dist(rng);
    std::fill(C.begin(), C.end(), 0.0f); // C must be zeroed before GEMM
    std::fill(C_ref.begin(), C_ref.end(), 0.0f); // C_ref must be zeroed before GEMM

    if (dump_matrices) {
        write_matrix_to_file("workspace/A.txt", A.data(), M, K, lda);
        write_matrix_to_file("workspace/B.txt", B.data(), K, N, ldb);
        std::cout << "Matrices A and B written to workspace/A.txt and workspace/B.txt\n";
    }

    // Warm-up run for consistent timing
    std::cout << "Performing warm-up run...\n";
    std::fill(C.begin(), C.end(), 0.0f);
    gemm(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);

    // Measure performance
    std::cout << "Measuring performance...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A.data(), B.data(), C.data(), M, N, K, lda, ldb, ldc);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
    double gflops = 2.0 * M * N * K / (elapsed_ms.count() * 1e6);

    std::cout << "\n--- Performance Report ---\n";
    std::cout << "Time: " << elapsed_ms.count() << " ms\n";
    std::cout << "GFLOPS: " << gflops << std::endl;

    if (dump_matrices) {
        write_matrix_to_file("workspace/C.txt", C.data(), M, N, ldc);
        std::cout << "Result matrix C written to workspace/C.txt\n";
    }

    // Optional correctness check
    if (check_correctness || (M <= 256 && N <= 256 && K <= 256)) { // Default for small matrices
        std::cout << "\nPerforming correctness check with scalar reference...\n";
        gemm_scalar(A.data(), B.data(), C_ref.data(), M, N, K, lda, ldb, ldc);

        float max_diff = 0.0f;
        float epsilon = 1e-4f; // Tolerance
        int errors = 0;
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float diff = std::fabs(C[i * ldc + j] - C_ref[i * ldc + j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (diff > epsilon * std::max(std::fabs(C[i * ldc + j]), std::fabs(C_ref[i * ldc + j])) + 1e-6f) {
                    if (errors < 10) { // Print first few errors
                        std::cerr << "Mismatch at C[" << i << "][" << j << "]: Optimized="
                                  << C[i * ldc + j] << ", Scalar=" << C_ref[i * ldc + j]
                                  << ", Diff=" << diff << std::endl;
                    }
                    errors++;
                }
            }
        }

        if (errors == 0) {
            std::cout << "Correctness check PASSED. Max difference: " << max_diff << std::endl;
        } else {
            std::cerr << "Correctness check FAILED! Total errors: " << errors
                      << ", Max difference: " << max_diff << std::endl;
            return 1; // Indicate failure
        }
    }

    return 0;
}