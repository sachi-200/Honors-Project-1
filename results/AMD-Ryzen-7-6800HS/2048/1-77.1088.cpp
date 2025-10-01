// Compile instructions:
// For maximum performance and targeting a specific CPU family (e.g., Intel Broadwell/Skylake or AMD Zen2/Zen3),
// use -march=native or specific -march flags.
//
// AVX-512 enabled build (for CPUs that support AVX-512, e.g., Intel Skylake-X, Ice Lake, Sapphire Rapids):
// g++ -O3 -std=c++17 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_avx512
//
// AVX2 enabled build (fallback for CPUs without AVX-512, e.g., AMD Ryzen, Intel Haswell/Broadwell/Skylake):
// g++ -O3 -std=c++17 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
//
// Portable build (detects native capabilities at compile time for -march=native, or uses runtime dispatch):
// g++ -O3 -std=c++17 -march=native -fopenmp gemm.cpp -o gemm_native
//
// For the target AMD Ryzen 7 6800HS (Zen3-based APU), AVX2 is the maximum supported SIMD.
// So, the 'gemm_native' or 'gemm_avx2' commands would be most appropriate.

#include <immintrin.h> // SIMD intrinsics (AVX2, AVX-512, FMA)
#include <iostream>    // std::cout, std::cerr
#include <vector>      // std::vector
#include <cstring>     // memcpy
#include <chrono>      // std::chrono for timing
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <cassert>     // assert
#include <numeric>     // std::iota
#include <algorithm>   // std::min, std::max
#include <fstream>     // std::ofstream
#include <string>      // std::string
#include <filesystem>  // std::filesystem::create_directory (C++17)
#include <limits>      // For std::numeric_limits in allocator
#include <cmath>       // For std::abs

#ifdef _OPENMP
#include <omp.h> // OpenMP for multi-threading
#endif

// --- Tunable Parameters ---
// These parameters control the tiling strategy and micro-kernel unrolling.
// They are chosen to promote data reuse in L1/L2/L3 caches.
// BM: Block size for M dimension (rows of A, C)
// BN: Block size for N dimension (columns of B, C)
// BK: Block size for K dimension (columns of A, rows of B)
// UNROLL_K: K-loop unroll factor within the micro-kernel. Should be a multiple of vector width.

// Default values, good starting point for x86-64 with AVX2/AVX512 on Ryzen 7 6800HS.
// L1d cache: 32KB/core, L2 cache: 512KB/core, L3 cache: 16MB shared.
// Aim: A_block (BMxBK), B_block (BKxBN) should fit within L2/L3 or at least maximize L1 hit rates.
// C_block (BMxBN) is generally large and written to, so register blocking is key.
// Example for these defaults (float is 4 bytes):
// A_block: 96*64*4B = 24.576 KB (fits L1d cache, allowing for A to be reused effectively)
// B_block: 64*192*4B = 49.152 KB (fits L2 cache, allowing for B to be streamed efficiently)
// C_block: 96*192*4B = 73.728 KB (mostly accumulates in registers, then written to L2/L3/main memory)
// BM and BN should be multiples of the micro-kernel sizes (MR, NR) for clean vectorization.
constexpr int DEFAULT_BM = 96;
constexpr int DEFAULT_BN = 192;
constexpr int DEFAULT_BK = 64;
constexpr int DEFAULT_UNROLL_K = 8; // K-loop unroll factor, usually multiple of vector width (8 for AVX2, 16 for AVX-512)

// --- Aligned Memory Allocator ---
// Custom allocator to ensure memory is aligned to a specified boundary (e.g., 64 bytes for AVX-512).
// This is necessary for optimal performance with SIMD loads/stores and to prevent potential
// crashes if using aligned load instructions on unaligned data.
// This allocator provides all necessary type aliases and functions as per C++17 Allocator requirements.
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    // Standard allocator type definitions
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;
    using const_void_pointer = const void*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Required for `std::allocator_traits` to rebind the allocator to a different type
    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {} // Copy constructor for rebind

    // Allocate memory
    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        // Check for multiplication overflow
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc(); // or std::length_error("Allocation size overflow");
        }
        void* ptr = nullptr;
#ifdef _MSC_VER
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
#else
        // On POSIX systems, posix_memalign is standard and widely supported.
        // aligned_alloc (C11/C++17) is an alternative but posix_memalign is very common on Linux.
        // `posix_memalign` returns 0 on success, an error code otherwise.
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
#endif
        return static_cast<pointer>(ptr);
    }

    // Deallocate memory
    void deallocate(pointer p, size_type) {
#ifdef _MSC_VER
        _aligned_free(p);
#else
        free(p); // `free` is compatible with memory allocated by `posix_memalign`
#endif
    }

    // Comparators (all instances of a stateless allocator type are considered equal)
    bool operator==(const AlignedAllocator& other) const noexcept { return true; }
    bool operator!=(const AlignedAllocator& other) const noexcept { return false; }
};

// --- Matrix Storage Convention ---
// All matrices A, B, C are assumed to be stored in row-major order.
// A: M x K matrix, lda is leading dimension (>= K)
// B: K x N matrix, ldb is leading dimension (>= N)
// C: M x N matrix, ldc is leading dimension (>= N)

// --- Helper function for writing matrices to file ---
void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write matrix data. The verification script expects only data, not dimensions on the first line.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ofs << matrix[i * ld + j] << (j == cols - 1 ? "" : " ");
        }
        ofs << std::endl;
    }
    ofs.close();
}

// --- Scalar Reference GEMM ---
// This is a simple, unoptimized scalar implementation, primarily for correctness checking.
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

// --- AVX2 Optimized GEMM Kernel ---
// Uses AVX2 (256-bit) intrinsics with FMA for performance.
// Includes tiling, OpenMP parallelization, and register blocking.
// Targets AMD Ryzen 7 6800HS which supports AVX2 and FMA.
// The `__attribute__((target("avx2,fma")))` ensures this function is compiled with
// the necessary instruction set, allowing runtime dispatch.
void __attribute__((target("avx2,fma"))) gemm_avx2(const float* A, const float* B, float* C,
                                                int M, int N, int K,
                                                int lda, int ldb, int ldc) {
    // Vector width for AVX2 is 8 floats (256 bits)
    constexpr int VEC_FLOAT_COUNT = 8; // Number of floats in a __m256 register

    // Micro-kernel register blocking dimensions (MR x NR)
    // MR: rows of C to compute simultaneously (from A)
    // NR: columns of C to compute simultaneously (from B, typically one vector width)
    // Register blocking for C: accumulate MR x NR floats in registers.
    constexpr int MR = 4; // Number of rows of C to accumulate in registers (e.g., C0..C3)
    constexpr int NR = VEC_FLOAT_COUNT; // Number of columns of C to accumulate (e.g., C_vec0..C_vec7)

    // Ensure tile sizes are compatible with micro-kernel dimensions for clean looping
    const int BM = (DEFAULT_BM / MR) * MR;
    const int BN = (DEFAULT_BN / NR) * NR;
    const int BK = DEFAULT_BK;
    const int UNROLL_K = DEFAULT_UNROLL_K;

#ifdef _OPENMP
    // OpenMP for parallelizing the outer loops (M and N blocks).
    // `collapse(2)` parallelizes both M_BLOCK and N_BLOCK loops.
    // `schedule(static)` is chosen for predictable workload distribution if blocks are uniform.
    // `schedule(guided)` or `dynamic` might be better for irregular workloads.
    // Given matrix sizes are often uniform, static is a good default.
    // Thread-safe writes are ensured because each thread writes to a distinct C block (m_start, n_start).
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m_start = 0; m_start < M; m_start += BM) {
        for (int n_start = 0; n_start < N; n_start += BN) {
            // Determine current block sizes, handling matrix tails
            int current_BM = std::min(BM, M - m_start);
            int current_BN = std::min(BN, N - n_start);

            // Accumulate C_mn = A_mk * B_kn
            // Loop over K-blocks (outermost loop for K)
            for (int k_start = 0; k_start < K; k_start += BK) {
                int current_BK = std::min(BK, K - k_start);

                // Innermost loops for actual computation (micro-kernel level)
                for (int m = m_start; m < m_start + current_BM; m += MR) {
                    int mr_actual = std::min(MR, M - m); // Actual rows for current micro-panel of A/C

                    for (int n = n_start; n < n_start + current_BN; n += NR) {
                        int nr_actual = std::min(NR, N - n); // Actual cols for current micro-panel of B/C

                        // Mask for N-tail processing (AVX2 requires __m256i mask).
                        // This mask has 0xFFFFFFFF for elements < nr_actual, 0x00000000 otherwise.
                        // _mm256_set_epi32(7, 6, ..., 0) creates conceptual indices {7,6,5,4,3,2,1,0}.
                        // _mm256_cmpgt_epi32(nr_actual_vec, indices) sets bits to true (all ones) where index < nr_actual.
                        const __m256i indices = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0); // Reverse order for set_epi32 to match index logic
                        const __m256i nr_actual_vec = _mm256_set1_epi32(nr_actual);
                        const __m256i n_mask_avx2 = _mm256_cmpgt_epi32(nr_actual_vec, indices);

                        // Accumulator registers for C block (MR x NR)
                        // Each c_regs[r] stores an __m256 vector, representing one row of C block.
                        __m256 c_regs[MR];
                        for (int r = 0; r < mr_actual; ++r) {
                            if (k_start == 0) {
                                // Initialize accumulators to zero for the first K-block
                                c_regs[r] = _mm256_setzero_ps();
                            } else {
                                // Load existing C values for accumulation from subsequent K-blocks
                                // Using masked load (_mm256_maskload_ps) to handle N-tails (unaligned access with mask).
                                c_regs[r] = _mm256_maskload_ps(C + (m + r) * ldc + n, n_mask_avx2);
                            }
                        }

                        // K-loop for dot product accumulations (inner-most loop)
                        for (int k_uk_start = k_start; k_uk_start < k_start + current_BK; k_uk_start += UNROLL_K) {
                            // Prefetch A and B data to L1/L2 cache ahead of time.
                            // This helps hide memory latency. _MM_HINT_T0 prefetches to all cache levels.
                            // Prefetch `A` block for the next M x K_UNROLL elements:
                            _mm_prefetch((const char*)(A + (m + mr_actual - 1) * lda + k_uk_start + UNROLL_K), _MM_HINT_T0);
                            // Prefetch `B` block for the next K_UNROLL x N elements:
                            _mm_prefetch((const char*)(B + (k_uk_start + UNROLL_K) * ldb + n + nr_actual - 1), _MM_HINT_T0);

                            // Process UNROLL_K elements of K dimension
                            int unroll_k_actual = std::min(UNROLL_K, K - k_uk_start);
                            for (int uk = 0; uk < unroll_k_actual; ++uk) {
                                // Current K index within the block
                                int k_idx = k_uk_start + uk;

                                // Load A element and broadcast it across a vector (A is row-major)
                                // a_broadcast_val[r] holds A[m+r][k_idx] broadcast to all 8 floats.
                                __m256 a_broadcast_val[MR];
                                for (int r = 0; r < mr_actual; ++r) {
                                    a_broadcast_val[r] = _mm256_broadcast_ss(A + (m + r) * lda + k_idx);
                                }

                                // Load B vector (a row from B for the current K index)
                                // b_vec holds B[k_idx][n...n+7], using mask for N-tail.
                                __m256 b_vec = _mm256_maskload_ps(B + k_idx * ldb + n, n_mask_avx2);

                                // FMA operation: c_regs = c_regs + a_broadcast_val * b_vec
                                // FMA combines multiplication and addition into a single instruction for higher throughput.
                                for (int r = 0; r < mr_actual; ++r) {
                                    c_regs[r] = _mm256_fmadd_ps(a_broadcast_val[r], b_vec, c_regs[r]);
                                }
                            }
                        }

                        // Store results from accumulators back to C.
                        // Use masked store to correctly handle N-tail (partial vectors).
                        for (int r = 0; r < mr_actual; ++r) {
                            _mm256_maskstore_ps(C + (m + r) * ldc + n, n_mask_avx2, c_regs[r]);
                        }
                    } // end of N-loop (micro-kernel)
                } // end of M-loop (micro-kernel)
            } // end of K-block loop
        } // end of N-block loop
    } // end of M-block loop
}

// --- AVX-512 Optimized GEMM Kernel ---
// Uses AVX-512 (512-bit) intrinsics with FMA for performance.
// Similar structure to AVX2, but leverages wider vectors and dedicated mask registers (__mmask16).
// Note: AMD Ryzen 7 6800HS does NOT support AVX-512. This function is included as per requirements
// and for systems that do support it (e.g., modern Intel CPUs).
// The `__attribute__((target("avx512f,fma")))` ensures this function is compiled with
// the necessary instruction set, allowing runtime dispatch.
void __attribute__((target("avx512f,fma"))) gemm_avx512(const float* A, const float* B, float* C,
                                                     int M, int N, int K,
                                                     int lda, int ldb, int ldc) {
    // Vector width for AVX-512 is 16 floats (512 bits)
    constexpr int VEC_FLOAT_COUNT = 16; // Number of floats in a __m512 register

    // Micro-kernel register blocking dimensions (MR x NR)
    constexpr int MR = 4; // Rows of C to compute simultaneously
    constexpr int NR = VEC_FLOAT_COUNT; // Columns of C to compute simultaneously (1 vector)

    // Ensure tile sizes are compatible with micro-kernel dimensions
    const int BM = (DEFAULT_BM / MR) * MR;
    const int BN = (DEFAULT_BN / NR) * NR;
    const int BK = DEFAULT_BK;
    const int UNROLL_K = DEFAULT_UNROLL_K;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int m_start = 0; m_start < M; m_start += BM) {
        for (int n_start = 0; n_start < N; n_start += BN) {
            int current_BM = std::min(BM, M - m_start);
            int current_BN = std::min(BN, N - n_start);

            for (int k_start = 0; k_start < K; k_start += BK) {
                int current_BK = std::min(BK, K - k_start);

                for (int m = m_start; m < m_start + current_BM; m += MR) {
                    int mr_actual = std::min(MR, M - m);

                    for (int n = n_start; n < n_start + current_BN; n += NR) {
                        int nr_actual = std::min(NR, N - n);

                        // AVX-512 uses specific __mmask16 type for masks, simpler to create.
                        // (1U << nr_actual) - 1 creates a bitmask where the first `nr_actual` bits are 1.
                        __mmask16 n_mask = (__mmask16)((1U << nr_actual) - 1);

                        // Accumulator registers for C block (MR x NR)
                        __m512 c_regs[MR];
                        for (int r = 0; r < mr_actual; ++r) {
                            if (k_start == 0) {
                                c_regs[r] = _mm512_setzero_ps();
                            } else {
                                // Load existing C values for accumulation, using mask for N-tail
                                // _mm512_maskz_loadu_ps loads zero for masked-out elements, then accumulates.
                                c_regs[r] = _mm512_maskz_loadu_ps(n_mask, C + (m + r) * ldc + n);
                            }
                        }

                        // K-loop for dot product accumulations
                        for (int k_uk_start = k_start; k_uk_start < k_start + current_BK; k_uk_start += UNROLL_K) {
                            // Prefetching.
                            _mm_prefetch((const char*)(A + (m + mr_actual - 1) * lda + k_uk_start + UNROLL_K), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B + (k_uk_start + UNROLL_K) * ldb + n + nr_actual - 1), _MM_HINT_T0);

                            int unroll_k_actual = std::min(UNROLL_K, K - k_uk_start);
                            for (int uk = 0; uk < unroll_k_actual; ++uk) {
                                int k_idx = k_uk_start + uk;

                                // Load A element and broadcast it
                                __m512 a_broadcast_val[MR];
                                for (int r = 0; r < mr_actual; ++r) {
                                    a_broadcast_val[r] = _mm512_set1_ps(A[(m + r) * lda + k_idx]);
                                }

                                // Load B vector (from current K row), using mask for N-tail
                                // _mm512_maskz_loadu_ps loads zero for elements outside the mask.
                                __m512 b_vec = _mm512_maskz_loadu_ps(n_mask, B + k_idx * ldb + n);

                                // FMA operation
                                for (int r = 0; r < mr_actual; ++r) {
                                    c_regs[r] = _mm512_fmadd_ps(a_broadcast_val[r], b_vec, c_regs[r]);
                                }
                            }
                        }

                        // Store results from accumulators back to C, using mask for N-tail
                        // _mm512_mask_storeu_ps stores only the elements indicated by the mask.
                        for (int r = 0; r < mr_actual; ++r) {
                            _mm512_mask_storeu_ps(C + (m + r) * ldc + n, n_mask, c_regs[r]);
                        }
                    } // end of N-loop (micro-kernel)
                } // end of M-loop (micro-kernel)
            } // end of K-block loop
        } // end of N-block loop
    } // end of M-block loop
}

// --- Top-level GEMM Dispatcher ---
// This function determines which optimized kernel to use based on CPU capabilities at runtime.
// It uses GCC's `__builtin_cpu_supports` for feature detection.
void gemm(const float* A, const float* B, float* C,
          int M, int N, int K,
          int lda, int ldb, int ldc) {
    // Check for AVX-512 support.
    // Note: The specific AVX-512 feature set needs to be checked (e.g., avx512f for foundation, avx512cd, etc.)
    // For general float operations, avx512f is usually sufficient.
    if (__builtin_cpu_supports("avx512f")) {
        // AVX-512 is available, use the AVX-512 kernel
        std::cout << "INFO: Using AVX-512 kernel." << std::endl;
        gemm_avx512(A, B, C, M, N, K, lda, ldb, ldc);
    }
    // Check for AVX2 and FMA support.
    // AMD Ryzen 7 6800HS supports AVX2 and FMA.
    else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        // AVX2 with FMA is available, use the AVX2 kernel
        std::cout << "INFO: Using AVX2+FMA kernel." << std::endl;
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
    }
    // Fallback to scalar if no suitable SIMD extensions are found.
    else {
        std::cerr << "WARNING: No AVX2 or AVX-512 support detected. Falling back to scalar GEMM." << std::endl;
        gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
    }
}

// --- Main Function for Demo and Benchmarking ---
int main(int argc, char* argv[]) {
    int M = 512, N = 512, K = 512; // Default matrix dimensions
    long long seed = 0;           // Default random seed (0 means std::random_device)
    int num_threads = 0;          // 0 means OpenMP will decide based on environment/max cores
    bool dump_matrices = false;   // Flag to dump matrices to files

    // Parse command line arguments
    // Usage: ./gemm [M] [N] [K] [--seed SEED] [--threads N_THREADS] [--dump-matrices]
    if (argc > 1) M = std::stoi(argv[1]);
    if (argc > 2) N = std::stoi(argv[2]);
    if (argc > 3) K = std::stoi(argv[3]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoll(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        } else if (arg == "--dump-matrices") {
            dump_matrices = true;
        }
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    } else {
        num_threads = omp_get_max_threads(); // Get default from OMP environment or system
    }
#else
    num_threads = 1; // No OpenMP, only one thread
#endif

    std::cout << "GEMM Test with M=" << M << ", N=" << N << ", K=" << K
              << ", Threads=" << num_threads << ", Seed=" << seed << std::endl;

    // Allocate matrices using aligned allocator (64 bytes for AVX-512, which covers AVX2 32 bytes)
    // std::vector will handle deallocation automatically.
    using AlignedFloatVector = std::vector<float, AlignedAllocator<float, 64>>;

    AlignedFloatVector A_vec(static_cast<std::size_t>(M) * K);
    AlignedFloatVector B_vec(static_cast<std::size_t>(K) * N);
    AlignedFloatVector C_vec(static_cast<std::size_t>(M) * N);

    float* A = A_vec.data();
    float* B = B_vec.data();
    float* C = C_vec.data();

    // Leading dimensions (strides). For row-major, this is simply the number of columns.
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Initialize matrices with random values
    std::mt19937 gen(seed == 0 ? std::random_device{}() : static_cast<unsigned int>(seed));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (long long i = 0; i < static_cast<long long>(M) * K; ++i) A[i] = dist(gen);
    for (long long i = 0; i < static_cast<long long>(K) * N; ++i) B[i] = dist(gen);
    for (long long i = 0; i < static_cast<long long>(M) * N; ++i) C[i] = 0.0f; // Initialize C to zero

    // --- Optional: Matrix Dumping ---
    if (dump_matrices) {
        std::filesystem::create_directory("workspace"); // Create directory if it doesn't exist
        std::cout << "Dumping A to workspace/A.txt..." << std::endl;
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        std::cout << "Dumping B to workspace/B.txt..." << std::endl;
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
    }

    // --- Performance Measurement ---
    auto start_time = std::chrono::high_resolution_clock::now();
    gemm(A, B, C, M, N, K, lda, ldb, ldc); // Call the top-level dispatcher
    auto end_time = std::chrono::high_resolution_clock::now();

    // --- Optional: Matrix Dumping C ---
    if (dump_matrices) {
        std::cout << "Dumping C to workspace/C.txt..." << std::endl;
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
    }

    // --- Timing Report ---
    double duration_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end_time - start_time).count();
    // A GEMM operation performs 2*M*N*K floating point operations (M*N*K multiplications and M*N*K additions).
    double gflops = (2.0 * M * N * K) / (duration_ms * 1e6);

    std::cout << "Computation Time: " << duration_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    // --- Correctness Check (Optional) ---
    // Perform correctness check for smaller matrices to avoid excessive reference computation time.
    if (M <= 256 && N <= 256 && K <= 256) {
        std::cout << "Running scalar reference for correctness check..." << std::endl;
        AlignedFloatVector C_ref_vec(static_cast<std::size_t>(M) * N);
        float* C_ref = C_ref_vec.data();
        for (long long i = 0; i < static_cast<long long>(M) * N; ++i) C_ref[i] = 0.0f;

        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);

        float max_diff = 0.0f;
        float max_val_C = 0.0f;
        for (long long i = 0; i < static_cast<long long>(M) * N; ++i) {
            max_diff = std::max(max_diff, std::abs(C[i] - C_ref[i]));
            max_val_C = std::max(max_val_C, std::abs(C_ref[i]));
        }

        // Use a relative tolerance for comparison, or absolute if C_ref values are very small.
        float tolerance = 1e-4f * max_val_C; // Tolerance based on the magnitude of results.
        if (max_val_C == 0.0f) tolerance = 1e-6f; // If C is zero, use a small absolute tolerance.

        if (max_diff > tolerance) {
            std::cout << "Correctness Check: FAILED! (Max difference: " << max_diff
                      << ", Max C value: " << max_val_C << ", Tolerance: " << tolerance << ")" << std::endl;
        } else {
            std::cout << "Correctness Check: PASSED (Max difference: " << max_diff
                      << ", Max C value: " << max_val_C << ", Tolerance: " << tolerance << ")" << std::endl;
        }
    } else {
        std::cout << "Correctness check skipped for large matrices (M,N,K > 256)." << std::endl;
    }

    return 0;
}