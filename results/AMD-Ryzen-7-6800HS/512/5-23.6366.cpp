// Example compile commands for the specific optimized kernel (gemm_avx2):
// Using -march=x86-64-v2 which implies AVX2/FMA for modern x86-64 CPUs:
// g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2
// Using -march=native (if your CPU supports AVX2/FMA):
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_avx2
// For scalar-only (no specific ISA flags needed):
// g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_scalar

#include <immintrin.h> // For AVX2 intrinsics
#include <iostream>    // For std::cout, std::cerr
#include <vector>      // For std::vector
#include <cstring>     // For std::memcpy
#include <chrono>      // For std::chrono (timing)
#include <random>      // For std::random_device, std::mt19937, std::uniform_real_distribution
#include <cassert>     // For assert
#include <fstream>     // For std::ofstream
#include <string>      // For std::string
#include <iomanip>     // For std::setprecision, std::fixed
#include <algorithm>   // For std::min
#include <cmath>       // For std::abs
#include <memory>      // For std::unique_ptr with custom deleter for aligned memory

#if defined(_OPENMP)
#include <omp.h>       // For OpenMP
#endif

// --- Autotuning Parameters ---
// These parameters are crucial for cache-aware tiling and performance.
// They are chosen based on typical cache sizes and AVX2 vector width for an AMD Ryzen 7 6800HS.
// L1d cache (Ryzen 6800HS): 32KB/core (8-way, 64B line)
// L2 cache (Ryzen 6800HS): 512KB/core (8-way, 64B line)
// L3 cache (Ryzen 6800HS): 16MB shared (16-way, 64B line)

// AVX2 vector width for float (8 floats per __m256)
constexpr int VEC_WIDTH = 8; 

// Micro-kernel register blocking dimensions
// MR: Number of rows of C computed concurrently (from A). Chosen to fill AVX registers.
// NR: Number of columns of C computed concurrently (from B). Must be VEC_WIDTH for vector loads.
// MR=6 (utilizing 6 __m256 accumulators, plus 1 for B vector and 1 for A broadcast = ~8 YMM registers)
// proved to be a good balance for Instruction-Level Parallelism (ILP) without causing
// excessive register spills on AMD Zen architecture. Increasing MR to 8 previously led to significant
// performance regression due to increased register pressure and spills to L1 cache.
constexpr int MR = 6; 
constexpr int NR = VEC_WIDTH; // Fixed by vector width (8 floats).

// Main tile sizes for M, N, K dimensions
// BM, BN for L3 cache blocking and OpenMP thread distribution.
// BK for L1/L2 cache blocking to ensure A and B blocks fit and are reused.
constexpr int BM = 96;  // M-dimension tile size. Chosen as a multiple of MR (96 is divisible by 6) for cleaner tail handling.
                        // BM * sizeof(float) ~ 96 * 4B = 384B (for A block row, fits L1)
constexpr int BN = 96;  // N-dimension tile size. Chosen as a multiple of NR (96 is divisible by 8) for cleaner tail handling.
                        // BN * sizeof(float) ~ 96 * 4B = 384B (for B block row, fits L1)
                        // BM * BN * sizeof(float) = 96 * 96 * 4B = 36KB (for C tile, fits well in L1/L2)
constexpr int BK = 256; // K-dimension tile size. This block of A and B should fit in L2 cache.
                        // (BM * BK + BK * BN) * sizeof(float)
                        // (96 * 256 + 256 * 96) * 4B = 98304B + 98304B = 196KB. This fits well within L2 (512KB per core).

// Explicit K-loop unroll factor. Set to 1 to let the compiler handle unrolling.
// Previous attempts showed that explicit unrolling for K (e.g., 4) led to performance regression,
// possibly due to increased register pressure or suboptimal scheduling on AMD Zen architectures.
// The current code relies on the compiler's unrolling capabilities, which proved more effective.
constexpr int UNROLL_K_FACTOR = 1; 

// --- Helper Functions ---

// Aligned memory allocation for floats (important for SIMD performance)
float* aligned_alloc_float(size_t num_elements, size_t alignment = 32) {
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(num_elements * sizeof(float), alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
#else // Linux/Unix
    // posix_memalign requires alignment to be a power of 2 and a multiple of sizeof(void*)
    // 32 bytes (AVX2 register size) is a common alignment for SIMD.
    int ret = posix_memalign(&ptr, alignment, num_elements * sizeof(float));
    if (ret != 0) {
        throw std::bad_alloc();
    }
#endif
    return static_cast<float*>(ptr);
}

// Aligned memory deallocation
void aligned_free_float(float* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else // Linux/Unix
    free(ptr);
#endif
}

// Custom deleter for unique_ptr to use with aligned_alloc/free
struct AlignedDeleter {
    void operator()(float* ptr) const {
        aligned_free_float(ptr);
    }
};

// Function to write a matrix to a file (for debugging/verification)
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

// --- GEMM Implementations ---

// gemm_scalar: Simple reference implementation (ijk loop order, row-major)
// C = A * B + C (assuming C is zero-initialized or existing values are added to)
// M: rows of A, C
// N: cols of B, C
// K: cols of A, rows of B
// lda, ldb, ldc: leading dimensions (strides)
// This implementation uses an i-k-j loop order, which is generally better for row-major
// matrices as it promotes sequential access for A (row-wise) and B (row-wise for K-dim, then column-wise for J-dim).
void gemm_scalar(const float* A, const float* B, float* C,
                 int M, int N, int K,
                 int lda, int ldb, int ldc) {
    // Zero-initialize the C block explicitly within this function for a clean A*B product.
    // The main function will initialize C to zero for both optimized and scalar calls.
    // This loop just ensures that the `+=` operation behaves as a sum from zero.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * ldc + j] = 0.0f; 
        }
    }

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // Read A[i][k] once per k-loop iteration (row-major access)
            float a_val = A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                // Read B[k][j] (sequential for j if ldb=N) and update C[i][j]
                C[i * ldc + j] += a_val * B[k * ldb + j];
            }
        }
    }
}

// gemm_avx2: Optimized AVX2 + FMA implementation with cache-aware tiling and OpenMP
// C = A * B + C (assuming C is zero-initialized or existing values are added to)
// M: rows of A, C
// N: cols of B, C
// K: cols of A, rows of B
// lda, ldb, ldc: leading dimensions (strides)
void gemm_avx2(const float* A, const float* B, float* C,
               int M, int N, int K,
               int lda, int ldb, int ldc) {

#if defined(__AVX2__) && defined(__FMA__)
    // Outer loops for M, N, K blocks (tiling for L3/L2 cache).
    // The outermost loops (i_block, j_block) are parallelized using OpenMP.
    // The 'collapse(2)' clause tells OpenMP to parallelize across both loops,
    // which can help with load balancing if BM*BN tiles are numerous.
    // 'schedule(static)' provides predictable and usually good load balancing.
#if defined(_OPENMP)
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i_block = 0; i_block < M; i_block += BM) {
        for (int j_block = 0; j_block < N; j_block += BN) {
            // Calculate actual dimensions for the current M-N tile (handling edges)
            int current_M_block_size = std::min(BM, M - i_block);
            int current_N_block_size = std::min(BN, N - j_block);

            // Inner loop for K-blocks (tiling for L2/L1 cache)
            for (int k_block = 0; k_block < K; k_block += BK) {
                // Calculate actual dimensions for the current K tile (handling edges)
                int current_K_block_size = std::min(BK, K - k_block);

                // Micro-kernel loops: processes C[i][j] within the current (current_M x current_N x current_K) block
                // MR rows of C and NR columns of C are processed together in registers.
                for (int i = i_block; i < i_block + current_M_block_size; i += MR) {
                    // Actual number of rows to process in this micro-panel (handling M-tail)
                    int actual_MR = std::min(MR, (i_block + current_M_block_size) - i); 
                    
                    for (int j = j_block; j < j_block + current_N_block_size; j += NR) {
                        // Actual number of columns to process in this micro-panel (handling N-tail)
                        int actual_NR = std::min(NR, (j_block + current_N_block_size) - j); 

                        // Initialize MR * NR C-accumulators to zero for the current micro-panel.
                        // Each c_acc[r] is an __m256 vector, accumulating 8 elements of C.
                        // Using MR accumulators for MR rows.
                        __m256 c_acc[MR]; 
                        for (int r = 0; r < actual_MR; ++r) {
                            c_acc[r] = _mm256_setzero_ps();
                        }
                        
                        // Innermost K-loop: computes the dot product for MR x NR C elements
                        // UNROLL_K_FACTOR is 1, so this simplifies to a direct k loop.
                        for (int k = k_block; k < k_block + current_K_block_size; ++k) {
                            // Load B vector: B[k][j ... j+NR-1]
                            // Conditional load for N-tails: use unmasked load for full vectors,
                            // and memcpy + aligned load for partial vectors. This strategy proved faster
                            // on the target AMD Zen architecture than masked loads (_mm256_maskload_ps).
                            __m256 b_vec;
                            if (actual_NR == NR) {
                                b_vec = _mm256_loadu_ps(B + k * ldb + j); // Unmasked load for full vectors
                            } else {
                                // Handle N-tail for B: Load elements one by one into a temporary aligned buffer,
                                // then load as a vector. This pads with zeros for correctness (zero-multiplication).
                                alignas(32) float b_buffer[NR] = {0.0f}; // Initialize to zero
                                std::memcpy(b_buffer, B + k * ldb + j, actual_NR * sizeof(float));
                                b_vec = _mm256_load_ps(b_buffer); // Aligned load from temp buffer
                            }

                            // Perform MR Fused Multiply-Add (FMA) operations for the current K element
                            // This loop effectively performs (MR * VEC_WIDTH) FMA operations in parallel per k.
                            for (int r = 0; r < actual_MR; ++r) {
                                // Load A scalar: A[i+r][k]
                                // Attempted A-scalar preloading into a stack buffer was removed.
                                // It was found to cause a performance regression on this AMD Zen architecture,
                                // likely due to overhead or less effective prefetching compared to direct access.
                                float a_scalar = A[(i + r) * lda + k];
                                // Broadcast A scalar to all elements of a vector (efficient with FMA)
                                __m256 a_bcast = _mm256_broadcast_ss(&a_scalar);
                                // C_acc[r] = A_scalar * B_vec + C_acc[r] (Fused Multiply-Add)
                                c_acc[r] = _mm256_fmadd_ps(a_bcast, b_vec, c_acc[r]);
                            }
                        }

                        // Store accumulated results back to C matrix
                        for (int r = 0; r < actual_MR; ++r) {
                            // C[i+r][j ... j+NR-1]
                            // Load existing C values, add accumulators, then store back.
                            // Conditional store for N-tails: unmasked store for full vectors,
                            // and scalar store for partial vectors. This was found to be faster
                            // on the target AMD Zen architecture than masked stores (_mm256_blendv_ps).
                            if (actual_NR == NR) {
                                // Full vector load, add, and store
                                __m256 c_current_val = _mm256_loadu_ps(C + (i + r) * ldc + j);
                                c_current_val = _mm256_add_ps(c_current_val, c_acc[r]);
                                _mm256_storeu_ps(C + (i + r) * ldc + j, c_current_val);
                            } else {
                                // Handle N-tail for C: Scalar store for remaining columns.
                                // Extract elements from c_acc[r] and add to C.
                                for (int col = 0; col < actual_NR; ++col) {
                                    C[(i + r) * ldc + j + col] += ((float*)&c_acc[r])[col];
                                }
                            }
                        }
                    } // end of j loop (NR-size micro-panel)
                } // end of i loop (MR-size micro-panel)
            } // end of k_block loop
        } // end of j_block loop
    } // end of i_block loop
#else // __AVX2__ or __FMA__ not defined
    // Fallback to scalar if AVX2/FMA not available during compilation.
    std::cerr << "Warning: gemm_avx2 called but compiled without AVX2/FMA support. Using scalar fallback." << std::endl;
    gemm_scalar(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}

// --- Main Function for Demo and Benchmarking ---

int main(int argc, char** argv) {
    int M = 0, N = 0, K = 0;
    bool dump_matrices = false;

    // Parse command line arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " M N K [--dump-matrices]" << std::endl;
        return 1;
    }

    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);
    K = std::stoi(argv[3]);

    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--dump-matrices") {
            dump_matrices = true;
        }
    }

    // Set number of OpenMP threads if not already set by environment variable (optional)
#if defined(_OPENMP)
    // omp_set_num_threads(NUM_THREADS); // Can be used to explicitly set threads if NUM_THREADS constant is defined.
    std::cout << "OpenMP is enabled. Using up to " << omp_get_max_threads() << " threads." << std::endl;
#else
    std::cout << "OpenMP is NOT enabled. Running single-threaded." << std::endl;
#endif

    // Define leading dimensions (strides) for row-major matrices
    int lda = K; // For A[M][K], lda is K
    int ldb = N; // For B[K][N], ldb is N
    int ldc = N; // For C[M][N], ldc is N

    // Allocate matrices using aligned memory (using unique_ptr for RAII)
    std::unique_ptr<float, AlignedDeleter> A_ptr(aligned_alloc_float(M * K), AlignedDeleter());
    std::unique_ptr<float, AlignedDeleter> B_ptr(aligned_alloc_float(K * N), AlignedDeleter());
    std::unique_ptr<float, AlignedDeleter> C_ptr(aligned_alloc_float(M * N), AlignedDeleter());
    
    float* A = A_ptr.get();
    float* B = B_ptr.get();
    float* C = C_ptr.get();

    // Initialize A and B with random values for testing/benchmarking
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            A[i * lda + k] = dis(gen);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            B[k * ldb + j] = dis(gen);
        }
    }

    // Zero-initialize C. GEMM implementations assume C starts at 0 for C = A*B.
    std::memset(C, 0, M * N * sizeof(float));

    if (dump_matrices) {
        // --- Test Mode: Dump matrices, compute reference, compare, write results ---
        // Allocate reference C matrix
        std::unique_ptr<float, AlignedDeleter> C_ref_ptr(aligned_alloc_float(M * N), AlignedDeleter());
        float* C_ref = C_ref_ptr.get();
        std::memset(C_ref, 0, M * N * sizeof(float)); // Zero-initialize reference C

        // Create workspace directory if it doesn't exist
        std::system("mkdir -p workspace");

        // Write A and B to files
        write_matrix_to_file("workspace/A.txt", A, M, K, lda);
        write_matrix_to_file("workspace/B.txt", B, K, N, ldb);
        std::cout << "Matrices A and B written to workspace/A.txt and workspace/B.txt" << std::endl;

        // Compute reference result with scalar GEMM
        std::cout << "Running gemm_scalar for reference..." << std::endl;
        auto start_scalar = std::chrono::high_resolution_clock::now();
        gemm_scalar(A, B, C_ref, M, N, K, lda, ldb, ldc);
        auto end_scalar = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> scalar_duration = end_scalar - start_scalar;
        std::cout << "Scalar GEMM took: " << scalar_duration.count() * 1000 << " ms" << std::endl;

        // Compute optimized result with AVX2 GEMM
        std::cout << "Running gemm_avx2 (optimized) for result..." << std::endl;
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_duration = end_optimized - start_optimized;
        std::cout << "Optimized AVX2 GEMM took: " << optimized_duration.count() * 1000 << " ms" << std::endl;

        // Write optimized C to file
        write_matrix_to_file("workspace/C.txt", C, M, N, ldc);
        std::cout << "Optimized C matrix written to workspace/C.txt" << std::endl;

        // Correctness check by comparing optimized C with reference C_ref
        bool passed = true;
        float max_diff = 0.0f;
        // A common relative tolerance for float comparisons (e.g., M * K * FLT_EPSILON)
        // For general use, a fixed absolute tolerance like 1e-4 or 1e-5 is often pragmatic.
        const float tolerance = 1e-4f; 

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float diff = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                if (diff > tolerance) { 
                    passed = false;
                    // Optional: print specific mismatch details for debugging
                    // std::cerr << "Mismatch at (" << i << "," << j << "): C_opt=" << C[i * ldc + j] << ", C_ref=" << C_ref[i * ldc + j] << ", diff=" << diff << std::endl;
                    // break; 
                }
            }
            // if (!passed) break;
        }

        if (passed) {
            std::cout << "Internal check: PASSED (Max difference: " << max_diff << ")" << std::endl;
        } else {
            std::cout << "Internal check: FAILED (Max difference: " << max_diff << " > " << tolerance << ")" << std::endl;
        }

    } else {
        // --- Performance Mode: Only run the optimized GEMM and time it ---
        std::cout << "Running gemm_avx2 (optimized) for performance measurement..." << std::endl;
        
        auto start_optimized = std::chrono::high_resolution_clock::now();
        gemm_avx2(A, B, C, M, N, K, lda, ldb, ldc);
        auto end_optimized = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> optimized_duration = end_optimized - start_optimized;

        // Calculate GFLOPS (2 * M * N * K floating point operations)
        double gflops = 2.0 * M * N * K / optimized_duration.count() / 1e9;
        std::cout << "M=" << M << ", N=" << N << ", K=" << K << std::endl;
        std::cout << "Optimized AVX2 GEMM took: " << optimized_duration.count() * 1000 << " ms" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(3) << gflops << " GFLOPS" << std::endl;
    }

    return 0;
}