#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <omp.h>
#include <immintrin.h> // For AVX intrinsics

// This header is needed for _mm_malloc and _mm_free
#include <xmmintrin.h>
const int TILE_SIZE = 32;


// +---------------------------------------------------------------------------+
// | PASTE YOUR LLM-GENERATED GEMM FUNCTION HERE                               |
// | This is the only section you need to edit.                                |
// +---------------------------------------------------------------------------+

void your_gemm_function(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int kk = 0; kk < N; kk += TILE_SIZE) {
            if (kk + TILE_SIZE < N) {
                __builtin_prefetch(&A[ii * N + (kk + TILE_SIZE)], 0, 3);
            }
            for (int jj = 0; jj < N; jj += TILE_SIZE) {
                if (jj + TILE_SIZE < N) {
                    __builtin_prefetch(&B[kk * N + (jj + TILE_SIZE)], 0, 1);
                }

                alignas(32) float B_tile[TILE_SIZE][TILE_SIZE];
                for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                    for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                        if (kk + k_local < N && jj + j_local < N) {
                           B_tile[k_local][j_local] = B[(kk + k_local) * N + (jj + j_local)];
                        }
                    }
                }

                // --- Vectorized and Unrolled Inner Computation Kernel ---
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        __m256 a_vec = _mm256_broadcast_ss(&A[i * N + k]);

                        int j = jj;
                        const int bound = std::min(jj + TILE_SIZE, N);

                        // Unroll by 4: Process 32-float (4-vector) chunks.
                        for (; j + 31 < bound; j += 32) {
                            __m256 c_vec0 = _mm256_load_ps(&C[i * N + j + 0]);
                            __m256 c_vec1 = _mm256_load_ps(&C[i * N + j + 8]);
                            __m256 c_vec2 = _mm256_load_ps(&C[i * N + j + 16]);
                            __m256 c_vec3 = _mm256_load_ps(&C[i * N + j + 24]);

                            __m256 b_vec0 = _mm256_load_ps(&B_tile[k - kk][j - jj + 0]);
                            __m256 b_vec1 = _mm256_load_ps(&B_tile[k - kk][j - jj + 8]);
                            __m256 b_vec2 = _mm256_load_ps(&B_tile[k - kk][j - jj + 16]);
                            __m256 b_vec3 = _mm256_load_ps(&B_tile[k - kk][j - jj + 24]);

                            c_vec0 = _mm256_fmadd_ps(a_vec, b_vec0, c_vec0);
                            c_vec1 = _mm256_fmadd_ps(a_vec, b_vec1, c_vec1);
                            c_vec2 = _mm256_fmadd_ps(a_vec, b_vec2, c_vec2);
                            c_vec3 = _mm256_fmadd_ps(a_vec, b_vec3, c_vec3);

                            _mm256_store_ps(&C[i * N + j + 0], c_vec0);
                            _mm256_store_ps(&C[i * N + j + 8], c_vec1);
                            _mm256_store_ps(&C[i * N + j + 16], c_vec2);
                            _mm256_store_ps(&C[i * N + j + 24], c_vec3);
                        }

                        // Cleanup loop for remaining 8-float chunks.
                        for (; j + 7 < bound; j += 8) {
                             __m256 c_vec = _mm256_load_ps(&C[i * N + j]);
                             __m256 b_vec = _mm256_load_ps(&B_tile[k - kk][j - jj]);
                             c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                             _mm256_store_ps(&C[i * N + j], c_vec);
                        }

                        // Scalar cleanup for the last <8 elements.
                        for (; j < bound; ++j) {
                            C[i * N + j] += A[i * N + k] * B_tile[k - kk][j - jj];
                        }
                    }
                }
            }
        }
    }
}


// +---------------------------------------------------------------------------+
// | Helper functions to read/write matrices from/to binary files.             |
// | DO NOT MODIFY BELOW THIS LINE.                                            |
// +---------------------------------------------------------------------------+

void read_matrix_from_file(const std::string& filename, float* matrix, int n) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Error: Cannot open file " << filename << " for reading." << std::endl;
        exit(1);
    }
    infile.read(reinterpret_cast<char*>(matrix), (long)n * n * sizeof(float));
    infile.close();
}

void write_matrix_to_file(const std::string& filename, const float* matrix, int n) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot open file " << filename << " for writing." << std::endl;
        exit(1);
    }
    outfile.write(reinterpret_cast<const char*>(matrix), (long)n * n * sizeof(float));
    outfile.close();
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size_N>" << std::endl;
        return 1;
    }

    int N = std::stoi(argv[1]);
    size_t matrix_size_bytes = (size_t)N * N * sizeof(float);

    // --- FIX ---
    // Allocate memory with 32-byte alignment, which is required for AVX instructions.
    // _mm_malloc is the correct way to do this when using intrinsics.
    float* A = (float*)_mm_malloc(matrix_size_bytes, 32);
    float* B = (float*)_mm_malloc(matrix_size_bytes, 32);
    float* C = (float*)_mm_malloc(matrix_size_bytes, 32);

    if (A == nullptr || B == nullptr || C == nullptr) {
        std::cerr << "Error: Memory allocation failed." << std::endl;
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        C[i] = 0.0f;
    }

    // Read input matrices generated by the Python script
    read_matrix_from_file("A.bin", A, N);
    read_matrix_from_file("B.bin", B, N);

    // Call the GEMM function that you pasted above
    your_gemm_function(A, B, C, N);

    // Write the result to a file for the Python script to verify
    write_matrix_to_file("C.bin", C, N);

    // --- FIX ---
    // Free the aligned memory using the corresponding _mm_free function.
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}