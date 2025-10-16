/**
 * @file sequential_tiled_matmul.cpp
 * @brief A basic, sequential C++ program for tiled matrix multiplication.
 *
 * This program demonstrates the fundamental concept of tiled (or blocked)
 * matrix multiplication (C = A * B) in its simplest form. The matrices are
 * conceptually divided into smaller square tiles, and the computation
* proceeds tile by tile. This approach improves cache locality compared
* to a naive i-j-k loop implementation.
 *
 * This version is strictly sequential and does not use OpenMP or any other
 * parallelization library. Its purpose is to clearly illustrate the tiling
 * algorithm itself.
 *
 * @author Gemini
 * @date 2025-10-16
 */

#include <iostream>
#include <vector>
#include <cstdlib> // For std::atoi, std::rand, std::srand
#include <ctime>   // For std::time
#include <algorithm> // For std::min
#include <chrono>  // For high-resolution timing

// Define the tile size. This determines the dimensions of the smaller
// matrix blocks that are processed at one time. A size of 32x32 is
// often a good starting point as it fits well within L1 data caches.
const int TILE_SIZE = 32;

/**
 * @brief Performs a basic, sequential tiled matrix multiplication C = A * B.
 *
 * @param A Pointer to the first input matrix (N x N), row-major.
 * @param B Pointer to the second input matrix (N x N), row-major.
 * @param C Pointer to the output matrix (N x N), row-major.
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    // The three outer loops iterate over the tiles of the matrices.
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // --- Inner loops for the actual computation on a single tile ---
                // These loops multiply a tile from A by a tile from B and
                // accumulate the result into a tile of C.
                // Loop bounds use std::min to correctly handle matrix sizes
                // that are not perfectly divisible by TILE_SIZE.
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        const float a_ik = A[i * N + k];
                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief Main function to drive the matrix multiplication.
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Matrix_Size_N>" << std::endl;
        return 1;
    }
    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: Matrix size N must be a positive integer." << std::endl;
        return 1;
    }

    std::cout << "🚀 Starting sequential tiled matmul for N = " << N << std::endl;

    // Allocate matrices in a contiguous 1D array (row-major order)
    size_t matrix_elements = (size_t)N * N;
    float* A = new float[matrix_elements];
    float* B = new float[matrix_elements];
    float* C = new float[matrix_elements];

    // Initialize matrices
    std::srand(std::time(0));
    for (size_t i = 0; i < matrix_elements; ++i) {
        A[i] = (float)std::rand() / RAND_MAX;
        B[i] = (float)std::rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    std::cout << "Matrices allocated and initialized." << std::endl;
    std::cout << "Performing computation..." << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    perform_tiled_multiplication(A, B, C, N);
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "✅ Computation finished in " << elapsed.count() << " seconds." << std::endl;

    // Free the allocated memory
    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}