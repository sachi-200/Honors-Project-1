/**
 * @file tiled_matmul.cpp
 * @brief A complete C++ program for parallel tiled matrix multiplication.
 *
 * This program performs the multiplication of two dense N x N single-precision
 * floating-point matrices (C = A * B) using a tiled (blocked) algorithm.
 * The computation is parallelized with OpenMP by distributing tile computations
 * across multiple threads.
 *
 * The matrix size N is provided as a command-line argument.
 *
 * @author Gemini
 * @date 2025-10-16
 */

#include <iostream>
#include <vector>
#include <cstdlib> // For std::atoi, std::rand, std::srand
#include <ctime>   // For std::time
#include <omp.h>   // For OpenMP
#include <algorithm> // For std::min

// Define the tile size for the blocked algorithm.
// This size is crucial for performance and should be tuned based on the
// specific hardware's cache size. 32 is often a good starting point.
const int TILE_SIZE = 32;

/**
 * @brief Performs tiled matrix multiplication C = A * B.
 *
 * This function implements a cache-friendly, tiled matrix multiplication
 * algorithm. The matrices are divided into smaller blocks (tiles) of size
 * TILE_SIZE x TILE_SIZE. The outer loops iterate over these tiles, and the
 * inner loops perform the matrix multiplication for the elements within a tile.
 * This approach improves data locality and cache utilization.
 *
 * The outermost loop (iterating over rows of tiles) is parallelized using
 * OpenMP, allowing multiple threads to work on different horizontal strips
 * of the resulting matrix C simultaneously.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    // The #pragma directive parallelizes the outermost loop (the 'ii' loop).
    // Each thread will be assigned a different set of 'ii' values, effectively
    // giving each thread a horizontal strip of tiles of matrix C to compute.
    // The shared clause specifies that A, B, C, and N are shared among all threads.
    // Loop variables (ii, jj, kk, i, k, j) are private to each thread by default.
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // This is the mini-kernel that multiplies one tile of A by one tile of B
                // and adds the result to a tile of C.
                // Loop bounds use std::min to handle cases where N is not perfectly
                // divisible by TILE_SIZE.
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        // Pre-load A[i][k] into a register to reduce memory access
                        // within the innermost loop.
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
 *
 * It handles command-line argument parsing, memory allocation,
 * matrix initialization, calling the core multiplication function,
 * and finally, memory deallocation.
 */
int main(int argc, char* argv[]) {
    // 1. Argument Parsing
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <Matrix_Size_N>" << std::endl;
        return 1;
    }

    int N = std::atoi(argv[1]);
    if (N <= 0) {
        std::cerr << "Error: Matrix size N must be a positive integer." << std::endl;
        return 1;
    }

    std::cout << "🚀 Starting tiled matrix multiplication for N = " << N
              << " with tile size " << TILE_SIZE << "." << std::endl;

    // 2. Dynamic Memory Allocation
    // Matrices are stored in a contiguous 1D array in row-major order.
    size_t matrix_size_bytes = (size_t)N * N * sizeof(float);
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // 3. Matrix Initialization
    std::srand(std::time(0)); // Seed for random number generation

    // Initialize A and B with random values between 0.0 and 1.0
    // Initialize C to all zeros.
    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)std::rand() / RAND_MAX;
        B[i] = (float)std::rand() / RAND_MAX;
        C[i] = 0.0f;
    }

    std::cout << "Matrices allocated and initialized." << std::endl;
    std::cout << "Performing computation..." << std::endl;

    // 4. Core Matrix Multiplication
    double start_time = omp_get_wtime();
    perform_tiled_multiplication(A, B, C, N);
    double end_time = omp_get_wtime();

    std::cout << "✅ Computation finished in " << end_time - start_time << " seconds." << std::endl;

    // Optional: Print a few elements of the result matrix for verification.
    // std::cout << "Result C[0][0]: " << C[0] << std::endl;
    // std::cout << "Result C[N-1][N-1]: " << C[(N - 1) * N + (N - 1)] << std::endl;

    // 5. Memory Deallocation
    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}