/**
 * @file tiled_matmul_packing.cpp
 * @brief A C++ program for parallel tiled matrix multiplication using a copy optimization.
 *
 * This program implements the multiplication of two N x N matrices (C = A * B)
 * using a tiled algorithm. To resolve poor cache locality on matrix B, this
 * version uses a "copy optimization" or "packing" technique.
 *
 * Before a tile of B is used in the computation, it is copied into a small,
 * contiguous, stack-allocated buffer. This ensures that the memory access in the
 * innermost compute kernel is sequential (stride-1), dramatically improving
 * cache performance and overall speed.
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
const int TILE_SIZE = 32;

/**
 * @brief Performs tiled matrix multiplication C = A * B using a copy optimization.
 *
 * This function improves the cache performance of the tiled algorithm by "packing"
 * or copying tiles of matrix B into a contiguous local buffer before computation.
 * The standard row-major storage of B leads to large strides (poor spatial
 * locality) when accessed column-wise in the multiplication kernel.
 *
 * By copying the B tile into `B_tile`, we pay a small, one-time cost to
 * rearrange the data. The subsequent, much more intensive computation in the
 * innermost loops can then access B's data from this buffer with a perfect
 * stride-1 pattern, maximizing cache line utilization and performance.
 *
 * @param A Pointer to the first input matrix (N x N).
 * @param B Pointer to the second input matrix (N x N).
 * @param C Pointer to the output matrix (N x N).
 * @param N The dimension of the square matrices.
 */
void perform_tiled_multiplication(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for shared(A, B, C, N)
    for (int ii = 0; ii < N; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            for (int kk = 0; kk < N; kk += TILE_SIZE) {
                // --- Copy Optimization (Packing) ---
                // Declare a small, contiguous array on the stack to hold a tile of B.
                // Its size is fixed by the compile-time constant TILE_SIZE.
                float B_tile[TILE_SIZE][TILE_SIZE];

                // This loop copies the relevant tile from the global matrix B
                // into the local B_tile. This isolates the non-contiguous memory
                // access to this small copy phase.
                for (int k_local = 0; k_local < TILE_SIZE; ++k_local) {
                    for (int j_local = 0; j_local < TILE_SIZE; ++j_local) {
                        // Handle boundary conditions for matrices not perfectly divisible by TILE_SIZE
                        if (kk + k_local < N && jj + j_local < N) {
                           B_tile[k_local][j_local] = B[(kk + k_local) * N + (jj + j_local)];
                        }
                    }
                }

                // --- Inner Computation Kernel ---
                // This kernel now reads from the contiguous B_tile buffer, ensuring
                // stride-1 memory access for B and maximizing cache efficiency.
                for (int i = ii; i < std::min(ii + TILE_SIZE, N); ++i) {
                    for (int k = kk; k < std::min(kk + TILE_SIZE, N); ++k) {
                        const float a_ik = A[i * N + k];
                        for (int j = jj; j < std::min(jj + TILE_SIZE, N); ++j) {
                            // Read from the packed B_tile instead of the global B matrix.
                            // We use local indices (k-kk, j-jj) to access the tile.
                            C[i * N + j] += a_ik * B_tile[k - kk][j - jj];
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
              << " with tile size " << TILE_SIZE << " and copy optimization." << std::endl;

    // 2. Dynamic Memory Allocation
    float* A = new float[N * N];
    float* B = new float[N * N];
    float* C = new float[N * N];

    // 3. Matrix Initialization
    std::srand(std::time(0));
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

    // 5. Memory Deallocation
    delete[] A;
    delete[] B;
    delete[] C;

    std::cout << "Memory freed. Program finished successfully." << std::endl;

    return 0;
}
