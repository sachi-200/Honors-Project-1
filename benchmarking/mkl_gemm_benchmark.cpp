#include <iostream>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include "mkl.h"

// This program takes one command-line argument: the matrix size N.
// It calculates C = A * B where A, B, and C are N x N matrices.
// It prints the performance in GFLOPS/s to standard output.
int main(int argc, char *argv[]) {
    // 1. Parse command-line arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        return 1;
    }
    int n;
    try {
        n = std::stoi(argv[1]);
        if (n <= 0) throw std::invalid_argument("Matrix size must be positive.");
    } catch (const std::exception &e) {
        std::cerr << "Error: Invalid matrix size. " << e.what() << std::endl;
        return 1;
    }

    // 2. Setup matrix dimensions and data
    // For C = alpha * A * B + beta * C
    MKL_INT m = n, k = n, l = n; // M, K, N in BLAS terms
    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate and initialize matrices
    std::vector<float> A(m * k);
    std::vector<float> B(k * l);
    std::vector<float> C(m * l, 0.0f);

    for (int i = 0; i < (m * k); i++) A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < (k * l); i++) B[i] = (float)rand() / RAND_MAX;

    // 3. Perform a warm-up run to initialize MKL threads, etc.
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, l, k, alpha, A.data(), k, B.data(), l, beta, C.data(), l);

    // 4. Time the actual benchmark run
    auto start = std::chrono::high_resolution_clock::now();

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, l, k, alpha, A.data(), k, B.data(), l, beta, C.data(), l);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // 5. Calculate and print GFLOPS/s
    double flops = 2.0 * m * k * l;
    double gflops_per_second = (flops / 1e9) / duration.count();

    // Print only the final result to stdout for easy parsing by Python
    std::cout << gflops_per_second << std::endl;

    return 0;
}