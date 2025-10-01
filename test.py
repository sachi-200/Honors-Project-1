# AIzaSyAxtd0l68vktXBcKmGIZV8Vk-83vsqALd8

import requests

API_KEY = "AIzaSyAxtd0l68vktXBcKmGIZV8Vk-83vsqALd8"
MODEL = "gemini-2.5-flash"   # or "gemini-2.5-flash" if your account has access
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

def ask_gemini(prompt: str):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }

    response = requests.post(URL, headers=headers, json=data)
    response.raise_for_status()
    resp_json = response.json()

    return resp_json["candidates"][0]["content"]["parts"][0]["text"]

if __name__ == "__main__":
    # ✅ Hard-coded prompt instead of asking via input()
    prompt = """
        You are an expert **C++** programmer specializing in **CPU-optimized dense matrix multiplication (GEMM)** for x86-64.
        Generate a **single, complete C++ source file** implementing the requested GEMM with **SIMD intrinsics** and **multi-threading**, tuned for the CPU below.

        **Target Platform (Host CPU):**
        - Architecture: x86_64 (Intel 11th Gen Core i7-1195G7)
        - SIMD ISA: AVX2, FMA, and AVX-512 (avx512f, avx512bw, avx512dq, avx512vl, avx512cd, avx512vbmi, avx512_vbmi2, avx512_vpopcntdq, avx512_vp2intersect, etc. as available at runtime)
        - Threads: 8 logical CPUs (4 cores, SMT/HT=2)
        - NUMA: single node
        - Endianness: little
        - OS: Linux (assume recent GCC/Clang toolchain)

        **Request:**
        {instruction}

        **CRITICAL FUNCTION INFORMATION:**
        Based on analysis, the implementation requires these EXACT function signatures (C++):
        {function_signatures}

        **Output Requirements:**
        1) **Language & File:**
        - Output a single, self-contained **.cpp** file (no external headers beyond the standard library and intrinsics).
        - C++17 or later.
        - Include compile instructions as a comment at the top (e.g., GCC/Clang).
        - No CUDA/ROCm/Triton—**pure CPU** implementation.

        2) **Includes & Build:**
        - Mandatory includes: <immintrin.h>, <iostream>, <vector>, <cstring>, <chrono>, <random>, <cassert>.
        - If using OpenMP: also include <omp.h> and guard logic if OpenMP is not available.
        - Provide example compile commands (both AVX-512 and AVX2 fallbacks), e.g.:
            - AVX-512: `g++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm`
            - AVX2:    `g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm`
            - Portable: `g++ -O3 -march=native -fopenmp gemm.cpp -o gemm`

        3) **SIMD & Dispatch:**
        - Implement **runtime dispatch**:
            - Prefer **AVX-512** kernel when `__builtin_cpu_supports("avx512f")` is true.
            - Else use **AVX2+FMA** kernel when `__builtin_cpu_supports("avx2")`.
            - Else fall back to a **scalar** reference kernel.
        - Each kernel must share the **same external function signature(s)** given in {function_signatures}.
        - Use 64-byte alignment where helpful for AVX-512 (32-byte for AVX2); handle unaligned tails safely.

        4) **Parallelization:**
        - Use **OpenMP** for outer-loop parallelism over tiles/blocks (M×N).
        - Choose a sane default schedule (e.g., static or guided) and justify in comments.
        - Ensure thread-safe writes (distinct C tiles per thread or proper reductions).

        5) **Blocking/Tiling & Memory:**
        - Implement **cache-aware tiling** with tunable tile sizes **BM, BN, BK**.
        - Favor row-major storage; document your convention.
        - Coalesce accesses and prefetch when beneficial (`_mm_prefetch`).
        - Use **float32 accumulation** for numerical stability even if inputs are float16/bfloat16 (if supported); otherwise use float32 inputs.

        6) **Autotuning Parameters (Mandatory to expose as constants at the top or via constexpr):**
        - **BM, BN, BK** (tile sizes) — explore values like: {32, 48, 64, 96, 128, 192}.
        - **UNROLL_K** — inner K unroll factor (e.g., {1, 2, 4, 8}).
        - **NUM_THREADS** — optionally allow setting via OMP or environment; provide a CLI flag or environment read (`OMP_NUM_THREADS`).
        - Provide a simple **autotune harness** (optional but preferred): try a few tile combos on a small warm-up and pick the best for the current run (time-boxed).

        7) **Edge Handling & Correctness:**
        - Correctly handle M, N, K not divisible by tile or vector widths (tail processing).
        - Avoid UB: masks or scalar tails for leftover columns/rows.
        - Provide a **reference (scalar) implementation** used for optional correctness checking (A, B random init; compare C vs C_ref with a tolerance).

        8) **Function Signatures (CRITICAL):**
        - Define EACH function with EXACTLY the signature(s) listed in {function_signatures}.
        - **Do NOT** change parameter names, counts, order, or const-ness.
        - All function calls must match their definitions exactly.
        - If the interface includes strides/leading dimensions (lda/ldb/ldc), use them correctly.
        - If half-precision is requested, convert to float for accumulation.

        9) **CLI / Demo Main:**
        - Provide a `main()` that:
            - Parses optional CLI args: M N K, seed, threads.
            - Allocates and initializes A, B (random), C (zero).
            - Calls the top-level API from {function_signatures}, which internally dispatches to AVX-512/AVX2/scalar.
            - Prints a short timing report (ms, effective GFLOP/s = 2*M*N*K / time).
            - Optionally runs correctness check vs reference (toggle with a flag).

        10) **Documentation & Comments:**
        - Briefly explain blocking choices, threading strategy, and ISA dispatch.
        - Comment the intrinsics code (register layout, accumulation pattern).
        - Mention cache levels (L1/L2/L3) and how tiles aim to fit/reuse.

        **Performance Hints (Follow where feasible):**
        - Pack micro-panels of B for better locality; consider packing A or both if time permits.
        - Favor **register blocking** for the inner micro-kernel (e.g., 8×8 or 16×8 accumulators for AVX-512/AVX2).
        - Use FMA where available; minimize gathers/scatters; prefer contiguous loads/stores.
        - Use `_mm512_setzero_ps`, `_mm512_fmadd_ps`, `_mm512_loadu_ps`, `_mm512_storeu_ps` (AVX-512) or corresponding AVX2 intrinsics (`__m256`).
        - Guard intrinsics with `#if defined(__AVX512F__)` / `#if defined(__AVX2__)` as needed.

        **Final Verification Checklist:**
        1) ALL functions you define **exactly** match the signatures in {function_signatures}.
        2) ALL calls match function definitions (names, params, order, types).
        3) No function is called without being defined.
        4) All parameters required by the signatures are used meaningfully.
        5) Tails (non-multiple-of-vector/tile sizes) are correct and tested.
        6) The code compiles on GCC/Clang with the provided commands.

        **Generated CPU-Optimized C++ GEMM Source (.cpp):**
        """

    answer = ask_gemini(prompt)
    print("Prompt:", prompt)
    print("Gemini:", answer)


############# First Prompt Response ########################

# #include <immintrin.h> // For SIMD intrinsics (__m256, __m512, _mm256_*, _mm512_*)
# #include <iostream>    // For input/output (std::cout, std::cerr, std::endl)
# #include <vector>      // For std::vector (dynamic arrays for matrices)
# #include <cstring>     // For memcpy, memset (not explicitly used for data copies, but part of standard library)
# #include <chrono>      // For timing (std::chrono::high_resolution_clock)
# #include <random>      // For random number generation (std::mt19937, std::uniform_real_distribution)
# #include <cassert>     // For assert (runtime checks for invariants)
# #include <omp.h>       // For OpenMP multi-threading (pragma omp, omp_set_num_threads)
# #include <numeric>     // For std::iota (not directly used but common)
# #include <cmath>       // For std::abs (floating-point absolute value)
# #include <algorithm>   // For std::min, std::max, std::fill (utility functions)
# #include <limits>      // For std::numeric_limits (e.g., epsilon for float comparisons)


# // CRITICAL FUNCTION INFORMATION:
# // Based on analysis, the implementation requires these EXACT function signatures (C++):
# // void my_sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);
# // This specific signature is assumed and used throughout the implementation.

# // Forward declaration of the critical function signature.
# // Its implementation will act as a dispatcher.
# void my_sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);


# // =================================================================================================
# // Compiler Instructions and Build Commands
# // =================================================================================================

# /*
# To compile this GEMM implementation:

# Recommended for target CPU (Intel 11th Gen Core i7-1195G7) with AVX-512 support:
#   g++ -O3 -march=native -fopenmp gemm.cpp -o gemm_native_avx512
#   (This will enable AVX-512 and other native optimizations based on your current CPU.
#    For 11th Gen Intel, `native` includes AVX512F/BW/DQ/VL/CD and FMA.)

# Explicitly enable AVX-512 (x86-64-v3 architecture provides core AVX-512 features like F/BW/DQ/VL for this CPU):
#   g++ -O3 -march=x86-64-v3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -fopenmp gemm.cpp -o gemm_avx512

# Explicitly enable AVX2+FMA (fallback if AVX-512 not desired or available):
#   g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm_avx2

# Portable build (will compile to x86-64-v2 baseline, using scalar or AVX2 if available at runtime):
#   g++ -O3 -march=x86-64-v2 -fopenmp gemm.cpp -o gemm_portable

# Using Clang (similar options):
#   clang++ -O3 -march=native -fopenmp gemm.cpp -o gemm_clang_native
#   clang++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm_clang_avx512
# */


# // =================================================================================================
# // Tunable Parameters & Global State for Autotuning
# // =================================================================================================

# // Structure to hold tunable parameters for a specific architecture/kernel.
# // These parameters define the blocking strategy and micro-kernel characteristics.
# struct TunableParams {
#     int bm, bn, bk;               // Block sizes for M, N, K dimensions (outer tiling loops)
#     int mr, nr;                   // Micro-kernel dimensions for M, N (register blocking)
#     int k_unroll;                 // K-dimension unroll factor within the micro-kernel
#     int alignment_bytes;          // Recommended memory alignment for temporary buffers (e.g., 64 for AVX-512, 32 for AVX2)
#     int vector_width_floats;      // Number of float elements in a SIMD vector (e.g., 8 for AVX2, 16 for AVX-512)
#     const char* kernel_name;      // Descriptive name of the kernel (e.g., "AVX-512", "AVX2", "Scalar")
# };

# // Default parameters for different SIMD ISAs. These are initial guesses
# // that can be refined by the autotuning step in `main()`.

# // AVX-512 specific parameters, tuned for Intel 11th Gen Core i7-1195G7.
# // L1d: 48KB, L2: 1.25MB (per core), L3: 12MB (shared).
# // BM, BN, BK are chosen to keep data within L2/L3 cache during block processing.
# // MR=8, NR=16: The micro-kernel computes an 8x16 sub-block of C.
# // This requires 8 `__m512` accumulators (one for each row in C's sub-block).
# constexpr TunableParams AVX512_PARAMS = {
#     128, 192, 48,   // BM, BN, BK (M, N, K block sizes)
#     8, 16,          // MR, NR (Micro-kernel M, N dimensions. NR should match vector_width_floats)
#     4,              // K_UNROLL (unroll factor for the inner K loop of the micro-kernel)
#     64,             // ALIGNMENT_BYTES (64 bytes for `__m512` vectors)
#     16,             // VECTOR_WIDTH_FLOATS (16 floats per `__m512` vector)
#     "AVX-512"
# };

# // AVX2 specific parameters, serving as a fallback for Intel 11th Gen.
# // MR=8, NR=8: The micro-kernel computes an 8x8 sub-block of C.
# // This requires 8 `__m256` accumulators.
# constexpr TunableParams AVX2_PARAMS = {
#     96, 128, 64,    // BM, BN, BK
#     8, 8,           // MR, NR (NR should match vector_width_floats)
#     4,              // K_UNROLL
#     32,             // ALIGNMENT_BYTES (32 bytes for `__m256` vectors)
#     8,              // VECTOR_WIDTH_FLOATS (8 floats per `__m256` vector)
#     "AVX2"
# };

# // Scalar fallback parameters, used if no SIMD instruction set is available or enabled.
# constexpr TunableParams SCALAR_PARAMS = {
#     32, 32, 32,     // BM, BN, BK (smaller blocks for basic cache locality)
#     1, 1,           // MR, NR (1x1 scalar operation in the micro-kernel)
#     1,              // K_UNROLL
#     1,              // ALIGNMENT_BYTES (not strictly needed but for consistency)
#     1,              // VECTOR_WIDTH_FLOATS (single float)
#     "Scalar"
# };

# // Global static variable to hold the currently selected tunable parameters.
# // This allows the `main()` function's autotuner to influence `my_sgemm` behavior.
# // It's initialized to `SCALAR_PARAMS` and updated by the runtime dispatcher or autotuner.
# static TunableParams current_tunable_params = SCALAR_PARAMS;

# // Default number of threads to use if not specified via CLI or OMP_NUM_THREADS environment variable.
# constexpr int DEFAULT_NUM_THREADS = 8; // Based on 4 physical cores with 2-way SMT/Hyper-threading.


# // =================================================================================================
# // Aligned Memory Allocator
# // =================================================================================================

# // `aligned_malloc` allocates memory aligned to `alignment` bytes.
# // This is crucial for performance with SIMD loads, especially for `_mm512_load_ps` and `_mm256_load_ps`.
# void* aligned_malloc(size_t size, size_t alignment) {
# #ifdef _MSC_VER // Microsoft Visual C++ compiler uses `_aligned_malloc`
#     return _aligned_malloc(size, alignment);
# #else // GCC/Clang and other POSIX-compliant compilers use `posix_memalign`
#     void* ptr = nullptr;
#     // `posix_memalign` requires alignment to be a power of 2 and a multiple of `sizeof(void*)`.
#     if (posix_memalign(&ptr, alignment, size) != 0) {
#         return nullptr; // Allocation failed
#     }
#     return ptr;
# #endif
# }

# // `aligned_free` deallocates memory allocated by `aligned_malloc`.
# void aligned_free(void* ptr) {
# #ifdef _MSC_VER
#     _aligned_free(ptr);
# #else
#     free(ptr); // `free` works with `posix_memalign` allocations
# #endif
# }


# // =================================================================================================
# // Reference (Scalar) GEMM Implementation
# // =================================================================================================

# // `sgemm_ref` computes `C = alpha * A * B + beta * C` using a basic triple nested loop.
# // This serves as a ground truth for correctness checking.
# // Matrices A, B, C are assumed to be stored in row-major order.
# // `lda`, `ldb`, `ldc` are the leading dimensions (strides) of A, B, C respectively.
# void sgemm_ref(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
#     for (int i = 0; i < M; ++i) { // Loop over rows of A and C
#         for (int j = 0; j < N; ++j) { // Loop over columns of B and C
#             float c_val_accum = 0.0f;
#             for (int k = 0; k < K; ++k) { // Loop over inner dimension K
#                 // A is M x K, B is K x N.
#                 // Element A[i][k] * B[k][j] is accumulated.
#                 c_val_accum += A[i * lda + k] * B[k * ldb + j];
#             }
#             // Apply beta: C_new = beta * C_old
#             if (beta == 0.0f) {
#                 C[i * ldc + j] = alpha * c_val_accum;
#             } else {
#                 C[i * ldc + j] = alpha * c_val_accum + beta * C[i * ldc + j];
#             }
#         }
#     }
# }


# // =================================================================================================
# // Micro-kernels (AVX-512, AVX2, Scalar)
# // =================================================================================================
# // These micro-kernels operate on small, cache-resident blocks of data.

# // `sgemm_ukernel_scalar` is the micro-kernel for the scalar fallback.
# // It computes a 1x1 block of C (MR=1, NR=1) by iterating over the K_panel dimension.
# // A_panel: packed A block, effectively a column from A_block.
# // B_panel: packed B block, effectively a row from B_block.
# // C_panel: a single element in the C matrix.
# void sgemm_ukernel_scalar(int K_panel, const float* A_panel, const float* B_panel, float* C_panel, int ldc_panel, int MR, int NR) {
#     // For the scalar micro-kernel, MR=1, NR=1.
#     // The packed A_panel effectively stores K_panel values for the current row.
#     // The packed B_panel effectively stores K_panel values for the current column.
#     float c_val_accum = 0.0f;
#     for (int k = 0; k < K_panel; ++k) {
#         // A_panel is packed as K-major, MR-minor, so A[i_micro][k] is at A_panel[k*MR + i_micro].
#         // B_panel is packed as K-major, NR-minor, so B[k][j_micro] is at B_panel[k*NR + j_micro].
#         c_val_accum += A_panel[k * MR + 0] * B_panel[k * NR + 0];
#     }
#     C_panel[0] = c_val_accum; // Store the single accumulated C value.
# }

# #if defined(__AVX2__) && defined(__FMA__)
# // `sgemm_ukernel_avx2` is the micro-kernel leveraging AVX2 and FMA instructions.
# // It computes an `MR` x `NR` block of C, where `NR` is the AVX2 vector width (8 floats).
# // A_panel: points to a packed A block. Accesses are scalar, but `MR` values per `k` step.
# // B_panel: points to a packed B block. Accesses are vector loads (`__m256`).
# // C_panel: points to the start of the C sub-block to be updated.
# void sgemm_ukernel_avx2(int K_panel, const float* A_panel, const float* B_panel, float* C_panel, int ldc_panel, int MR, int NR, int K_UNROLL) {
#     // For AVX2, MR typically 8, NR typically 8 (AVX2 vector width).
#     __m256 c_regs[8]; // Array of MR `__m256` accumulators for the C block (8x8 floats).
#     for (int i = 0; i < MR; ++i) {
#         c_regs[i] = _mm256_setzero_ps(); // Initialize accumulators to zero.
#     }

#     const float* a_ptr = A_panel; // Pointer to packed A data
#     const float* b_ptr = B_panel; // Pointer to packed B data

#     // Loop over the K_panel dimension, processing `K_UNROLL` steps at a time.
#     int k_unroll_limit = K_panel - (K_panel % K_UNROLL);
#     for (int k = 0; k < k_unroll_limit; k += K_UNROLL) {
#         for (int u = 0; u < K_UNROLL; ++u) {
#             // Optional prefetching for B data to bring it into cache ahead of time.
#             // _mm_prefetch(b_ptr + 2 * NR, _MM_HINT_T0); 
#             // Load a vector from B_panel (8 floats). This load is aligned due to packing strategy.
#             __m256 b_vec = _mm256_load_ps(b_ptr); 
#             b_ptr += NR; // Advance B pointer to the next vector.

#             // Perform Fused Multiply-Add (FMA) for each row of the C block.
#             // `c_regs[i] = a_scalar_broadcast * b_vec + c_regs[i]`
#             for (int i = 0; i < MR; ++i) {
#                 // Broadcast an A scalar (a_ptr[i]) to all elements of an `__m256` vector.
#                 __m256 a_scalar_broadcast = _mm256_set1_ps(a_ptr[i]);
#                 c_regs[i] = _mm256_fmadd_ps(a_scalar_broadcast, b_vec, c_regs[i]);
#             }
#             a_ptr += MR; // Advance A pointer to the next set of MR scalars for the next K-step.
#         }
#     }

#     // Process the remaining K_panel elements (the tail after unrolling).
#     for (int k = k_unroll_limit; k < K_panel; ++k) {
#         // _mm_prefetch(b_ptr + 2 * NR, _MM_HINT_T0); 
#         __m256 b_vec = _mm256_load_ps(b_ptr);
#         b_ptr += NR;

#         for (int i = 0; i < MR; ++i) {
#             __m256 a_scalar_broadcast = _mm256_set1_ps(a_ptr[i]);
#             c_regs[i] = _mm256_fmadd_ps(a_scalar_broadcast, b_vec, c_regs[i]);
#         }
#         a_ptr += MR;
#     }

#     // Store the accumulated results from `c_regs` back into the C matrix.
#     // `_mm256_storeu_ps` is used for unaligned stores, as the target C_panel in main memory
#     // might not be 32-byte aligned, even if C itself is aligned.
#     for (int i = 0; i < MR; ++i) {
#         _mm256_storeu_ps(C_panel + i * ldc_panel, c_regs[i]);
#     }
# }
# #endif // __AVX2__

# #if defined(__AVX512F__) && defined(__FMA__)
# // `sgemm_ukernel_avx512` is the micro-kernel leveraging AVX-512 and FMA instructions.
# // It computes an `MR` x `NR` block of C, where `NR` is the AVX-512 vector width (16 floats).
# // The logic is analogous to the AVX2 kernel, but uses `__m512` vectors and associated intrinsics.
# void sgemm_ukernel_avx512(int K_panel, const float* A_panel, const float* B_panel, float* C_panel, int ldc_panel, int MR, int NR, int K_UNROLL) {
#     // For AVX-512, MR typically 8, NR typically 16 (AVX-512 vector width).
#     __m512 c_regs[8]; // Array of MR `__m512` accumulators for the C block (8x16 floats).
#     for (int i = 0; i < MR; ++i) {
#         c_regs[i] = _mm512_setzero_ps(); // Initialize accumulators to zero.
#     }

#     const float* a_ptr = A_panel; // Pointer to packed A data
#     const float* b_ptr = B_panel; // Pointer to packed B data

#     // Loop over K_panel with unrolling.
#     int k_unroll_limit = K_panel - (K_panel % K_UNROLL);
#     for (int k = 0; k < k_unroll_limit; k += K_UNROLL) {
#         for (int u = 0; u < K_UNROLL; ++u) {
#             // Optional prefetching.
#             // _mm_prefetch(b_ptr + 2 * NR, _MM_HINT_T0); 
#             // Load a vector from B_panel (16 floats). This load is aligned due to packing strategy.
#             __m512 b_vec = _mm512_load_ps(b_ptr);
#             b_ptr += NR; // Advance B pointer.

#             // FMA operation.
#             for (int i = 0; i < MR; ++i) {
#                 // Broadcast an A scalar (a_ptr[i]) to all elements of an `__m512` vector.
#                 __m512 a_scalar_broadcast = _mm512_set1_ps(a_ptr[i]);
#                 c_regs[i] = _mm512_fmadd_ps(a_scalar_broadcast, b_vec, c_regs[i]);
#             }
#             a_ptr += MR; // Advance A pointer.
#         }
#     }

#     // K tail processing.
#     for (int k = k_unroll_limit; k < K_panel; ++k) {
#         // _mm_prefetch(b_ptr + 2 * NR, _MM_HINT_T0); 
#         __m512 b_vec = _mm512_load_ps(b_ptr);
#         b_ptr += NR;

#         for (int i = 0; i < MR; ++i) {
#             __m512 a_scalar_broadcast = _mm512_set1_ps(a_ptr[i]);
#             c_regs[i] = _mm512_fmadd_ps(a_scalar_broadcast, b_vec, c_regs[i]);
#         }
#         a_ptr += MR;
#     }

#     // Store results back to C_panel (unaligned store).
#     for (int i = 0; i < MR; ++i) {
#         _mm512_storeu_ps(C_panel + i * ldc_panel, c_regs[i]);
#     }
# }
# #endif // __AVX512F__


# // =================================================================================================
# // Tiling and Packing Logic
# // =================================================================================================

# // `sgemm_tiled_impl` implements the blocked GEMM algorithm.
# // It orchestrates tiling, data packing, and calls the appropriate micro-kernels.
# // Matrices A, B, C are assumed to be in row-major order.
# void sgemm_tiled_impl(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc, const TunableParams& params) {
#     const int BM = params.bm; // Block size for M dimension
#     const int BN = params.bn; // Block size for N dimension
#     const int BK = params.bk; // Block size for K dimension
#     const int MR = params.mr; // Micro-kernel M dimension (register blocking)
#     const int NR = params.nr; // Micro-kernel N dimension (register blocking, should match vector width)
#     const int K_UNROLL = params.k_unroll;
#     const int VEC_WIDTH = params.vector_width_floats;
#     const int ALIGNMENT = params.alignment_bytes;

#     assert(NR == VEC_WIDTH); // Critical assumption for this micro-kernel design: NR equals vector width.

#     // Define the micro-kernel to use based on the `params.kernel_name`.
#     // This uses a lambda to capture `MR`, `NR`, `K_UNROLL` and simplify the call site.
#     auto micro_kernel_dispatch = [&](int k_panel, const float* a_p, const float* b_p, float* c_p, int ldc_p) {
#         if (params.kernel_name == AVX512_PARAMS.kernel_name) {
# #if defined(__AVX512F__) && defined(__FMA__)
#             sgemm_ukernel_avx512(k_panel, a_p, b_p, c_p, ldc_p, MR, NR, K_UNROLL);
# #else
#             // This case should ideally not be reached if compile flags are correct.
#             sgemm_ukernel_scalar(k_panel, a_p, b_p, c_p, ldc_p, MR, NR);
# #endif
#         } else if (params.kernel_name == AVX2_PARAMS.kernel_name) {
# #if defined(__AVX2__) && defined(__FMA__)
#             sgemm_ukernel_avx2(k_panel, a_p, b_p, c_p, ldc_p, MR, NR, K_UNROLL);
# #else
#             // This case should ideally not be reached.
#             sgemm_ukernel_scalar(k_panel, a_p, b_p, c_p, ldc_p, MR, NR);
# #endif
#         } else {
#             sgemm_ukernel_scalar(k_panel, a_p, b_p, c_p, ldc_p, MR, NR);
#         }
#     };

#     // Allocate thread-local temporary buffers for packing A and B blocks.
#     // These buffers store sub-blocks of A and B in an optimized layout for the micro-kernel.
#     // They are allocated once per thread to avoid race conditions and reduce overhead.
#     // The sizes are calculated to accommodate the largest possible blocks (BM*BK and BK*BN).
#     // The layout ensures elements accessed by SIMD loads are contiguous and aligned.

#     // Packed A block layout: K-major, MR-minor.
#     // `packed_A_block_buffer[k_idx * MR + i_micro]` stores `A[i_micro][k_idx]`.
#     // The `(BM + MR - 1) / MR * MR` ensures padding to a multiple of `MR`.
#     size_t packed_A_max_size = ((size_t)BM + MR - 1) / MR * MR * BK;
#     float* packed_A_block_buffer = (float*)aligned_malloc(packed_A_max_size * sizeof(float), ALIGNMENT);
#     if (!packed_A_block_buffer) {
#         std::cerr << "Error: Failed to allocate aligned memory for packed_A_block_buffer" << std::endl;
#         return; 
#     }

#     // Packed B block layout: K-major, NR-vector-block minor.
#     // `packed_B_block_buffer[k_idx * (ceil(BN/NR)) * NR + (n_vec_idx) * NR + n_elem_in_vec]`
#     // The `(BN + NR - 1) / NR * NR` ensures padding to a multiple of `NR` (vector width).
#     size_t packed_B_max_size = ((size_t)BN + NR - 1) / NR * NR * BK;
#     float* packed_B_block_buffer = (float*)aligned_malloc(packed_B_max_size * sizeof(float), ALIGNMENT);
#     if (!packed_B_block_buffer) {
#         aligned_free(packed_A_block_buffer); // Free A buffer if B allocation fails
#         std::cerr << "Error: Failed to allocate aligned memory for packed_B_block_buffer" << std::endl;
#         return;
#     }

#     // Outer loops for tiling the C matrix (M x N dimensions).
#     // OpenMP parallelizes these loops, distributing C blocks among threads.
#     // `collapse(2)` combines the `m_start` and `n_start` loops into a single parallel region.
#     // `schedule(static)` assigns fixed-size chunks of iterations to threads, which works well
#     // for uniform work distribution typical in GEMM.
# #pragma omp parallel for collapse(2) schedule(static)
#     for (int m_start = 0; m_start < M; m_start += BM) {
#         for (int n_start = 0; n_start < N; n_start += BN) {
#             // Determine actual dimensions of the current C block, handling edge cases.
#             int current_BM = std::min(BM, M - m_start);
#             int current_BN = std::min(BN, N - n_start);

#             // Pointer to the start of the current C block in the original C matrix.
#             float* C_block_ptr = C + m_start * ldc + n_start;

#             // Apply beta: `C_block = beta * C_block_original`.
#             // If `beta` is 0, the C block is initialized to zeros.
#             // If `beta` is 1, the original C values are preserved for accumulation.
#             // Otherwise, C is scaled by `beta`.
#             if (beta != 1.0f) {
#                 for (int i = 0; i < current_BM; ++i) {
#                     for (int j = 0; j < current_BN; ++j) {
#                         if (beta == 0.0f) {
#                             C_block_ptr[i * ldc + j] = 0.0f;
#                         } else {
#                             C_block_ptr[i * ldc + j] *= beta;
#                         }
#                     }
#                 }
#             }
#             // If beta is 1.0f, no initial modification to C is needed, as the micro-kernel will
#             // accumulate on existing values.

#             // K loop for accumulating `C += A * B` (block by block).
#             // This is the inner-most tiling loop.
#             for (int k_start = 0; k_start < K; k_start += BK) {
#                 int current_BK = std::min(BK, K - k_start); // Actual K dimension of the current blocks.

#                 // --- Pack A_block (current_BM x current_BK) ---
#                 // Data from A[m_start:m_start+current_BM][k_start:k_start+current_BK]
#                 // is copied and rearranged into `packed_A_block_buffer`.
#                 // This packing prepares A for efficient scalar access in the micro-kernel.
#                 // The layout is K-major, MR-minor: `packed_A_block_buffer[k_idx * MR + i_micro]`
#                 // stores `A[m_start + i_block_offset + i_micro][k_start + k_idx]`.
#                 for (int k_idx = 0; k_idx < current_BK; ++k_idx) {
#                     for (int i_block_offset = 0; i_block_offset < current_BM; i_block_offset += MR) {
#                         int mr_rows_in_block = std::min(MR, current_BM - i_block_offset);
#                         // Destination pointer in `packed_A_block_buffer` for current `k` column and `MR` rows.
#                         float* dest_ptr = packed_A_block_buffer + (i_block_offset * current_BK / MR + k_idx) * MR; 
#                         // Source base pointer in original `A` matrix.
#                         const float* src_ptr_base = A + (m_start + i_block_offset) * lda + (k_start + k_idx);

#                         for (int i_micro = 0; i_micro < mr_rows_in_block; ++i_micro) {
#                             dest_ptr[i_micro] = src_ptr_base[i_micro * lda]; // Copy `A[row][k]`
#                         }
#                         // Pad with zeros for `MR` tail if `mr_rows_in_block < MR`.
#                         // This ensures the micro-kernel can always read `MR` elements without out-of-bounds.
#                         for (int i_micro = mr_rows_in_block; i_micro < MR; ++i_micro) {
#                             dest_ptr[i_micro] = 0.0f;
#                         }
#                     }
#                 }

#                 // --- Pack B_block (current_BK x current_BN) ---
#                 // Data from B[k_start:k_start+current_BK][n_start:n_start+current_BN]
#                 // is copied and rearranged into `packed_B_block_buffer`.
#                 // This packing ensures `NR` (vector_width) elements are contiguous, facilitating aligned SIMD loads.
#                 // Layout: K-major, NR-vector-block minor.
#                 // `packed_B_block_buffer[k_idx * BN_ceil_NR * NR + (n_block_idx / NR) * NR + n_elem]`
#                 const int BN_ceil_NR = (BN + NR - 1) / NR; // Number of NR-sized vector blocks needed to cover BN.
#                 for (int k_idx = 0; k_idx < current_BK; ++k_idx) {
#                     for (int n_block_idx = 0; n_block_idx < current_BN; n_block_idx += NR) {
#                         int current_NR_local = std::min(NR, current_BN - n_block_idx);
                        
#                         // Destination pointer in `packed_B_block_buffer` for current `k` row and `NR` columns.
#                         float* dest_ptr = packed_B_block_buffer + k_idx * BN_ceil_NR * NR + (n_block_idx / NR) * NR;
#                         // Source base pointer in original `B` matrix.
#                         const float* src_ptr = B + (k_start + k_idx) * ldb + (n_start + n_block_idx);

#                         // Copy data elements
#                         for (int i = 0; i < current_NR_local; ++i) {
#                             dest_ptr[i] = src_ptr[i];
#                         }
#                         // Pad with zeros for vector tail if `current_NR_local < NR`.
#                         // This prevents reading garbage data if the block's N dimension isn't a multiple of `NR`.
#                         for (int i = current_NR_local; i < NR; ++i) {
#                             dest_ptr[i] = 0.0f;
#                         }
#                     }
#                 }
                
#                 // Loops over MR x NR micro-panels within the current BM x BN block.
#                 // This is where the micro-kernel is invoked.
#                 for (int i_block = 0; i_block < current_BM; i_block += MR) {
#                     // `mr_rows_for_ukernel` is relevant for tail handling; the micro-kernel expects `MR` rows.
#                     // The packed A buffer has already been padded with zeros if `current_BM` is not a multiple of `MR`.
#                     // int mr_rows_for_ukernel = std::min(MR, current_BM - i_block);

#                     for (int j_block = 0; j_block < current_BN; j_block += NR) {
#                         // `nr_cols_for_ukernel` is similarly handled by padding in the packed B buffer.
#                         // int nr_cols_for_ukernel = std::min(NR, current_BN - j_block);

#                         // Pointers to the micro-panels for the micro-kernel call.
#                         // `A_micro_panel` points to the relevant section of `packed_A_block_buffer`.
#                         const float* A_micro_panel = packed_A_block_buffer + (i_block * current_BK / MR);
#                         // `B_micro_panel` points to the relevant section of `packed_B_block_buffer`.
#                         const float* B_micro_panel = packed_B_block_buffer + (j_block / NR) * (current_BK * NR); 
#                         // `C_micro_panel` points directly into the original C matrix.
#                         float* C_micro_panel = C_block_ptr + i_block * ldc + j_block;

#                         // Call the chosen micro-kernel to compute the MR x NR sub-block.
#                         micro_kernel_dispatch(current_BK, A_micro_panel, B_micro_panel, C_micro_panel, ldc);
#                     }
#                 }
#             } // End K loop (accumulating blocks)

#             // Apply alpha: `C_block = alpha * C_block_accumulated`.
#             // This is done after all K-accumulations for the current C-block are complete.
#             if (alpha != 1.0f) {
#                 for (int i = 0; i < current_BM; ++i) {
#                     for (int j = 0; j < current_BN; ++j) {
#                         C_block_ptr[i * ldc + j] *= alpha;
#                     }
#                 }
#             }
#         } // End N loop (over BN blocks)
#     } // End M loop (over BM blocks)

#     // Free the thread-local temporary packed buffers.
#     aligned_free(packed_A_block_buffer);
#     aligned_free(packed_B_block_buffer);
# }


# // =================================================================================================
# // Runtime Dispatcher for my_sgemm (the critical API function)
# // =================================================================================================

# // `my_sgemm` is the public API function that matches the requested signature.
# // It first performs runtime dispatch to select the best available SIMD ISA (AVX-512, AVX2, or Scalar).
# // It then calls `sgemm_tiled_impl` with the chosen `TunableParams`.
# void my_sgemm(int M, int N, int K, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
#     // --- Runtime SIMD ISA Dispatch or use autotuned parameters ---
#     // If `current_tunable_params` is still set to `SCALAR_PARAMS` (its initial default),
#     // it means no autotuning has been performed in `main()` to override it.
#     // In this case, `my_sgemm` performs a default feature detection.
#     if (current_tunable_params.kernel_name == SCALAR_PARAMS.kernel_name) {
#         // Perform CPU feature detection to pick the best available kernel.
#         // The order of checks is crucial: AVX-512 preferred > AVX2 > Scalar.
# #if defined(__AVX512F__) && defined(__FMA__)
#         // Check for AVX-512F (foundation), AVX-512BW (byte/word), AVX-512DQ (doubleword/quadword), AVX-512VL (vector length extensions)
#         // These are commonly available together on modern Intel CPUs supporting AVX-512.
#         if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
#             __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl")) {
#             current_tunable_params = AVX512_PARAMS;
#         } else
# #endif
# #if defined(__AVX2__) && defined(__FMA__)
#         // If AVX-512 is not available or not enabled, check for AVX2 and FMA.
#         if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
#             current_tunable_params = AVX2_PARAMS;
#         } else
# #endif
#         {
#             // Fallback to scalar if no suitable SIMD extensions are found.
#             current_tunable_params = SCALAR_PARAMS;
#         }
#     }

#     // Call the actual tiled implementation with the parameters chosen by either
#     // the autotuner in `main()` or the default runtime dispatcher above.
#     sgemm_tiled_impl(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, current_tunable_params);
# }


# // =================================================================================================
# // Main Function and Test Harness
# // =================================================================================================

# int main(int argc, char* argv[]) {
#     // Default matrix dimensions (can be overridden via CLI)
#     int M = 1024, N = 1024, K = 1024;
#     int seed = 42;             // Default random seed
#     bool check_correctness = false; // Flag to enable correctness check
#     int num_threads = DEFAULT_NUM_THREADS; // Default number of OpenMP threads

#     // Parse command line arguments for M, N, K, seed, threads, and correctness check.
#     for (int i = 1; i < argc; ++i) {
#         std::string arg = argv[i];
#         if (arg == "-m" && i + 1 < argc) {
#             M = std::stoi(argv[++i]);
#         } else if (arg == "-n" && i + 1 < argc) {
#             N = std::stoi(argv[++i]);
#         } else if (arg == "-k" && i + 1 < argc) {
#             K = std::stoi(argv[++i]);
#         } else if (arg == "-s" && i + 1 < argc) {
#             seed = std::stoi(argv[++i]);
#         } else if (arg == "-t" && i + 1 < argc) {
#             num_threads = std::stoi(argv[++i]);
#         } else if (arg == "--check") {
#             check_correctness = true;
#         } else if (arg == "--help" || arg == "-h") {
#             std::cout << "Usage: " << argv[0] << " [-m M] [-n N] [-k K] [-s SEED] [-t THREADS] [--check]" << std::endl;
#             std::cout << "  -m M       Set M dimension (rows of A, C)" << std::endl;
#             std::cout << "  -n N       Set N dimension (cols of B, C)" << std::endl;
#             std::cout << "  -k K       Set K dimension (cols of A, rows of B)" << std::endl;
#             std::cout << "  -s SEED    Set random seed for matrix initialization" << std::endl;
#             std::cout << "  -t THREADS Set number of OpenMP threads" << std::endl;
#             std::cout << "  --check    Enable correctness check against scalar reference" << std::endl;
#             return 0;
#         } else {
#             std::cerr << "Error: Unknown or incomplete argument: " << arg << std::endl;
#             return 1;
#         }
#     }

#     // Set the number of threads for OpenMP.
#     omp_set_num_threads(num_threads);
#     std::cout << "Running GEMM with M=" << M << ", N=" << N << ", K=" << K
#               << ", Threads=" << num_threads << ", Seed=" << seed << std::endl;

#     // Allocate matrices using `std::vector` for heap allocation.
#     // Matrices are represented in row-major order: A (M x K), B (K x N), C (M x N).
#     std::vector<float> A_vec(M * K);
#     std::vector<float> B_vec(K * N);
#     std::vector<float> C_vec(M * N);
#     std::vector<float> C_ref_vec; // Used only if `check_correctness` is true.

#     // Initialize matrices A and B with random float values between -1.0 and 1.0.
#     // Initialize C to zeros.
#     std::mt19937 gen(seed);
#     std::uniform_real_distribution<> dis(-1.0, 1.0);

#     for (int i = 0; i < M * K; ++i) A_vec[i] = static_cast<float>(dis(gen));
#     for (int i = 0; i < K * N; ++i) B_vec[i] = static_cast<float>(dis(gen));
#     for (int i = 0; i < M * N; ++i) C_vec[i] = 0.0f; // C must be initialized to 0 for initial alpha*A*B

#     float alpha = 1.0f; // Standard value for C = A * B
#     float beta = 0.0f;  // Standard value for C = A * B (discards original C values)

#     // Get raw pointers to the underlying data of the vectors.
#     const float* A = A_vec.data();
#     const float* B = B_vec.data();
#     float* C = C_vec.data();

#     // Leading dimensions (lda, ldb, ldc) for row-major matrices are simply their column counts.
#     int lda = K;
#     int ldb = N;
#     int ldc = N;


#     // --- Autotuning Harness (Optional but preferred) ---
#     // This section dynamically evaluates a few candidate `TunableParams` sets
#     // on a smaller problem size to find the best performing configuration for the current runtime environment.
#     std::vector<TunableParams> candidate_params_list;

#     // Populate the list of candidate parameter sets based on available SIMD features.
#     // This prioritizes AVX-512, then AVX2, then scalar.
# #if defined(__AVX512F__) && defined(__FMA__)
#     if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
#         __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("avx512vl")) {
#         candidate_params_list.push_back(AVX512_PARAMS);
#         // Add more AVX-512 specific variants to explore different tiling strategies.
#         candidate_params_list.push_back({192, 192, 64, 8, 16, 4, 64, 16, "AVX-512_alt_BM192"});
#         candidate_params_list.push_back({96, 256, 32, 8, 16, 4, 64, 16, "AVX-512_alt_BN256"});
#     }
# #endif
# #if defined(__AVX2__) && defined(__FMA__)
#     // Add AVX2 candidates if AVX-512 is not available/selected.
#     if (candidate_params_list.empty() && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
#         candidate_params_list.push_back(AVX2_PARAMS);
#         // Add more AVX2 specific variants.
#         candidate_params_list.push_back({128, 96, 48, 8, 8, 4, 32, 8, "AVX2_alt_BN96"});
#         candidate_params_list.push_back({64, 192, 32, 8, 8, 4, 32, 8, "AVX2_alt_BM64"});
#     }
# #endif
#     // If no SIMD kernels were added (e.g., due to unsupported CPU or compile flags), add the scalar fallback.
#     if (candidate_params_list.empty()) {
#         candidate_params_list.push_back(SCALAR_PARAMS);
#     }
    
#     double best_gflops = -1.0;
#     TunableParams chosen_params = SCALAR_PARAMS; // Initialize with scalar as a safe default.

#     if (candidate_params_list.size() > 1) { // Only perform autotuning if there are multiple options to compare.
#         std::cout << "\nStarting autotuning for micro-kernel parameters..." << std::endl;
#         // Use a smaller, fixed-size problem for tuning to keep the autotuning process fast.
#         int tune_M = std::min(M, 256), tune_N = std::min(N, 256), tune_K = std::min(K, 256);
#         std::cout << "  Tuning problem size: " << tune_M << "x" << tune_K << " * " << tune_K << "x" << tune_N << std::endl;

#         // Allocate matrices for the tuning problem.
#         std::vector<float> A_tune(tune_M * tune_K), B_tune(tune_K * tune_N), C_tune(tune_M * tune_N);
#         // Initialize tuning matrices with random data.
#         std::generate(A_tune.begin(), A_tune.end(), [&]() { return static_cast<float>(dis(gen)); });
#         std::generate(B_tune.begin(), B_tune.end(), [&]() { return static_cast<float>(dis(gen)); });

#         for (const auto& p : candidate_params_list) {
#             std::fill(C_tune.begin(), C_tune.end(), 0.0f); // Reset C for each tuning run.

#             // Perform a warm-up run for each candidate configuration.
#             // This helps ensure caches are warm and CPU frequency is boosted.
#             sgemm_tiled_impl(tune_M, tune_N, tune_K, alpha, A_tune.data(), tune_K, B_tune.data(), tune_N, beta, C_tune.data(), tune_N, p);
            
#             double total_time_ms = 0;
#             const int timed_runs = 3; // Average over a few runs for more stable timing.

#             // Execute timed runs for each candidate.
#             for (int r = 0; r < timed_runs; ++r) {
#                 std::fill(C_tune.begin(), C_tune.end(), 0.0f); // Reset C for each run.
#                 auto start = std::chrono::high_resolution_clock::now();
#                 sgemm_tiled_impl(tune_M, tune_N, tune_K, alpha, A_tune.data(), tune_K, B_tune.data(), tune_N, beta, C_tune.data(), tune_N, p);
#                 auto end = std::chrono::high_resolution_clock::now();
#                 total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
#             }
#             double avg_time_ms = total_time_ms / timed_runs;
#             double gflops = (2.0 * tune_M * tune_N * tune_K) / (avg_time_ms * 1e6); // Calculate GFLOP/s.

#             std::cout << "  Candidate " << p.kernel_name << " (BM=" << p.bm << ", BN=" << p.bn << ", BK=" << p.bk << "): "
#                       << avg_time_ms << " ms, " << gflops << " GFLOP/s" << std::endl;

#             // Keep track of the best performing configuration.
#             if (gflops > best_gflops) {
#                 best_gflops = gflops;
#                 chosen_params = p;
#             }
#         }
#         std::cout << "Autotuning complete. Best parameters found: " << chosen_params.kernel_name
#                   << " (BM=" << chosen_params.bm << ", BN=" << chosen_params.bn << ", BK=" << chosen_params.bk << ")" << std::endl;
#     } else if (!candidate_params_list.empty()) {
#         chosen_params = candidate_params_list[0]; // If only one option, just use it.
#         std::cout << "\nOnly one kernel option (" << chosen_params.kernel_name << ") available, skipping autotuning." << std::endl;
#     } else {
#         std::cerr << "\nError: No GEMM kernels compiled or available. This should not happen." << std::endl;
#         return 1;
#     }
    
#     // Set the globally active tunable parameters for `my_sgemm` to the chosen best ones.
#     current_tunable_params = chosen_params;


#     // --- Main Performance Measurement ---
#     std::cout << "\nStarting main GEMM computation with " << current_tunable_params.kernel_name << " kernel..." << std::endl;
#     std::fill(C_vec.begin(), C_vec.end(), 0.0f); // Ensure C is zeroed before the main computation.

#     // Warm-up run for the main problem size.
#     my_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

#     // Timed runs for the main problem, averaged for accuracy.
#     int num_main_runs = 5; 
#     // For very small problems, more runs might be needed to get stable timings.
#     if ((long long)M * N * K < 1024LL * 1024 * 1024 / 8) num_main_runs = 10; 
    
#     total_time_ms = 0;
#     for (int i = 0; i < num_main_runs; ++i) {
#         std::fill(C_vec.begin(), C_vec.end(), 0.0f); // Reset C for each run.
#         auto start = std::chrono::high_resolution_clock::now();
#         my_sgemm(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#         auto end = std::chrono::high_resolution_clock::now();
#         total_time_ms += std::chrono::duration<double, std::milli>(end - start).count();
#     }
#     avg_time_ms = total_time_ms / num_main_runs;

#     gflops = (2.0 * M * N * K) / (avg_time_ms * 1e6); // Calculate GFLOP/s.

#     std::cout << "\nGEMM completed." << std::endl;
#     std::cout << "Average time: " << avg_time_ms << " ms" << std::endl;
#     std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;


#     // --- Correctness Check ---
#     if (check_correctness) {
#         std::cout << "\nRunning correctness check against scalar reference..." << std::endl;
#         C_ref_vec.resize(M * N);
#         std::fill(C_ref_vec.begin(), C_ref_vec.end(), 0.0f);

#         // Compute reference result using the scalar GEMM.
#         sgemm_ref(M, N, K, alpha, A, lda, B, ldb, beta, C_ref_vec.data(), ldc);

#         float max_diff = 0.0f;
#         float max_relative_diff = 0.0f;
#         bool ok = true;

#         // Calculate a dynamic tolerance based on problem size and machine epsilon.
#         // This accounts for floating-point accumulation errors in large computations.
#         float tolerance = 1e-4f; // Baseline tolerance.
#         tolerance = std::max(tolerance, (float)(M * K) * std::numeric_limits<float>::epsilon() * 100.0f); // ~100x epsilon * number of operations.
#         std::cout << "  Using error tolerance (abs or rel): " << tolerance << std::endl;

#         for (int i = 0; i < M * N; ++i) {
#             float diff = std::abs(C_vec[i] - C_ref_vec[i]);
#             max_diff = std::max(max_diff, diff);

#             // Calculate relative difference, avoiding division by zero or tiny numbers.
#             if (std::abs(C_ref_vec[i]) > 1e-9f) { // If reference is not near zero.
#                 max_relative_diff = std::max(max_relative_diff, diff / std::abs(C_ref_vec[i]));
#             } else if (diff > tolerance) { // If reference is near zero but computed value is not.
#                 max_relative_diff = std::max(max_relative_diff, diff / tolerance); // Use tolerance as a pseudo-denominator for relative check.
#             }
            
#             // Check for discrepancy: either absolute difference too large,
#             // or relative difference too large (if reference value is significant).
#             if (diff > tolerance && (std::abs(C_ref_vec[i]) > tolerance ? (diff / std::abs(C_ref_vec[i]) > tolerance) : true)) {
#                 ok = false;
#                 // Optional: print first few mismatches for debugging.
#                 // std::cout << "Mismatch at index " << i << ": C=" << C_vec[i] << ", C_ref=" << C_ref_vec[i] 
#                 //           << ", Abs Diff=" << diff << ", Rel Diff=" << (std::abs(C_ref_vec[i]) > 1e-9f ? diff / std::abs(C_ref_vec[i]) : diff / tolerance) << std::endl;
#                 // if (mismatches_found++ > 5) break; 
#             }
#         }

#         if (ok) {
#             std::cout << "Correctness check PASSED." << std::endl;
#             std::cout << "  Max absolute difference: " << max_diff << std::endl;
#             std::cout << "  Max relative difference: " << max_relative_diff << std::endl;
#         } else {
#             std::cerr << "Correctness check FAILED." << std::endl;
#             std::cerr << "  Max absolute difference: " << max_diff << std::endl;
#             std::cerr << "  Max relative difference: " << max_relative_diff << std::endl;
#         }
#     }

#     return 0;
# }