import requests
import time

class GeneratorAgent:
    def __init__(self):
        self.API_KEY = "AIzaSyAxtd0l68vktXBcKmGIZV8Vk-83vsqALd8"
        self.MODEL = "gemini-2.5-flash"
        self.URL = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent?key={self.API_KEY}"

    def _ask_gemini(self, prompt: str, retries=3, backoff_factor=0.5) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        for i in range(retries):
            try:
                response = requests.post(self.URL, headers=headers, json=data)
                response.raise_for_status()
                resp_json = response.json()
                return resp_json["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.HTTPError as e:
                if 500 <= e.response.status_code < 600:
                    print(f"Server error ({e.response.status_code}), retrying in {backoff_factor * (2 ** i)} seconds...")
                    time.sleep(backoff_factor * (2 ** i))
                else:
                    raise

    def generate_code(self, history: dict, architecture: str) -> str:
        history_text = ""
        for i, entry in history.items():
            history_text += f"\n--- Iteration {i} ---\n"
            history_text += f"Code:\n{entry['code']}\n"
            history_text += f"Feedback:\n{entry['feedback']}\n"

        function_signatures = """
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
        """

        architecture_details = {
            "AMD-Ryzen-7-6800HS": """
            - Architecture: x86_64 (AMD Ryzen 7 6800HS)
            - SIMD ISA: AVX, AVX2, and FMA
            - Threads: 16 logical CPUs (8 cores, SMT/HT=2)
            - OS: Linux (assume recent GCC/Clang toolchain)
            """,
            "Intel-i7-1195G7": """
            - Architecture: x86_64 (Intel 11th Gen Core i7-1195G7)
            - SIMD ISA: AVX2, FMA, and AVX-512
            - Threads: 8 logical CPUs (4 cores, SMT/HT=2)
            - OS: Linux (assume recent GCC/Clang toolchain)
            """,
        }

        full_prompt = f"""
            You are an expert **C++** programmer specializing in **CPU-optimized dense matrix multiplication (GEMM)** for x86-64.
            Generate a **single, complete C++ source file** implementing the requested GEMM with **SIMD intrinsics** and **multi-threading**, tuned for the CPU below.

            **Target Platform (Host CPU):**
            {architecture_details[architecture]}

            **CRITICAL FUNCTION INFORMATION:**
            Based on analysis, the implementation requires these EXACT function signatures (C++):
            {function_signatures}

            **Output Requirements:**
            1) **Language & File:**
            - Output a single, self-contained **.cpp** file (no external headers beyond the standard library and intrinsics).
            - C++17 or later.
            - Include compile instructions as a comment at the top (e.g., GCC/Clang).
            - No CUDA/ROCm/Triton—**pure CPU** implementation.
            - Your entire response must be only the raw C++ source code.
            - DO NOT wrap the code in Markdown code fences (e.g., ```cpp or ```).
            - Do NOT include any explanations or text outside the code.
            - The output must be a single, self-contained block of text that can be saved directly to a .cpp file and compiled without any modification.

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
                - Parses optional CLI args: M N K, seed, threads, and an optional **--dump-matrices** flag.
                - Allocates and initializes A, B (random), C (zero).
                - **If the `--dump-matrices` flag is present, it must write matrices A and B to `workspace/A.txt` and `workspace/B.txt` respectively.**
                - Calls the top-level API from {function_signatures}, which internally dispatches to AVX-512/AVX2/scalar.
                - **After the computation, if the `--dump-matrices` flag is present, it must write matrix C to `workspace/C.txt`.**
                - Prints a short timing report (ms, effective GFLOP/s = 2*M*N*K / time).
            - Include a helper function `void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld)` that writes a matrix to a text file in a space-separated format (one row per line). This function must correctly handle the leading dimension `ld`.

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

            Here is the history of previous attempts and feedback:
            {history_text}
            Now, based on this history, generate an improved version of the code.
        """

        return self._ask_gemini(full_prompt)
