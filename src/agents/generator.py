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

        # Base signature template
        def get_signature(func_name: str) -> str:
            return f"""
            void {func_name}(const float* A, const float* B, float* C,
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

        if architecture not in architecture_details:
            raise ValueError(f"Unknown architecture: {architecture}")

        arch_info = architecture_details[architecture]

        signatures_to_generate = { "gemm_scalar": get_signature("gemm_scalar") }
        main_optimized_call = "gemm_scalar"

        if "AVX-512" in arch_info:
            signatures_to_generate["gemm_avx512"] = get_signature("gemm_avx512")
            main_optimized_call = "gemm_avx512"
        elif "AVX2" in arch_info:
            signatures_to_generate["gemm_avx2"] = get_signature("gemm_avx2")
            main_optimized_call = "gemm_avx2"
        elif "AVX" in arch_info:
            signatures_to_generate["gemm_avx"] = get_signature("gemm_avx")
            main_optimized_call = "gemm_avx"

        function_signatures = "\n".join(signatures_to_generate.values())

        full_prompt = f"""
            You are an expert **C++** programmer specializing in **CPU-optimized dense matrix multiplication (GEMM)** for x86-64.
            Generate a **single, complete C++ source file** implementing the requested GEMM functions, specifically tuned for the CPU below.

            **Target Platform (Host CPU):**
            {arch_info}

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
            - Provide example compile commands for the *specific optimized kernel* being generated.
                - If generating `gemm_avx512`: `g++ -O3 -march=x86-64-v3 -mavx512f -mfma -fopenmp gemm.cpp -o gemm`
                - If generating `gemm_avx2`:    `g++ -O3 -march=x86-64-v2 -mavx2 -mfma -fopenmp gemm.cpp -o gemm`
                - If generating `gemm_avx`:     `g++ -O3 -march=native -mavx -fopenmp gemm.cpp -o gemm`
                - If generating `gemm_scalar` only: `g++ -O3 -march=native -fopenmp gemm.cpp -o gemm`

            3) **SIMD Implementation:**
            - Implement **only the kernels requested** in the function signatures.
            - `gemm_scalar` **must** be implemented as a simple, correct C++ reference.
            - If `gemm_avx512` is requested, implement it using AVX-512 intrinsics (and guard with `#if defined(__AVX512F__)`).
            - If `gemm_avx2` is requested, implement it using AVX2+FMA intrinsics (and guard with `#if defined(__AVX2__)`).
            - If `gemm_avx` is requested, implement it using AVX intrinsics (and guard with `#if defined(__AVX__)`).
            - **Do NOT implement runtime dispatch.** The `main` function will call the specific, optimized function directly.

            4) **Parallelization:**
            - Use **OpenMP** for outer-loop parallelism over tiles/blocks (M×N) in the *optimized* kernel (`{main_optimized_call}`).
            - The `gemm_scalar` reference implementation can be single-threaded.
            - Choose a sane default schedule (e.g., static or guided) and justify in comments.

            5) **Blocking/Tiling & Memory:**
            - Implement **cache-aware tiling** with tunable tile sizes **BM, BN, BK** in the optimized kernel.
            - Favor row-major storage; document your convention.
            - Coalesce accesses and prefetch when beneficial (`_mm_prefetch`).

            6) **Autotuning Parameters (Mandatory to expose as constants at the top or via constexpr):**
            - **BM, BN, BK** (tile sizes) — explore values like: {{32, 48, 64, 96, 128, 192}}.
            - **UNROLL_K** — inner K unroll factor (e.g., {{1, 2, 4, 8}}).
            - **NUM_THREADS** — optionally allow setting via OMP or environment; provide a CLI flag or environment read (`OMP_NUM_THREADS`).

            7) **Edge Handling & Correctness:**
            - Correctly handle M, N, K not divisible by tile or vector widths (tail processing) in all kernels.
            - Avoid UB: masks or scalar tails for leftover columns/rows.
            - The `gemm_scalar` implementation will be used for correctness checking.

            8) **Function Signatures (CRITICAL):**
            - Define EACH function with EXACTLY the signature(s) listed in {function_signatures}.
            - **Do NOT** change parameter names, counts, order, or const-ness.
            - All function calls must match their definitions exactly.
            - If the interface includes strides/leading dimensions (lda/ldb/ldc), use them correctly.

            9) **CLI / Demo Main (CRITICAL LOGIC):**
            - Provide a `main()` that parses CLI args: M N K, and an optional **--dump-matrices** flag.
            - The `main` function must detect this flag to determine its behavior.
            - Include the helper function `void write_matrix_to_file(const std::string& filename, const float* matrix, int rows, int cols, int ld)`.

            - **IF `--dump-matrices` IS PRESENT (Test Mode):**
                - Allocate A, B, C (for optimized), and C_ref (for scalar).
                - Initialize A, B (random), C (zero), C_ref (zero).
                - Call `write_matrix_to_file` to save A to `workspace/A.txt` and B to `workspace/B.txt`.
                - Call `gemm_scalar(...)` to compute the reference result in `C_ref`.
                - Call the **optimized function (`{main_optimized_call}`)** to compute the result in `C`.
                - Call `write_matrix_to_file` to save the optimized result `C` to `workspace/C.txt`.
                - Perform an internal correctness check by comparing `C` and `C_ref` and print "Internal check: PASSED" or "Internal check: FAILED".

            - **IF `--dump-matrices` IS NOT PRESENT (Perf Mode):**
                - Allocate A, B, C.
                - Initialize A, B (random), C (zero).
                - **ONLY** call the **optimized function (`{main_optimized_call}`)** to compute `C`.
                - **Do NOT** call `gemm_scalar`.
                - **Do NOT** write any files.
                - **Do NOT** perform the internal correctness check.
                - (Optional) You may still include an internal timer around the optimized call and print the GFLOPS, as this is good for debugging, but the Python evaluator will use its own timer.

            10) **Documentation & Comments:**
            - Briefly explain blocking choices, threading strategy, and the specific ISA kernel.
            - Comment the intrinsics code (register layout, accumulation pattern).
            - Mention cache levels (L1/L2/L3) and how tiles aim to fit/reuse.

            **Performance Hints (Follow where feasible):**
            - Pack micro-panels of B for better locality; consider packing A or both if time permits.
            - Favor **register blocking** for the inner micro-kernel (e.g., 8×8 or 16×8 accumulators for AVX-512/AVX2).
            - Use FMA where available; minimize gathers/scatters; prefer contiguous loads/stores.
            - Guard intrinsics with `#if defined(__AVX512F__)` / `#if defined(__AVX2__)` as needed, so the file still compiles even if the user passes the wrong `-march` flag (though it won't run the kernel).

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
