# import os
# import subprocess
# import sys

# CPP_SOURCE = "mkl_gemm_benchmark.cpp"
# EXECUTABLE = "./mkl_gemm_benchmark"
# MATRIX_SIZES = [128, 256, 512, 1024, 2048, 4096]

# def compile_cpp_code():
#     """Compiles the C++ source file using g++ and MKL flags."""
#     print("--- Compiling C++ benchmark code ---")

#     compile_command = [
#         "g++", CPP_SOURCE,
#         "-o", EXECUTABLE,
#         "-I" + os.environ.get("MKLROOT", "") + "/include",
#         "-L" + os.environ.get("MKLROOT", "") + "/lib/intel64",
#         "-lmkl_rt",
#         "-fopenmp",
#         "-O3",
#         "-std=c++17"
#     ]

#     print(f"Executing: {' '.join(compile_command)}")

#     try:
#         subprocess.run(compile_command, check=True, capture_output=True, text=True)
#         print("Compilation successful.\n")
#     except subprocess.CalledProcessError as e:
#         print("--- COMPILATION FAILED ---", file=sys.stderr)
#         print(f"Return Code: {e.returncode}", file=sys.stderr)
#         print(f"Stdout:\n{e.stdout}", file=sys.stderr)
#         print(f"Stderr:\n{e.stderr}", file=sys.stderr)
#         sys.exit(1)
#     except FileNotFoundError:
#         print("Error: g++ not found. Is it installed and in your PATH?", file=sys.stderr)
#         sys.exit(1)


# def run_benchmarks():
#     """Runs the compiled C++ executable for each matrix size and prints results."""
#     results = {}

#     print("--- Running Benchmarks ---")
#     for size in MATRIX_SIZES:
#         print(f"Benchmarking size: {size}x{size}...")
#         command = [EXECUTABLE, str(size)]

#         try:
#             result = subprocess.run(command, check=True, capture_output=True, text=True)
#             gflops = float(result.stdout.strip())
#             results[size] = gflops
#             print(f"  -> Attained: {gflops:.2f} GFLOPS/s")
#         except subprocess.CalledProcessError as e:
#             print(f"Execution failed for size {size}:", file=sys.stderr)
#             print(e.stderr, file=sys.stderr)
#         except ValueError:
#             print(f"Could not parse GFLOPS value for size {size}.", file=sys.stderr)

#     print("\n" + "="*60)
#     print("               Benchmark Summary (SGEMM via C++)")
#     print("="*60)
#     print(f"{'Matrix Size':<20} | {'Attained GFLOPS/s':<20}")
#     print("-" * 43)
#     for size, gflops in results.items():
#         print(f"{f'{size}x{size}':<20} | {gflops:<20.2f}")
#     print("="*60)

# def main():
#     if 'MKLROOT' not in os.environ:
#         print("Warning: MKLROOT environment variable is not set.", file=sys.stderr)
#         print("Please source the MKL or oneAPI setvars.sh script first.", file=sys.stderr)

#     compile_cpp_code()
#     run_benchmarks()

# if __name__ == "__main__":
#     main()

import os
import subprocess
import sys


CPP_SOURCE = "mkl_gemm_benchmark.cpp"
EXECUTABLE = "./mkl_gemm_benchmark"
MATRIX_SIZES = [128, 256, 512, 1024, 2048, 4096]

def compile_cpp_code():
    """Compiles the C++ source file using g++ and MKL flags."""
    print("--- Compiling C++ benchmark code ---")

    compile_command = [
        "g++", CPP_SOURCE,
        "-o", EXECUTABLE,
        "-I" + os.environ.get("MKLROOT", "") + "/include",
        "-L" + os.environ.get("MKLROOT", "") + "/lib/intel64",
        "-lmkl_rt",
        "-fopenmp",
        "-O3",
        "-march=native", # Added: Tells GCC to optimize for Zen 4/5
        "-std=c++17"
    ]

    print(f"Executing: {' '.join(compile_command)}")

    try:
        subprocess.run(compile_command, check=True, capture_output=True, text=True)
        print("Compilation successful.\n")
    except subprocess.CalledProcessError as e:
        print("--- COMPILATION FAILED ---", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)
        print(f"Stdout:\n{e.stdout}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: g++ not found. Is it installed and in your PATH?", file=sys.stderr)
        sys.exit(1)


def run_benchmarks():
    """Runs the compiled C++ executable for each matrix size and prints results."""
    results = {}

    print("--- Running Benchmarks ---")
    for size in MATRIX_SIZES:
        print(f"Benchmarking size: {size}x{size}...")
        command = [EXECUTABLE, str(size)]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            gflops = float(result.stdout.strip())
            results[size] = gflops
            print(f"  -> Attained: {gflops:.2f} GFLOPS/s")
        except subprocess.CalledProcessError as e:
            print(f"Execution failed for size {size}:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        except ValueError:
            print(f"Could not parse GFLOPS value for size {size}.", file=sys.stderr)

    print("\n" + "="*60)
    print("               Benchmark Summary (SGEMM via C++)")
    print("="*60)
    print(f"{'Matrix Size':<20} | {'Attained GFLOPS/s':<20}")
    print("-" * 43)
    for size, gflops in results.items():
        print(f"{f'{size}x{size}':<20} | {gflops:<20.2f}")
    print("="*60)

def main():
    if 'MKLROOT' not in os.environ:
        print("Warning: MKLROOT environment variable is not set.", file=sys.stderr)
        print("Please source the MKL or oneAPI setvars.sh script first.", file=sys.stderr)

    compile_cpp_code()
    run_benchmarks()

if __name__ == "__main__":
    main()
