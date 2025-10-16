import numpy as np
import subprocess
import argparse
import os
import sys

def run_command(command):
    """Executes a command and returns its stdout and stderr."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: '{' '.join(e.cmd)}'")
        print(f"Return code: {e.returncode}")
        print(f"Stderr:\n{e.stderr}")
        print(f"Stdout:\n{e.stdout}")
        sys.exit(1)

def main(args):
    """Main verification function."""
    cpp_file = args.file
    n = args.size
    executable_name = "verify_executable"

    print(f"--- Correctness Verification for {cpp_file} ---")

    # 1. Generate random matrices with NumPy
    print(f"[Step 1/4] Generating {n}x{n} random matrices A and B...")
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)

    # 2. Calculate the "golden" result with NumPy
    print("[Step 2/4] Calculating reference result C_golden = A @ B...")
    C_golden = A @ B

    # 3. Save input matrices to binary files for the C++ program to read
    A.tofile('A.bin')
    B.tofile('B.bin')

    # 4. Compile and run the C++ program
    print(f"[Step 3/4] Compiling and running {cpp_file}...")
    compiler = "g++"
    compile_command = [
        compiler, "-O3", "-fopenmp", "-mavx2", "-mfma",
        cpp_file,
        "-o", executable_name
    ]
    run_command(compile_command)

    run_command([f"./{executable_name}", str(n)])

    # 5. Load the result from the C++ program
    print("[Step 4/4] Loading C++ result and comparing against golden reference...")
    try:
        C_test = np.fromfile('C.bin', dtype=np.float32).reshape(n, n)
    except FileNotFoundError:
        print("\nError: C++ program did not generate the 'C.bin' output file.")
        sys.exit(1)

    # 6. Compare the results
    # np.allclose handles floating-point comparisons with a tolerance.
    if np.allclose(C_golden, C_test):
        print("\n-------------------")
        print("✅ SUCCESS: The output of the C++ program is correct!")
        print("-------------------")
    else:
        print("\n-------------------")
        print("❌ FAILURE: The output of the C++ program is INCORRECT.")
        print("-------------------")
        # Optional: Print diff for debugging
        diff = np.abs(C_golden - C_test)
        print(f"Maximum absolute error: {np.max(diff)}")

    # 7. Cleanup
    for f in ['A.bin', 'B.bin', 'C.bin', executable_name]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correctness verification script for GEMM programs.")
    parser.add_argument('file', type=str, help="Path to the C++ source file to verify.")
    parser.add_argument('--size', type=int, default=256, help="The matrix size N for the test. Default is smaller for faster checks.")

    args = parser.parse_args()
    main(args)