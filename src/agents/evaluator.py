import subprocess
import re
import os
import numpy as np
import random

def compile_code(code_string, output_filename="workspace/a.out"):
    """Compiles a string of C++ code with OpenMP enabled."""
    os.makedirs("workspace", exist_ok=True)
    with open("workspace/temp.cpp", "w") as f:
        f.write(code_string)

    compile_command = [
        "g++",
        "-O3",
        "-std=c++17",
        "-Wall",
        "-march=native",
        "-fopenmp",
        "workspace/temp.cpp",
        "-o",
        output_filename,
    ]

    result = subprocess.run(
        compile_command, capture_output=True, text=True
    )

    if result.returncode == 0:
        return {"success": True, "output": result.stdout}
    else:
        return {"success": False, "error": result.stderr}

def verify_correctness_from_files(a_file, b_file, c_file):
    """
    Reads matrices A, B, and the C++ result C from files,
    computes the correct result with NumPy, and compares them.
    """
    try:
        A = np.loadtxt(a_file, dtype=np.float32)
        B = np.loadtxt(b_file, dtype=np.float32)
        C_from_cpp = np.loadtxt(c_file, dtype=np.float32)

        # Handle 1-dimensional case from loadtxt
        if A.ndim == 1: A = A.reshape(1, -1)
        if B.ndim == 1: B = B.reshape(1, -1)
        if C_from_cpp.ndim == 1: C_from_cpp = C_from_cpp.reshape(1, -1)

        C_expected = A @ B

        if np.allclose(C_from_cpp, C_expected, rtol=1e-4, atol=1e-5):
            return {"passed": True}
        else:
            diff = np.abs(C_from_cpp - C_expected)
            error_msg = (
                f"NumPy result did not match C++ result. "
                f"Max absolute difference: {np.max(diff)}"
            )
            return {"passed": False, "error": error_msg}

    except FileNotFoundError as e:
        return {"passed": False, "error": f"Could not find matrix file: {e.filename}"}
    except Exception as e:
        return {"passed": False, "error": f"An unexpected error occurred during verification: {e}"}

def run_unit_tests(executable_path):
    """Runs the compiled code against a list of test cases."""
    results = []

    test_cases = [
        (32, 32, 32),
        (64, 128, 256),
        (127, 31, 63),
        (512, 512, 512),
        (random.randint(1000, 1500), random.randint(1000, 1500), random.randint(1000, 1500))
    ]

    for m, n, k in test_cases:
        args = [str(m), str(n), str(k), "--dump-matrices"]
        run_command = [f"./{executable_path}"] + args
        test_result = {"args": args}

        try:
            proc = subprocess.run(run_command, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                test_result.update({"passed": False, "error": f"Execution failed: {proc.stderr}"})
                results.append(test_result)
                continue

            verification = verify_correctness_from_files("workspace/A.txt", "workspace/B.txt", "workspace/C.txt")
            test_result.update(verification)
            results.append(test_result)

        except subprocess.TimeoutExpired:
            test_result.update({"passed": False, "error": "Execution timed out"})
            results.append(test_result)

    return results

def run_and_analyze(executable_path):
    """Runs the code and extracts performance metrics."""
    events = [
        "cycles",
        "instructions",
        "cache-references",
        "cache-misses",
        "branches",
        "branch-misses",
    ]
    perf_args = ["2048", "2048", "2048"]
    target_command = [f"./{executable_path}"] + perf_args

    perf_command = [
        "perf",
        "stat",
        "-e",
        ",".join(events),
    ] + target_command

    try:
        result = subprocess.run(perf_command, capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        return {"success": False, "error": "perf tool not found. Please install it to measure performance."}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Performance measurement timed out."}

    if result.returncode != 0:
        return {"success": False, "error": result.stderr}

    output = result.stderr
    metrics = {"success": True, "args": perf_args}

    for event in events:
        pattern = rf"^\s*([\d,]+)\s+{re.escape(event)}"
        match = re.search(pattern, output, re.MULTILINE)
        if match:
            value = int(match.group(1).replace(",", ""))
            metrics[event] = value
        else:
            metrics[event] = None

    time_match = re.search(r"^\s*([\d.]+)\s+seconds time elapsed", output, re.MULTILINE)
    if time_match:
        metrics["execution_time_seconds"] = float(time_match.group(1))

        m, n, k = map(int, perf_args)
        if metrics["execution_time_seconds"] > 0:
            metrics["gflops"] = (2 * m * n * k) / (metrics["execution_time_seconds"] * 1e9)
        else:
            metrics["gflops"] = 0.0

    return metrics

def evaluate_code(code_string):
    """The main wrapper function for the evaluator."""
    feedback = {}

    print("\t 1) Compiling code")
    compile_result = compile_code(code_string)
    feedback["compilation"] = compile_result
    if not compile_result["success"]:
        return feedback

    print("\t 2) Running unit tests")
    run_results = run_unit_tests("workspace/a.out")
    feedback["tests"] = run_results
    if not all(test["passed"] for test in run_results):
        return feedback

    print("\t 3) Analyzing performance")
    performance_metrics = run_and_analyze("workspace/a.out")
    feedback["performance"] = performance_metrics

    return feedback
