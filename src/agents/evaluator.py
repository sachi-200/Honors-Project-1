import subprocess
import re
import os
import numpy as np
import random
import csv

def compute_roofline_metrics(performance_metrics, matrix_size):
    """Computes Roofline model metrics based on performance data."""
    CACHE_LINE_SIZE = 64
    peak_compute = 328.4 # GFLOP/s
    peak_memory = 28.9 # GB/s

    if not performance_metrics.get("success"):
        return {"error": "Performance metrics not available for Roofline analysis."}

    cache_misses = performance_metrics.get("cache-misses")
    runtime = performance_metrics.get("execution_time_seconds")
    gflops = performance_metrics.get("gflops")

    if cache_misses is None or runtime is None or gflops is None:
        return {"error": "Insufficient data for Roofline analysis."}

    total_flops = 2 * (matrix_size ** 3)
    total_bytes_moved = cache_misses * CACHE_LINE_SIZE if cache_misses > 0 else 1
    operational_intensity = total_flops / total_bytes_moved
    attained_perf = gflops

    ridge_point = peak_compute / peak_memory
    if operational_intensity < ridge_point:
        bound_by = "Memory Bound"
        max_perf = operational_intensity * peak_memory
    else:
        bound_by = "Compute Bound"
        max_perf = peak_compute

    efficiency = (attained_perf / max_perf) * 100 if max_perf > 0 else 0.0

    return {
        "success": True,
        "runtime_seconds": runtime,
        "cache_misses": cache_misses,
        "data_moved_GB": total_bytes_moved / 1e9,
        "operational_intensity": operational_intensity,
        "attained_performance_GFLOPS": attained_perf,
        "ridge_point": ridge_point,
        "bound_by": bound_by,
        "max_performance_GFLOPS": max_perf,
        "efficiency_percent": efficiency,
        "peak_compute_GFLOPS": peak_compute,
        "peak_memory_GBPS": peak_memory,
    }

def run_intel_advisor_roofline(executable_path):
    """Runs Intel Advisor Roofline analysis on the given executable."""
    project_dir = "./advisor_project"
    csv_output = os.path.join(project_dir, "roofline_report.csv")
    os.makedirs(project_dir, exist_ok=True)

    print("\tRunning Intel Advisor Roofline analysis")
    try:
        subprocess.run(
            ["advisor", "--report=roofline",
             f"--project-dir={project_dir}",
             "--format=csv",
             f"--report-output={csv_output}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        return {"error": f"Intel Advisor analysis failed: {e.stderr}"}

    if not os.path.exists(csv_output):
        return {"success": False, "error": "Roofline report CSV not found."}

    data = []
    with open(csv_output, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)

    if not data:
        return {"success": False, "error": "No data found in Roofline report."}

    try:
        row = data[0]
        attained_perf = float(row.get("GFLOPS", 0.0))
        operational_intensity = float(row.get("Operational Intensity (FLOPs/Byte)", 0.0))
        bound_type = row.get("Roofline Limit Type", "Unknown")

        result = {
            "success": True,
            "attained_performance_GFLOPS": attained_perf,
            "operational_intensity": operational_intensity,
            "bound_by": bound_type,
            "source": "Intel Advisor",
        }
        return result
    except Exception as e:
        return {"success": False, "error": f"Error parsing Roofline report: {e}"}

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

        if np.allclose(C_from_cpp, C_expected, rtol=1e-3, atol=1e-4):
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

def run_and_analyze(executable_path, matrix_size=2048):
    """Runs the code and extracts performance metrics."""
    events = [
        "cycles",
        "instructions",
        "cache-references",
        "cache-misses",
        "branches",
        "branch-misses",
    ]
    perf_args = [str(matrix_size)] * 3
    target_command = [f"./{executable_path}"] + perf_args

    perf_command = [
        "perf",
        "stat",
        "-e",
        ",".join(events),
    ] + target_command

    try:
        result = subprocess.run(perf_command, capture_output=True, text=True, timeout=1200)
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

def evaluate_code(code_string, matrix_size, system_type):
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
    performance_metrics = run_and_analyze("workspace/a.out", matrix_size=matrix_size)
    feedback["performance"] = performance_metrics

    if system_type.lower() == "intel":
        print("\t 4) Performing Roofline Analysis using Intel Advisor")
        roofline_results = run_intel_advisor_roofline("workspace/a.out")
    else:
        print("\t 4) Performing Roofline Analysis using perf data")
        roofline_results = compute_roofline_metrics(performance_metrics, matrix_size)
    feedback["roofline"] = roofline_results

    return feedback
