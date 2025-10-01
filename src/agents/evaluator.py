import subprocess
import re

def compile_code(code_string, output_filename="workspace/a.out"):
    """Compiles a string of C++ code with OpenMP enabled."""
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


def run_unit_tests(executable_path, test_cases):
    """Runs the compiled code against a list of test cases."""
    results = []
    for test in test_cases:
        input_data, expected_output = test
        run_command = [f"./{executable_path}"]

        result = subprocess.run(
            run_command,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            passed = output == expected_output
            results.append({"passed": passed, "output": output})
        else:
            results.append({"passed": False, "error": result.stderr})

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

    perf_command = [
        "perf",
        "stat",
        "-e",
        ",".join(events),
        f"./{executable_path}",
    ]

    result = subprocess.run(
        perf_command, capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        return {"success": False, "error": result.stderr}

    output = result.stderr
    metrics = {"success": True}

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

    return metrics

def evaluate_code(code_string, test_cases):
    """The main wrapper function for the evaluator."""
    feedback = {}

    compile_result = compile_code(code_string)
    feedback["compilation"] = compile_result
    if not compile_result["success"]:
        return feedback

    # run_results = run_unit_tests("workspace/a.out", test_cases)
    # feedback["tests"] = run_results
    # if not all(test["passed"] for test in run_results):
    #     return feedback

    # performance_metrics = run_and_analyze("workspace/a.out")
    # feedback["performance"] = performance_metrics

    return feedback