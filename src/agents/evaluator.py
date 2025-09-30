import subprocess
import re

def compile_code(code_string, output_filename="workspace/a.out"):
    """Compiles a string of C++ code with OpenMP enabled."""
    with open("temp.cpp", "w") as f:
        f.write(code_string)

    compile_command = [
        "g++",
        "-fopenmp",
        "temp.cpp",
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
    pass

def evaluate_code(code_string, test_cases):
    """The main wrapper function for the evaluator."""
    feedback = {}

    compile_result = compile_code(code_string)
    feedback["compilation"] = compile_result
    if not compile_result["success"]:
        return feedback

    run_results = run_unit_tests("workspace/a.out", test_cases)
    feedback["tests"] = run_results

    performance_metrics = run_and_analyze("workspace/a.out")
    feedback["performance"] = performance_metrics

    return feedback