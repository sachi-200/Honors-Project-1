#!/usr/bin/env python3
import os
import subprocess
import csv

# ==========================================================
# Standalone Intel Advisor Roofline Test
# ==========================================================
# Usage:
#   python3 test_intel_advisor_roofline.py gemm_simd_main.cpp
#
# This script:
#   1. Compiles the given C++ source with OpenMP and AVX.
#   2. Runs Intel Advisor Roofline analysis.
#   3. Parses the CSV report and prints key metrics.
# ==========================================================


def run_cmd(cmd, check=True):
    """Run a shell command with nice output."""
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    if check and result.returncode != 0:
        print("❌ Command failed:\n", result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def compile_program(src_file, exe_file="gemm_test"):
    """Compile C++ source with -O3, -fopenmp, and -march=native."""
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")

    print(f"🔧 Compiling {src_file} ...")
    run_cmd(["g++", "-O3", "-g", "-march=native", "-fopenmp", "-o", exe_file, src_file])
    print(f"✅ Compilation successful -> {exe_file}")
    return exe_file


def run_intel_advisor(exe_path, project_dir="./advisor_proj", csv_output="roofline.csv"):
    """Run Intel Advisor Roofline analysis and return CSV path."""
    os.makedirs(project_dir, exist_ok=True)

    print("🚀 Running Intel Advisor Roofline analysis (collection)...")
    run_cmd(["advisor", "--collect=roofline",
             f"--project-dir={project_dir}",
             "--", exe_path])

    print("📊 Generating Roofline report (CSV)...")
    run_cmd(["advisor", "--report=roofline",
             f"--project-dir={project_dir}",
             "--format=csv",
             f"--report-output={csv_output}"])

    if not os.path.exists(csv_output):
        raise FileNotFoundError(f"❌ CSV report not found at {csv_output}")

    print(f"✅ Roofline report generated: {csv_output}")
    return csv_output


def parse_roofline_csv(csv_file):
    """Parse Intel Advisor Roofline CSV and extract key metrics."""
    print("🔍 Parsing CSV results...")
    data = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    if not data:
        raise ValueError("No data found in Advisor CSV.")

    # Try to extract relevant columns (names vary by version)
    row = data[0]
    key_map = {
        "GFLOPS": ["GFLOPS", "GFLOP/s", "GFLOPS (FLOPs/sec)"],
        "Operational Intensity": [
            "Operational intensity (FLOPs/Byte)",
            "Operational intensity (FLOP/Byte)",
            "Operational Intensity"
        ],
        "Bound": ["Roofline Limit Type", "Bound Type"]
    }

    def find_value(keys):
        for k in keys:
            if k in row:
                return row[k]
        return None

    gflops = find_value(key_map["GFLOPS"])
    op_intensity = find_value(key_map["Operational Intensity"])
    bound = find_value(key_map["Bound"])

    print("\n===============================")
    print("📈 Parsed Roofline Results")
    print("===============================")
    print(f"GFLOPs/sec:          {gflops}")
    print(f"Operational Intensity: {op_intensity}")
    print(f"Bound Type:           {bound}")
    print("===============================\n")

    return {
        "GFLOPS": gflops,
        "Operational Intensity": op_intensity,
        "Bound Type": bound,
    }


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 test_intel.py <source.cpp>")
        raise SystemExit(1)

    src_file = sys.argv[1]
    exe_file = "gemm_test"
    project_dir = "./advisor_proj"
    csv_output = "roofline.csv"

    compile_program(src_file, exe_file)
    csv_path = run_intel_advisor(f"{os.getcwd()}/{exe_file}", project_dir, csv_output)
    parse_roofline_csv(csv_path)

    print("✅ Done. To view interactive Roofline chart:")
    print(f"   advisor-gui --project-dir={project_dir} &")


if __name__ == "__main__":
    main()
