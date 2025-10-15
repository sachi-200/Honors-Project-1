import subprocess
import re
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

# --- Configuration ---
CACHE_LINE_SIZE = 64

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

def parse_perf_output(perf_stderr):
    """Parses the stderr from 'perf stat' to extract key metrics."""
    metrics = {
        'runtime': None,
        'cache_misses': None
    }

    # Regex patterns for generic, portable perf events.
    # Runtime is now based on wall-clock time ('time elapsed'), which is correct for performance calculations.
    patterns = {
        'runtime': r'(\S+)\s+seconds\s+time elapsed',
        'cache_misses': r'(\S+)\s+cache-misses'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, perf_stderr)
        if match:
            value_str = match.group(1).replace(',', '')
            if value_str.replace('.', '', 1).isdigit():
                metrics[key] = float(value_str)
            else:
                print(f"Warning: Could not parse value for '{key}'. Found '{value_str}'. This event might not be supported.")
                metrics[key] = 0.0

    if None in metrics.values():
        print("Error: Could not parse all required metrics from perf output.")
        print("--- Perf Output ---")
        print(perf_stderr)
        print("-------------------")
        sys.exit(1)

    return metrics

def plot_roofline(oi, perf_gflops, peak_compute, peak_memory, file_path, history_data):
    """Generates and saves a Roofline plot and a performance history bar chart."""

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 14),
        gridspec_kw={'height_ratios': [3, 1.5]}
    )
    fig.suptitle(f'Analysis for {os.path.basename(file_path)}', fontsize=20)

    # --- Subplot 1: Roofline Plot ---
    oi_vals = np.logspace(-1, 3, 100)
    memory_bound = oi_vals * peak_memory
    compute_bound = np.full_like(oi_vals, peak_compute)
    roof = np.minimum(memory_bound, compute_bound)

    ax1.loglog(oi_vals, roof, 'k-', label='Hardware Roof')
    ax1.scatter(oi, perf_gflops, color='red', s=150, zorder=5, label='Current Performance')

    ax1.annotate(f'({oi:.2f}, {perf_gflops:.2f})', (oi, perf_gflops), textcoords="offset points", xytext=(0,15), ha='center', fontsize=12)

    ax1.set_title('Roofline Model', fontsize=16)
    ax1.set_xlabel('Operational Intensity [FLOPs/Byte]', fontsize=12)
    ax1.set_ylabel('Performance [GFLOP/s]', fontsize=12)
    ax1.grid(True, which="both", ls="--", alpha=0.6)
    ax1.legend(fontsize=12)
    ax1.set_ylim(bottom=1)

    # --- Subplot 2: Performance History Bar Chart ---
    sorted_history = sorted(history_data.items(), key=lambda item: item[1])
    versions = [item[0] for item in sorted_history]
    gflops_values = [item[1] for item in sorted_history]

    bars = ax2.bar(versions, gflops_values, color='skyblue')
    ax2.set_title('Performance History', fontsize=16)
    ax2.set_ylabel('Attained GFLOP/s', fontsize=12)
    ax2.tick_params(axis='x', labelrotation=45, labelsize=10)

    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    ax2.axhline(y=peak_compute, color='r', linestyle='--', label=f'Peak Compute ({peak_compute} GFLOP/s)')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_filename = f"roofline_analysis_{os.path.basename(file_path).replace('.c', '').replace('.cpp', '')}.png"
    plt.savefig(plot_filename)
    print(f"\n[Plot] Analysis plot saved to '{plot_filename}'")


def main(args):
    """Main analysis function."""
    c_program_path = args.file
    matrix_size = args.size
    peak_compute = args.peak_compute
    peak_memory = args.peak_memory
    threads = os.cpu_count() if args.threads is None else args.threads
    history_file = args.history
    executable_name = "matmul_executable"

    os.environ['OMP_NUM_THREADS'] = str(threads)

    print("--- Roofline Analysis ---")
    print(f"Analyzing: {c_program_path} for N={matrix_size} on {threads} threads")
    print(f"Hardware Roofs: {peak_compute} GFLOP/s (Compute), {peak_memory} GB/s (Memory)")

    print("\n[Step 1/3] Compiling code...")
    compiler = "g++" if c_program_path.endswith(".cpp") else "gcc"
    compile_command = [compiler, "-O3", "-fopenmp", "-o", executable_name, c_program_path]
    run_command(compile_command)
    print(f"Successfully compiled to '{executable_name}'")

    print("\n[Step 2/3] Running program under 'perf stat'...")
    # 'perf stat' prints 'time elapsed' by default, which is the correct wall-clock time.
    # We only need to explicitly ask for the cache-misses event.
    perf_command = [
        "perf", "stat",
        "-e", "cache-misses",
        f"./{executable_name}", str(matrix_size)
    ]
    _, perf_stderr = run_command(perf_command)
    print("'perf stat' execution complete.")

    print("\n[Step 3/3] Calculating metrics and diagnosing...")
    metrics = parse_perf_output(perf_stderr)

    total_flops = 2 * (matrix_size ** 3)
    # Total bytes moved is the number of last-level cache misses * cache line size.
    total_bytes_moved = metrics['cache_misses'] * CACHE_LINE_SIZE
    runtime = metrics['runtime']

    op_intensity = total_flops / total_bytes_moved if total_bytes_moved > 0 else 0
    attained_performance = total_flops / (runtime * 1e9) if runtime > 0 else 0

    ridge_point = peak_compute / peak_memory

    if op_intensity < ridge_point:
        bound = "Memory-Bound"
        max_achievable_perf = op_intensity * peak_memory
    else:
        bound = "Compute-Bound"
        max_achievable_perf = peak_compute

    print("\n--- Analysis Results ---")
    print(f"Execution Time:            {runtime:.4f} s")
    print(f"Cache Misses:              {metrics['cache_misses']:.0f}")
    print(f"Total Data Moved (Est.):   {(total_bytes_moved / 1e9):.4f} GB")
    print(f"Operational Intensity (I): {op_intensity:.4f} FLOPs/Byte")
    print(f"Attained Performance (P):  {attained_performance:.4f} GFLOP/s")
    print("------------------------")

    print("\n--- Diagnosis ---")
    print(f"Your application is: ** {bound} **")
    print(f"Performance Ceiling: {min(max_achievable_perf, peak_compute):.2f} GFLOP/s")
    efficiency = (attained_performance / min(max_achievable_perf, peak_compute) * 100) if min(max_achievable_perf, peak_compute) > 0 else 0
    print(f"Efficiency: {efficiency:.1f}% of ceiling.")
    print("-----------------\n")

    history_data = {}
    if os.path.exists(history_file):
        with open(history_file, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader, None) # Skip header
            history_data = {rows[0]: float(rows[1]) for rows in reader if rows}

    current_version_name = os.path.basename(c_program_path)
    history_data[current_version_name] = attained_performance

    with open(history_file, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Version', 'GFLOPs'])
        for version, gflops in history_data.items():
            writer.writerow([version, gflops])
    print(f"[History] Performance results saved to '{history_file}'")

    plot_roofline(op_intensity, attained_performance, peak_compute, peak_memory, c_program_path, history_data)

    if os.path.exists(executable_name):
        os.remove(executable_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Roofline Analysis Script using perf.")
    parser.add_argument('file', type=str, help="Path to the C/C++ source file to analyze.")
    parser.add_argument('--size', type=int, default=1024, help="The matrix size N.")
    parser.add_argument('--threads', type=int, default=None, help="Number of OMP threads. Defaults to all available cores.")
    parser.add_argument('--peak-compute', type=float, required=True, help="Peak compute performance (GFLOP/s).")
    parser.add_argument('--peak-memory', type=float, required=True, help="Peak memory bandwidth (GB/s).")
    parser.add_argument('--history', type=str, default='performance_history.csv', help="Path to a CSV file to store performance history.")

    try:
        subprocess.run(["which", "perf"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'perf' not found. Please install it.")
        print("On Debian/Ubuntu, use: sudo apt-get install linux-tools-common linux-tools-`uname -r`")
        sys.exit(1)

    args = parser.parse_args()
    main(args)
