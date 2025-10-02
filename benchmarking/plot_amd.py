import os
import re
import matplotlib.pyplot as plt

def plot_gflops_from_results(cpu_model_dir: str):
    """
    Parses GFLOPS values for LLM-generated code from filenames and
    plots them against a hardcoded Intel MKL baseline.

    Args:
        cpu_model_dir (str): The path to the specific CPU model's results directory.
    """
    if not os.path.isdir(cpu_model_dir):
        print(f"Error: Directory not found at '{cpu_model_dir}'")
        return

    # --- Data Parsing for LLM-generated Code ---
    performance_data_llm = []
    filename_pattern = re.compile(r'^\d+-(\d+\.\d+)\.cpp$')

    for dirpath, _, filenames in os.walk(cpu_model_dir):
        matrix_size_str = os.path.basename(dirpath)

        if matrix_size_str.isdigit():
            matrix_size = int(matrix_size_str)
            for filename in filenames:
                match = filename_pattern.match(filename)
                if match:
                    gflops = float(match.group(1))
                    performance_data_llm.append((matrix_size, gflops))
                    break

    if not performance_data_llm:
        print("No valid LLM data found to plot. Check the directory structure and filenames.")
        return

    performance_data_llm.sort()
    matrix_sizes_llm = [item[0] for item in performance_data_llm]
    gflops_values_llm = [item[1] for item in performance_data_llm]

    # --- Intel MKL Baseline Data (from your previous run) ---
    mkl_matrix_sizes = [128, 256, 512, 1024, 2048, 4096]
    mkl_gflops_values = [111.55, 325.76, 403.68, 420.23, 322.35, 316.00]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # Plot the LLM-generated code's performance
    plt.plot(matrix_sizes_llm, gflops_values_llm, marker='o', linestyle='-', color='b', label='LLM-Generated Code')

    # --- ADDED FOR MKL COMPARISON ---
    # Plot the Intel MKL baseline performance on the same axes
    plt.plot(mkl_matrix_sizes, mkl_gflops_values, marker='s', linestyle='--', color='r', label='Intel MKL Baseline')
    # --- END OF ADDITION ---

    # Use a log scale for the x-axis, ensuring ticks are set for all sizes
    all_sizes = sorted(list(set(matrix_sizes_llm + mkl_matrix_sizes)))
    plt.xscale('log', base=2)
    plt.xticks(all_sizes, labels=[str(s) for s in all_sizes])

    # --- MODIFIED TITLE ---
    plt.title(f'GEMM Performance Comparison on {os.path.basename(cpu_model_dir)}')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Performance (GFLOPS)')
    plt.legend()
    plt.minorticks_off()

    # Add data labels to each point for the LLM data
    for i, txt in enumerate(gflops_values_llm):
        plt.annotate(f'{txt:.2f}', (matrix_sizes_llm[i], gflops_values_llm[i]), textcoords="offset points", xytext=(0,10), ha='center', color='b')

    # Add data labels to each point for the MKL data
    for i, txt in enumerate(mkl_gflops_values):
        plt.annotate(f'{txt:.2f}', (mkl_matrix_sizes[i], mkl_gflops_values[i]), textcoords="offset points", xytext=(0,-15), ha='center', color='r')

    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    print("Comparative plot saved as performance_comparison.png")
    plt.show()


if __name__ == "__main__":
    RESULTS_ROOT = "../results"
    CPU_MODEL = "AMD-Ryzen-7-6800HS"

    full_path = os.path.join(RESULTS_ROOT, CPU_MODEL)

    plot_gflops_from_results(full_path)