import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Import the ticker

def plot_gflops_from_results(cpu_model_dir: str):
    """
    Parses GFLOPS values for 3 different versions of LLM-generated code
    from filenames and plots them against a hardcoded Intel MKL baseline.

    Uses log scale on both axes for clarity.

    Args:
        cpu_model_dir (str): The path to the specific CPU model's results directory.
    """
    if not os.path.isdir(cpu_model_dir):
        print(f"Error: Directory not found at '{cpu_model_dir}'")
        return

    # --- Data Parsing for LLM-generated Code ---
    perf_v1 = {}
    perf_v2 = {}
    perf_v3 = {}
    perf_v4 = {}
    perf_v5 = {}

    filename_pattern = re.compile(r'^(\d+)-(\d+\.\d+)\.cpp$')

    for dirpath, _, filenames in os.walk(cpu_model_dir):
        matrix_size_str = os.path.basename(dirpath)

        if matrix_size_str.isdigit():
            matrix_size = int(matrix_size_str)
            for filename in filenames:
                match = filename_pattern.match(filename)
                if match:
                    version = match.group(1)
                    gflops = float(match.group(2))

                    if version == '1':
                        perf_v1[matrix_size] = gflops
                    elif version == '2':
                        perf_v2[matrix_size] = gflops
                    elif version == '3':
                        perf_v3[matrix_size] = gflops
                    elif version == '4':
                        perf_v4[matrix_size] = gflops
                    elif version == '5':
                        perf_v5[matrix_size] = gflops

    if not perf_v1 and not perf_v2 and not perf_v3 and not perf_v4 and not perf_v5:
        print("No valid LLM data found to plot. Check the directory structure and filenames.")
        return

    # --- Prepare data for plotting (sort by matrix size) ---
    def get_sorted_plot_data(perf_dict):
        sorted_sizes = sorted(perf_dict.keys())
        sorted_gflops = [perf_dict[size] for size in sorted_sizes]
        return sorted_sizes, sorted_gflops

    matrix_sizes_v1, gflops_v1 = get_sorted_plot_data(perf_v1)
    matrix_sizes_v2, gflops_v2 = get_sorted_plot_data(perf_v2)
    matrix_sizes_v3, gflops_v3 = get_sorted_plot_data(perf_v3)
    matrix_sizes_v4, gflops_v4 = get_sorted_plot_data(perf_v4)
    matrix_sizes_v5, gflops_v5 = get_sorted_plot_data(perf_v5)

    # --- Intel MKL Baseline Data ---
    mkl_matrix_sizes = [128, 256, 512, 1024, 2048, 4096]
    mkl_gflops_values = [111.55, 325.76, 403.68, 420.23, 322.35, 316.00]

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))

    # --- PLOT ALL 4 LINES ---
    plt.plot(matrix_sizes_v1, gflops_v1, marker='o', linestyle=':', color='b', label='V1: No Roofline')
    plt.plot(matrix_sizes_v2, gflops_v2, marker='^', linestyle='--', color='g', label='V2: With Roofline')
    plt.plot(matrix_sizes_v3, gflops_v3, marker='D', linestyle='-', color='m', label='V3: Roofline + Cache')
    plt.plot(matrix_sizes_v4, gflops_v4, marker='s', linestyle='-', color='c', label='V4: Removed Redundancies + Memory/Compute Specific Optimizations')
    plt.plot(matrix_sizes_v5, gflops_v5, marker='p', linestyle='-', color='y', label='V5: Reflection Agent')
    plt.plot(mkl_matrix_sizes, mkl_gflops_values, marker='s', linestyle='-.', color='r', label='Intel MKL Baseline')

    # --- AXIS SCALING AND FORMATTING ---

    # Use a log scale for the x-axis (as before)
    all_sizes = sorted(list(set(matrix_sizes_v1 + matrix_sizes_v2 + matrix_sizes_v3 + mkl_matrix_sizes)))
    plt.xscale('log', base=2)
    plt.xticks(all_sizes, labels=[str(s) for s in all_sizes])
    plt.minorticks_off()

    # *** NEW: Use a log scale for the y-axis ***
    plt.yscale('log')

    # *** NEW: Format y-axis ticks as regular numbers (0.1, 1, 10, 100) ***
    # This prevents labels like 10^0, 10^1, 10^2
    y_ax = plt.gca().yaxis
    y_ax.set_major_formatter(ticker.ScalarFormatter())
    y_ax.set_minor_formatter(ticker.ScalarFormatter())
    # You may need to manually set ticks if auto-scaling isn't perfect
    # plt.yticks([0.1, 1, 10, 100, 500], labels=['0.1', '1', '10', '100', '500'])

    plt.title(f'GEMM Performance Comparison on {os.path.basename(cpu_model_dir)}')
    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Performance (GFLOPS) - Log Scale')
    plt.legend()


    # --- ANNOTATIONS (Adjusted offsets) ---
    text_offset_above = (0, 5)  # Place text slightly above the marker
    text_offset_below = (0, -10) # Place text slightly below the marker

    # V1 labels
    for i, txt in enumerate(gflops_v1):
        plt.annotate(f'{txt:.2f}', (matrix_sizes_v1[i], gflops_v1[i]), textcoords="offset points", xytext=text_offset_above, ha='center', color='b', fontsize=8)

    # V2 labels
    for i, txt in enumerate(gflops_v2):
        plt.annotate(f'{txt:.2f}', (matrix_sizes_v2[i], gflops_v2[i]), textcoords="offset points", xytext=text_offset_above, ha='center', color='g', fontsize=8)

    # V3 labels
    for i, txt in enumerate(gflops_v3):
        plt.annotate(f'{txt:.2f}', (matrix_sizes_v3[i], gflops_v3[i]), textcoords="offset points", xytext=text_offset_above, ha='center', color='m', fontsize=8)

    # V4 labels
    for i, txt in enumerate(gflops_v4):
        plt.annotate(f'{txt:.2f}', (matrix_sizes_v4[i], gflops_v4[i]), textcoords="offset points", xytext=text_offset_above, ha='center', color='c', fontsize=8)

    # V5 labels
    for i, txt in enumerate(gflops_v5):
        plt.annotate(f'{txt:.2f}', (matrix_sizes_v5[i], gflops_v5[i]), textcoords="offset points", xytext=text_offset_above, ha='center', color='y', fontsize=8)

    # MKL labels
    for i, txt in enumerate(mkl_gflops_values):
        plt.annotate(f'{txt:.2f}', (mkl_matrix_sizes[i], mkl_gflops_values[i]), textcoords="offset points", xytext=text_offset_below, ha='center', color='r', fontsize=8)

    plt.tight_layout()
    plt.savefig('performance_comparison_log.png')
    print("Comparative plot saved as performance_comparison_log.png")
    plt.show()


if __name__ == "__main__":
    RESULTS_ROOT = "../results"
    CPU_MODEL = "AMD-Ryzen-7-6800HS"

    full_path = os.path.join(RESULTS_ROOT, CPU_MODEL)
    plot_gflops_from_results(full_path)