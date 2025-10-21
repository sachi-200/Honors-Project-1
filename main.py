import os
import re
import json
from src.agents.evaluator import evaluate_code
from src.agents.generator import GeneratorAgent

def main():
    cpu_arch = ""
    while cpu_arch not in ["amd", "intel"]:
        cpu_arch = input("Enter CPU architecture (AMD/Intel): ").lower().strip()
        if cpu_arch not in ["amd", "intel"]:
            print("Invalid input. Please enter 'AMD' or 'Intel'.")

    if cpu_arch == "amd":
        cpu_model = 'AMD-Ryzen-7-6800HS'
    else:
        cpu_model = 'Intel-i7-1195G7'
    print(f"Using profile: {cpu_model}")

    matrix_sizes = [128, 256, 512, 1024, 2048, 4096]

    generator = GeneratorAgent()
    history = {}

    for matrix_size in matrix_sizes:
        print(f"\n===== Running benchmarks for matrix size {matrix_size} =====")

        results_dir = os.path.join("results", cpu_model, str(matrix_size))
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved in: {results_dir}")

        file_pattern = re.compile(r'^(\d+)-.*\.cpp$')
        next_file_num = 1
        try:
            existing_files = os.listdir(results_dir)
            file_numbers = [int(file_pattern.match(f).group(1)) for f in existing_files if file_pattern.match(f)]
            if file_numbers:
                next_file_num = max(file_numbers) + 1
        except FileNotFoundError:
            pass

        max_gflops = 0.0
        current_run_best_filepath = None

        for i in range(10):
            print(f"--- Iteration {i+1} (matrix {matrix_size}) ---")

            last_5_keys = sorted(history.keys())[-5:]
            recent_history = {k: history[k] for k in last_5_keys}

            print("Generating code...")
            generated_code = generator.generate_code(recent_history, architecture=cpu_model)
            print("Evaluating code...")
            feedback = evaluate_code(generated_code, matrix_size=matrix_size, system_type=cpu_arch)
            print(json.dumps(feedback, indent=4))
            history[(matrix_size, i)] = {"code": generated_code, "feedback": feedback}

            current_gflops = feedback.get('performance', {}).get('gflops')
            if current_gflops is not None:
                if current_gflops > max_gflops:
                    if current_run_best_filepath and os.path.exists(current_run_best_filepath):
                        os.remove(current_run_best_filepath)

                    max_gflops = current_gflops
                    filename = f"{next_file_num}-{max_gflops:.4f}.cpp"
                    filepath = os.path.join(results_dir, filename)
                    with open(filepath, 'w') as f:
                        f.write(generated_code)
                    current_run_best_filepath = filepath
                    print(f"New best performance! Saved to {filepath}")
                else:
                    print(f"Current performance: {current_gflops:.2f} GFLOPS (Best: {max_gflops:.2f} GFLOPS)")

if __name__ == "__main__":
    main()