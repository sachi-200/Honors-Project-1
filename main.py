import os
import re
from src.agents.evaluator import evaluate_code
from src.agents.generator import GeneratorAgent
import json

def main():
    generator = GeneratorAgent()
    history = {}
    max_gflops = 0.0
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    file_pattern = re.compile(r'^(\d+)-.*\.cpp$')
    next_file_num = 1
    try:
        existing_files = os.listdir(results_dir)
        file_numbers = [int(file_pattern.match(f).group(1)) for f in existing_files if file_pattern.match(f)]
        if file_numbers:
            next_file_num = max(file_numbers) + 1
    except FileNotFoundError:
        pass

    for i in range(10):
        print(f"--- Iteration {i+1} ---")
        print("Generating code...")
        generated_code = generator.generate_code(history)
        print("Evaluating code...")
        feedback = evaluate_code(generated_code)
        print(json.dumps(feedback, indent=4))
        history[i] = {"code": generated_code, "feedback": feedback}

        current_gflops = feedback.get('performance', {}).get('gflops')
        if current_gflops is not None:
            if current_gflops > max_gflops:
                max_gflops = current_gflops
                filename = f"{next_file_num}-{max_gflops:.4f}.cpp"
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(generated_code)
                print(f"New best performance! Saved to {filepath}")
            else:
                print(f"Current performance: {current_gflops:.2f} GFLOPS (Best: {max_gflops:.2f} GFLOPS)")

if __name__ == "__main__":
    main()