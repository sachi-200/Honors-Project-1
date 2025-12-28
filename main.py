import os
import re
import json
from typing import Optional
from src.agents.evaluator import evaluate_code
from src.agents.generator import GeneratorAgent
from src.agents.reflection import ReflectionAgent


def _human_in_loop_from_config(config_path: str = "config.yaml") -> bool:
    """Read a config and return True if human_in_the_loop is true."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("human_in_the_loop"):
                    parts = line.split("=", 1)
                    if len(parts) != 2:
                        return False
                    val = parts[1].strip()
                    return val.lower() in ("true", "1", "yes", "y")
    except FileNotFoundError:
        return False
    return False

def _get_reflection_agent_from_config(config_path: str = "config.yaml") -> bool:
    """Read a config and return True if reflection_agent is true."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("reflection_agent"):
                    parts = line.split("=", 1)
                    if len(parts) != 2:
                        return False
                    val = parts[1].strip()
                    return val.lower() in ("true", "1", "yes", "y")
    except FileNotFoundError:
        return False
    return False

def get_human_feedback():
    """Prompts the user for multi-line feedback until 'done' is entered."""
    print("\nWAITING FOR HUMAN FEEDBACK")
    print("Please enter your feedback for the generator.")
    print("Type 'done' on a new, empty line when you are finished.")

    lines = []
    while True:
        try:
            line = input()
            if line.strip().lower() == "done":
                break
            lines.append(line)
        except EOFError:
            break

    print("--- ✅ Feedback captured ---")
    return "\n".join(lines)

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

    # matrix_sizes = [128, 256, 512, 1024, 2048, 4096]
    matrix_sizes = [256]

    generator = GeneratorAgent()
    reflector = ReflectionAgent()
    history = {}

    human_in_loop = _human_in_loop_from_config()
    reflection_agent = _get_reflection_agent_from_config()

    for matrix_size in matrix_sizes:
        print(f"\n===== Running benchmarks for matrix size {matrix_size} =====")

        results_dir = os.path.join("results", cpu_model, str(matrix_size))
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved in: {results_dir}")

        file_pattern = re.compile(r'^(\d+)-.*\.cpp$')
        next_file_num = 1
        try:
            existing_files = os.listdir(results_dir)
            file_numbers = [
                int(file_pattern.match(f).group(1))
                for f in existing_files
                if file_pattern.match(f)
            ]
            if file_numbers:
                next_file_num = max(file_numbers) + 1
        except FileNotFoundError:
            pass

        successful_run = False
        attempt_count = 0

        while not successful_run:
            attempt_count += 1
            print(f"\n### Attempt {attempt_count} for matrix size {matrix_size} ###")

            max_gflops = 0.0
            current_run_best_filepath = None

            for i in range(10):
                print(f"--- Iteration {i + 1} (matrix {matrix_size}) ---")

                # Get last 5 items from history to guide generation
                last_5_keys = sorted(history.keys())[-5:]
                recent_history = {k: history[k] for k in last_5_keys}

                print("Generating code...")
                generated_code = generator.generate_code(
                    recent_history, architecture=cpu_model
                )

                print("Evaluating code...")
                feedback = evaluate_code(
                    generated_code, matrix_size=matrix_size, system_type=cpu_arch
                )

                print(json.dumps(feedback, indent=4))

                all_tests_passed = False
                if feedback.get("tests"):
                    all_tests_passed = all(test.get("passed", False) for test in feedback["tests"])

                if all_tests_passed and human_in_loop:
                    feedback["human_feedback"] = get_human_feedback()

                if reflection_agent:
                    # pass in the history so far, as well as the code generated in this agent and feedback received to the reflector
                    reflection_suggestions = reflector.reflect(
                        history,
                        generated_code,
                        feedback,
                        architecture=cpu_model,
                    )
                    if reflection_suggestions:
                        print("Reflection Agent Suggestions:")
                        print(reflection_suggestions)
                        feedback["reflection_suggestions"] = reflection_suggestions

                # Store history with a JSON-serializable key to avoid tuple key errors
                history[f"{matrix_size}:{i}"] = {
                    "code": generated_code,
                    "feedback": feedback,
                }

                # Extract GFLOPS performance
                current_gflops = feedback.get("performance", {}).get("gflops")

                if current_gflops is not None and current_gflops > 0:
                    if current_gflops > max_gflops:
                        # Remove previous best file
                        if current_run_best_filepath and os.path.exists(current_run_best_filepath):
                            try:
                                os.remove(current_run_best_filepath)
                            except OSError:
                                pass
                            # also remove the corresponding .txt feedback file if present
                            prev_txt = os.path.splitext(current_run_best_filepath)[0] + ".txt"
                            if os.path.exists(prev_txt):
                                try:
                                    os.remove(prev_txt)
                                except OSError:
                                    pass

                        max_gflops = current_gflops
                        filename_base = f"{next_file_num}-{max_gflops:.4f}"
                        cpp_filename = f"{filename_base}.cpp"
                        cpp_path = os.path.join(results_dir, cpp_filename)
                        with open(cpp_path, "w") as f:
                            f.write(generated_code)

                        # Save feedback next to the .cpp with same base name but .txt
                        txt_filename = f"{filename_base}.txt"
                        txt_path = os.path.join(results_dir, txt_filename)
                        try:
                            with open(txt_path, "w") as tf:
                                tf.write(json.dumps(feedback, indent=4))
                        except OSError:
                            print(f"Warning: failed to write feedback file {txt_path}")

                        current_run_best_filepath = cpp_path
                        print(f"New best performance! Saved to {cpp_path} (feedback: {txt_path})")
                    else:
                        print(
                            f"Current performance: {current_gflops:.2f} GFLOPS "
                            f"(Best: {max_gflops:.2f} GFLOPS)"
                        )
                else:
                    print("No valid performance result this iteration.")

            # ✅ Check if any successful run happened this batch
            if max_gflops > 0:
                successful_run = True
                print(
                    f"✅ Successful benchmark achieved for matrix size {matrix_size} "
                    f"after {attempt_count} attempt(s)."
                )
            else:
                print(
                    f"⚠️  No successful code found after {10 * attempt_count} iterations. "
                    f"Restarting with new generated code..."
                )


if __name__ == "__main__":
    main()
