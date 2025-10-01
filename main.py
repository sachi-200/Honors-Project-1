from src.agents.evaluator import evaluate_code
from src.agents.generator import GeneratorAgent

def main():
    generator = GeneratorAgent()

    prompt = "Write an OpenMP C++ function for matrix multiplication."
    unit_tests = [("unit_tests/dense_matmul/input1.txt", "unit_tests/dense_matmul/expected_output1.txt")]
    history = {}

    for i in range(5):
        print(f"--- Iteration {i+1} ---")
        print("Generating code...")
        generated_code = generator.generate_code(prompt, history)
        print("Evaluating code...")
        feedback = evaluate_code(generated_code, unit_tests)
        print(feedback)
        history[i] = {"code": generated_code, "feedback": feedback}

if __name__ == "__main__":
    main()