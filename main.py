from src.agents.evaluator import evaluate_code
from src.agents.generator import GeneratorAgent
import json

def main():
    generator = GeneratorAgent()
    history = {}

    for i in range(10):
        print(f"--- Iteration {i+1} ---")
        print("Generating code...")
        generated_code = generator.generate_code(history)
        print("Evaluating code...")
        feedback = evaluate_code(generated_code)
        print(json.dumps(feedback, indent=4))
        history[i] = {"code": generated_code, "feedback": feedback}

if __name__ == "__main__":
    main()