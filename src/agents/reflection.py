import json
import requests
import time
from src.tools.env_helper import load_api_key

class ReflectionAgent:
    def __init__(self):
        self.API_KEY = load_api_key(key_name="REFLECTION_KEY")
        self.MODEL = "gemini-2.5-flash"
        self.URL = f"https://generativelanguage.googleapis.com/v1beta/models/{self.MODEL}:generateContent?key={self.API_KEY}"

    def _ask_gemini(self, prompt: str, retries=3, backoff_factor=0.5) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [
                {"parts": [{"text": prompt}]}
            ]
        }

        for i in range(retries):
            try:
                response = requests.post(self.URL, headers=headers, json=data)
                response.raise_for_status()
                resp_json = response.json()
                return resp_json["candidates"][0]["content"]["parts"][0]["text"]
            except requests.exceptions.HTTPError as e:
                if 500 <= e.response.status_code < 600:
                    print(f"Server error ({e.response.status_code}), retrying in {backoff_factor * (2 ** i)} seconds...")
                    time.sleep(backoff_factor * (2 ** i))
                else:
                    raise

    def reflect(self, history: dict, generated_code: str, feedback: dict, architecture: str) -> str:
        # Ensure history is JSON-serializable even if keys are tuples or other non-JSON types
        try:
            history_json = json.dumps(history, indent=4)
        except TypeError:
            # Convert keys to strings as a fallback
            safe_history = {str(k): v for k, v in history.items()}
            history_json = json.dumps(safe_history, indent=4)

        full_prompt = (
    f"""You are an expert HPC programmer specializing in OpenMP optimization for CPU-based matrix multiplication (GEMM).
        Analyze the current implementation, test results, and historical performance to provide actionable insights for improvement.

        Architecture: {architecture}

        Currently Generated Code:
        ```cpp
        {generated_code}
        ```

        Feedback Received:
        {json.dumps(feedback, indent=4)}

        History of generated code and feedback:
        {history_json}

        # Analysis Framework

        Analyze the implementation across these key areas:

        1. **Performance Bottlenecks**: What's limiting GFLOPS? (memory bandwidth, cache misses, false sharing, compute vs memory bound)

        2. **OpenMP Parallelization**: Is the parallel region optimal? Thread balance? Scheduling policy? Overhead issues?

        3. **Memory & Cache**: Is loop tiling effective? Cache-friendly access patterns? Optimal tile sizes? Proper alignment?

        4. **Vectorization**: Are loops vectorizing (#pragma omp simd)? Any inhibitors? SIMD opportunities?

        5. **Algorithm & Loops**: Is loop order (IJK/IKJ/KIJ) optimal? Loop unrolling needed? Register reuse maximized?

        6. **Historical Trends**: What worked before? Repeated mistakes? Progress or regression?

        # Required Output Structure

        ## Critical Issues
        [List compilation errors, test failures, or blocking bugs - "None" if all passing]

        ## Top 3 Bottlenecks
        [Specific bottlenecks with evidence from feedback]
        1.
        2.
        3.

        ## Optimization Suggestions
        [3-5 concrete, implementable changes]
        1. [Specific change] → Expected: [impact estimate]
        2. [Specific change] → Expected: [impact estimate]
        3. [Specific change] → Expected: [impact estimate]

        ## Code Modifications
        [Specific code patterns or snippets to try next iteration]

        ## Strategy for Next Iteration
        [High-level direction based on history and current state]

        Be specific and technical. No generic advice. Focus on actionable changes.
        """
        )
        return self._ask_gemini(full_prompt)
