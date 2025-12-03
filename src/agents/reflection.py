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
            f"""You are a reflection agent that analyzes previously generated code and feedback to provide suggestions for improvement.

            Architecture: {architecture}

            Currently Generated Code:
            {generated_code}

            Feedback Received:
            {json.dumps(feedback, indent=4)}

            History of generated code and feedback:
            {history_json}

            Based on the above, provide suggestions on how to improve the generated code and its performance. Be specific and actionable in your recommendations.
            """
        )
        return self._ask_gemini(full_prompt)
