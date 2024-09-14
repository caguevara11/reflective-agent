import requests
from typing import Dict, List
from src.utils.retry_decorator import retry

class OpenAICompatibleAPI:
    """
    A class that handles interactions with an OpenAI compatible API.
    """

    def __init__(self, api_key: str, site_url: str = '', site_name: str = ''):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name

    def _create_headers(self) -> Dict[str, str]:
        """Create headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }

    def _create_payload(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 100) -> Dict:
        """Create the payload for the API request."""
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    @retry(max_attempts=3, delay=1.0)
    def make_request(self, payload: Dict) -> str:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=self._create_headers(), json=payload)
        response.raise_for_status()  # Raises HTTPError for bad HTTP status codes
        data = response.json()
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content'].strip()
        else:
            raise ValueError("No response received from the API.")

    def generate_reasoning_tokens(self, user_question: str) -> str:
        """Generates reasoning tokens for the given question."""
        messages = [
            {
                "role": "system",
                "content": (
                    "As an AI assistant, when presented with a question, you should think through the problem "
                    "in a detailed, step-by-step manner. Provide this reasoning process to help understand how "
                    "to approach and solve the problem. Do not include the final answer."
                )
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
        payload = self._create_payload(
            model="openai/gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.7,
            max_tokens=8000
        )
        return self.make_request(payload)

    def generate_final_answer(self, user_question: str, reasoning_tokens: str) -> str:
        """Generates the final answer based on the provided reasoning tokens."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Based on the provided reasoning, generate a clear and concise answer to the user's question. "
                    "Ensure that the answer is accurate and directly addresses the question."
                )
            },
            {
                "role": "assistant",
                "content": reasoning_tokens
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
        payload = self._create_payload(
            model="openai/gpt-4o-mini-2024-07-18",
            messages=messages,
            temperature=0.7,
            max_tokens=8000
        )
        return self.make_request(payload)
