import requests
from typing import Dict, List
from src.utils.retry_decorator import retry

class OpenAICompatibleAPI:
    """
    Handles interactions with an OpenAI-compatible API.
    """
    DEFAULT_MODEL = 'openai/gpt-4o-mini-2024-07-18'
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 8000

    def __init__(self, endpoint: str, api_key: str, site_url: str = '', site_name: str = ''):
        self.endpoint = endpoint
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name

    def _create_headers(self) -> Dict[str, str]:
        """Creates headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.site_name
        }

    def _create_payload(self, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> Dict:
        """Creates the payload for the API request."""
        return {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    @retry(max_attempts=3, delay=1.0)
    def _make_request(self, payload: Dict) -> str:
        """Handles the API request and processes the response."""
        response = requests.post(self.endpoint, headers=self._create_headers(), json=payload)
        response.raise_for_status()
        data = response.json()
        if 'choices' in data and data['choices']:
            return data['choices'][0]['message']['content'].strip()
        raise ValueError("No valid response received from the API.")

    def generate_reasoning_tokens(self, user_question: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Generates reasoning tokens for the given question."""
        messages = [
            {
                "role": "system",
                "content": (
                     """
                    You are an AI assistant designed to provide detailed, step-by-step responses to the final optimized prompt. Your outputs should follow this structure:

                    1. Begin with a <thinking> section. Everything in this section is invisible to the user.

                    2. Inside the thinking section:

                    a. Briefly analyze the question and outline your approach.

                    b. Present a clear plan of steps to solve the problem.

                    c. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.

                    3. Include a <reflection> section for each idea where you:

                    a. Review your reasoning.

                    b. Check for potential errors or oversights.

                    c. Confirm or adjust your conclusion if necessary.

                    4. Be sure to close all reflection sections.

                    5. Close the thinking section with </thinking>. Thats the final tag.

                    Do not do anything else.
                    Your answer should be only the reasoning tokens that will later be used to generate the final answer.
                    Thats means that you dont write anything else after closing the thinking section tag.
                    """
                )
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
        payload = self._create_payload(model, messages, temperature, max_tokens)
        return self._make_request(payload)

    def generate_final_answer(self, user_question: str, reasoning_tokens: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Generates the final answer based on the provided reasoning tokens."""
        messages = [
            {
                "role": "system",
                "content":
                    """
                    Based on the provided reasoning token inside <thinking> tag, generate a clear and concise answer to the user's question.
                    "Ensure that the answer is accurate and directly addresses the question.
                    """
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
        payload = self._create_payload(model, messages, temperature, max_tokens)
        return self._make_request(payload)

    def improve_user_question(self, user_question: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Improves the user's question to generate better reasoning tokens."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Act as a prompt engineer to refine the user's question for clarity and effectiveness. "
                    "Identify and apply relevant frameworks to optimize the prompt."
                )
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
        payload = self._create_payload(model, messages, temperature, max_tokens)
        return self._make_request(payload)
