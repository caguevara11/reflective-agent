import unittest
from api.openai_compatible_api import OpenRouterAPI
from unittest.mock import patch

class TestOpenRouterAPI(unittest.TestCase):

    @patch('src.api.open_router_api.requests.post')
    def test_generate_reasoning_tokens(self, mock_post):
        # Configurar el mock
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'choices': [{'message': {'content': 'This is a test reasoning token'}}]
        }

        api = OpenRouterAPI('dummy_api_key')
        result = api.generate_reasoning_tokens("What is the capital of France?")
        self.assertEqual(result, 'This is a test reasoning token')

    @patch('src.api.open_router_api.requests.post')
    def test_generate_final_answer(self, mock_post):
        # Configurar el mock
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            'choices': [{'message': {'content': 'The capital of France is Paris.'}}]
        }

        api = OpenRouterAPI('dummy_api_key')
        result = api.generate_final_answer("What is the capital of France?", "Some reasoning tokens")
        self.assertEqual(result, 'The capital of France is Paris.')

if __name__ == '__main__':
    unittest.main()
