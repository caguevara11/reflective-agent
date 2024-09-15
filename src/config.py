import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = 'openai/gpt-4o-mini-2024-07-18'
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 8000
LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT') or 'https://openrouter.ai/api/v1/chat/completions'
LLM_API_KEY = os.getenv('LLM_API_KEY')
YOUR_SITE_URL = os.getenv('YOUR_SITE_URL', 'https://your-site.com')
YOUR_SITE_NAME = os.getenv('YOUR_SITE_NAME', 'YourAppName')
