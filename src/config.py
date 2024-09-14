import os
from dotenv import load_dotenv

load_dotenv()

LLM_API_ENDPOINT = os.getenv('LLM_API_ENDPOINT') or 'https://openrouter.ai/api/v1/chat/completions'
LLM_API_KEY = os.getenv('LLM_API_KEY')
YOUR_SITE_URL = os.getenv('YOUR_SITE_URL', 'https://your-site.com')
YOUR_SITE_NAME = os.getenv('YOUR_SITE_NAME', 'YourAppName')
