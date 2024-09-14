import os
from dotenv import load_dotenv

load_dotenv()  # Carga las variables de entorno desde el archivo .env

API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
YOUR_SITE_URL = os.getenv('YOUR_SITE_URL', 'https://your-site.com')
YOUR_SITE_NAME = os.getenv('YOUR_SITE_NAME', 'YourAppName')
