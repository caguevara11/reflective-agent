import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.openai_compatible_api import OpenAICompatibleAPI
from src.config import LLM_API_ENDPOINT, LLM_API_KEY, YOUR_SITE_URL, YOUR_SITE_NAME
from src.logger import logger

def process_question(user_question: str) -> str:
    """
    Processes the user's question by generating reasoning tokens and then the final answer.
    """
    api = OpenAICompatibleAPI(LLM_API_ENDPOINT, LLM_API_KEY, YOUR_SITE_URL, YOUR_SITE_NAME)

    try:
        logger.info("Improving user question...")
        improved_user_question = api.improve_user_question(user_question)
        print("\nImproved User Question:\n")
        print(improved_user_question)
        logger.info("Generating reasoning tokens...")
        reasoning_tokens = api.generate_reasoning_tokens(improved_user_question)
        logger.info("Reasoning tokens generated.")
        print("\nReasoning Tokens:\n")
        print(reasoning_tokens)

        logger.info("Generating final answer...")
        final_answer = api.generate_final_answer(user_question, reasoning_tokens)
        logger.info("Final answer generated.")
        return final_answer
    except Exception as err:
        logger.error(f"An error occurred: {err}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    question = input("Please enter your question:\n")
    answer = process_question(question)
    print("\nAnswer:\n")
    print(answer)
