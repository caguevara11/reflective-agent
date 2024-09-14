import time
import functools
import logging
import requests

logger = logging.getLogger(__name__)

def retry(max_attempts: int, delay: float):
    """
    Decorator for retrying a function call with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ValueError) as e:
                    attempts += 1
                    logger.warning(f"Attempt {attempts} failed with error: {e}")
                    if attempts == max_attempts:
                        logger.error("Max retry attempts reached.")
                        raise
                    time.sleep(current_delay)
                    current_delay *= 2  # Exponential backoff
        return wrapper
    return decorator
