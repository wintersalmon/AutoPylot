import requests
import json
from typing import Any, Dict, Callable, Optional, Tuple

class LLMRequestAPI:
    HEADERS: Dict[str, str] = {
        "Content-Type": "application/json"
    }

    def __init__(self, OLLAMA_URL: str, model: str) -> None:
        """
        Initialize the LLMRequestAPI with the given URL and model.
        """
        self.OLLAMA_URL: str = OLLAMA_URL
        self.model: str = model

    def process_message(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process the given message by sending a request to the Ollama API.

        Args:
            message (str): The message to process.

        Returns:
            Tuple[Optional[str], Optional[str]]: A tuple containing the result text and an error message, if any.
        """
        result_text: Optional[str] = None
        error_message: Optional[str] = None

        # Request payload
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": message,
            "stream": False,
        }

        try:
            response: requests.Response = requests.post(
                self.OLLAMA_URL,
                json=payload,
                headers=self.HEADERS
            )
            response.raise_for_status()

            try:
                raw_result: Dict[str, Any] = response.json()
                result_text = raw_result.get('response', '')
            except json.JSONDecodeError as e:
                error_message = f"Error decoding JSON response from Ollama API: {e}. Response text: {response.text}"
        except requests.exceptions.RequestException as e:
            error_message = f"Error calling Ollama API: {e}"

        return result_text, error_message

def createLLMRequestAPI(OLLAMA_URL: str, model: str) -> Callable[[str], Tuple[Optional[str], Optional[str]]]:
    """
    Create an instance of LLMRequestAPI and return its process_message method.

    Args:
        OLLAMA_URL (str): The URL of the Ollama API.
        model (str): The model to use for processing messages.

    Returns:
        Callable[[str], Tuple[Optional[str], Optional[str]]]: The process_message method of the LLMRequestAPI instance.
    """
    api_instance: LLMRequestAPI = LLMRequestAPI(OLLAMA_URL, model)
    return api_instance.process_message
