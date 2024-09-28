import requests
import json
from typing import Any, Dict, Callable, Optional, Tuple

class LLMRequestAPI:
    def __init__(self, OLLAMA_URL: str, model: str) -> None:
        self.OLLAMA_URL: str = OLLAMA_URL
        self.model: str = model

    def process_message(self, message: str) -> Tuple[Optional[str], Optional[str]]:
        content: str = message
        result_text: Optional[str] = None
        error_message: Optional[str] = None

        # Ollama에 요청 (비스트리밍 처리)
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": content,
            "stream": False,
        }

        headers: Dict[str, str] = {
            "Content-Type": "application/json"
        }

        try:
            response: requests.Response = requests.post(
                self.OLLAMA_URL, json=payload, headers=headers
            )
            response.raise_for_status()

            try:
                raw_result: Dict[str, Any] = response.json()
                result_text = raw_result.get('response', '')
                return result_text, None  # 성공 시 [success_message, None] 반환
            except json.JSONDecodeError as e:
                error_message = f"Ollama API 응답 JSON 디코딩 오류: {e}. 응답 텍스트: {response.text}"
                return None, error_message  # 실패 시 [None, error_message] 반환
        except requests.exceptions.RequestException as e:
            error_message = f"Ollama API 호출 오류: {e}"
            return None, error_message  # 실패 시 [None, error_message] 반환

def createLLMRequestAPI(OLLAMA_URL: str, model: str) -> Callable[[str], Tuple[Optional[str], Optional[str]]]:
    api_instance: LLMRequestAPI = LLMRequestAPI(OLLAMA_URL, model)
    return api_instance.process_message
