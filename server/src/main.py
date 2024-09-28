from api import createLLMRequestAPI

LLM_URL: str = "http://ollama:11434/api/generate"
LLM_MODEL: str = "llama3.2:3b"

if __name__ == '__main__':
    try:
        api = createLLMRequestAPI(LLM_URL, LLM_MODEL)
        result = api("Why is the sky blue?")
        print(result)

    except Exception as e:
        print(f"API 생성 실패: {e}")
        exit(1)
