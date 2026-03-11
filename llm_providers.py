import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower().strip()
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:4b")


def get_llm(temperature: float = 0.3, max_tokens: int = 1024):
    """Function to choose between local and cloud LLM Providers"""

    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama

        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=LLM_MODEL,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
        )

    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini")
        return ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=max_tokens,
        )

    else:
        supported = ["ollama", "gemini"]
        raise ValueError(
            f"Unknown LLM_PROVIDER='{LLM_PROVIDER}'. "
            f"Supported: {', '.join(supported)}"
        )
