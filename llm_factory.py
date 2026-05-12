"""
Factory pentru LLM local servit prin LM Studio.

LM Studio expune un endpoint compatibil OpenAI pe localhost, ceea ce ne
permite sa schimbam provider-ul fara sa modificam pipeline-urile RAG/Quiz.
"""

from langchain_openai import ChatOpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL


def creeaza_chat_llm(temperature: float) -> ChatOpenAI:
    """Returneaza un client ChatOpenAI configurat pentru LM Studio local."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=temperature,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
    )


def model_suporta_input_vizual(model_name: str | None = None) -> bool:
    """Heuristica simpla pentru a decide daca modelul accepta imagini."""
    candidate = (model_name or LLM_MODEL).lower()
    indicatori = (
        "vl",
        "vision",
        "gemma-4",
        "llava",
        "gpt-4o",
        "qwen2.5-vl",
    )
    return any(indicator in candidate for indicator in indicatori)
