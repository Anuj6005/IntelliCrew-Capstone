"""LLM adapter: the rest of rag_service only needs callable(prompt: str) -> str."""


def _to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):          # newer SDKs return a list of parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def gemini_llm(model: str = None, api_key: str = None, temperature: float = 0.0):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from config import settings                       # IntelliCrew's existing settings
    chat = ChatGoogleGenerativeAI(
        model=model or settings.GEMINI_MODEL,
        temperature=temperature,
        google_api_key=api_key or settings.GOOGLE_API_KEY,
    )
    return lambda prompt: _to_text(chat.invoke(prompt).content)
