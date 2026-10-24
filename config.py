import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
EXA_API_KEY = os.getenv("EXA_API_KEY")

DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/checkpoints.sqlite")


def get_llm():
    """Return a Groq-backed chat model instance."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY in .env "
            "(sign up at console.groq.com)"
        )
    from langchain_groq import ChatGroq
    return ChatGroq(model=LLM_MODEL, temperature=0.3, api_key=GROQ_API_KEY)
