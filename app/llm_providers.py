import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from .config import OLLAMA_MODEL_CONFIGS, GROQ_MODEL_CONFIGS

def get_llm(provider: str = "ollama", model_name: str = None):
    """Initialize LLM based on provider and model"""
    if provider == "ollama":
        model_config = next((m for m in OLLAMA_MODEL_CONFIGS if m["name"] == model_name), OLLAMA_MODEL_CONFIGS[0])
        print("-------------------OLLAMA-----------------------------")

        return ChatOllama(
            base_url=model_config["url"],
            model=model_config["tag"],
            temperature=0.5
        )
    elif provider == "groq":
        print("------------------------------------------------GROQ")
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name or "llama-3.3-70b-versatile",
            temperature=0.5
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Initialize default LLM
default_llm = get_llm() 