import os
import requests
import logging
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)

# API Configuration
API_TITLE = "RAG API"
API_DESCRIPTION = "RAG system with hybrid search"

# Qdrant Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# Collection Configuration
COLLECTION_NAME = "hybrid_documents"

# Text Splitter Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_LIST_LLMS = f"{OLLAMA_URL}/api/tags"

def get_ollama_models() -> List[Dict]:
    """Dynamically fetch available Ollama models"""
    try:
        response = requests.get(OLLAMA_LIST_LLMS, timeout=10)
        response.raise_for_status()
        ollama_models = response.json().get("models", [])
        
        chat_model_configs = []
        
        # Process Ollama models
        for model in ollama_models:
            model_name = model.get("name", "")
            if model_name:
                # Clean up model name for display
                display_name = model_name.replace(":", " ").title()
                
                chat_model_configs.append({
                    "name": display_name,
                    "tag": model_name,
                    "provider": "ollama",
                    "is_active": True,
                    "url": OLLAMA_URL
                })
        
        return chat_model_configs
        
    except Exception as e:
        logger.error(f"Error fetching Ollama models: {str(e)}")
        # Return fallback model if Ollama is not available
        return [{
            "name": "Llama 3.2",
            "tag": "llama3.2",
            "provider": "ollama",
            "is_active": True,
            "url": OLLAMA_URL
        }]

# Dynamic Ollama model configurations
OLLAMA_MODEL_CONFIGS = get_ollama_models()

GROQ_MODEL_CONFIGS = [
    {"name": "G-QwQ 32B", "tag": "qwen-qwq-32b", "provider": "groq", "is_active": True},
    {"name": "G-DeepSeek R1 Distill Llama 70B", "tag": "deepseek-r1-distill-llama-70b", "provider": "groq", "is_active": True},
    {"name": "G-Llama 3.3 70B", "tag": "llama-3.3-70b-versatile", "provider": "groq", "is_active": True},
    {"name": "G-Llama 4 Maverick 17B 128E", "tag": "meta-llama/llama-4-maverick-17b-128e-instruct", "provider": "groq", "is_active": True},
    {"name": "G-Mistral Saba 24B", "tag": "mistral-saba-24b", "provider": "groq", "is_active": True}
]

# System Templates
SYSTEM_TEMPLATE = """
You are an expert QA Assistant with deep analytical capabilities. Your role is to provide comprehensive, well-structured answers using only the provided context as your source of information.

INSTRUCTIONS:
1. Analyze the provided context thoroughly and extract all relevant information
2. Provide detailed, comprehensive answers that cover multiple aspects of the question
3. Structure your response with clear sections and bullet points when appropriate
4. Include specific details, examples, and explanations from the context
5. If there are multiple perspectives or approaches mentioned in the context, present them all
6. Explain the reasoning behind your conclusions
7. If the question cannot be fully answered from the context, clearly state what information is missing
8. Always cite or reference specific parts of the context when making claims

RESPONSE FORMAT:
- Start with a brief summary of your answer
- Provide detailed explanations with supporting evidence from the context
- Use bullet points, numbered lists, or sections to organize complex information
- End with any relevant caveats or additional considerations

HANDLING EMPTY OR INSUFFICIENT CONTEXT:
If no context is provided or the context is empty, respond with:
"I don't have any relevant information in my knowledge base to answer your question about [topic]. Could you please:
1. Upload documents related to your question
2. Provide more specific details about what you're looking for
3. Clarify or rephrase your question

Once you provide relevant documents, I'll be able to give you a comprehensive answer based on that information."

If the context exists but doesn't contain sufficient information to answer the question, respond with:
"I don't have enough information in the provided context to answer this question comprehensively. The available documents don't seem to cover [specific topic]. Could you please provide more relevant documents or clarify what specific aspect you'd like me to focus on?"
"""

# Template for when no context is available
NO_CONTEXT_TEMPLATE = """
You are a helpful RAG assistant. The user has asked a question but no relevant documents were found in the knowledge base.

Your task is to politely inform the user that you don't have the necessary information and guide them on next steps.

Respond with:
"I don't have any relevant information in my knowledge base to answer your question about '{query}'. 

To help you better, could you please:
1. 📄 Upload documents related to your question using the sidebar
2. 🔍 Provide more specific details about what you're looking for
3. ✏️ Clarify or rephrase your question

Once you provide relevant documents, I'll be able to give you a comprehensive answer based on that information!"

Be helpful, friendly, and encouraging while clearly explaining the limitation.
"""

HUMAN_TEMPLATE = """
CONTEXT INFORMATION:
{context_str}

QUESTION: {query}

ANALYSIS INSTRUCTIONS:
Please provide a comprehensive answer to the question above using the context information provided. 

IMPORTANT: First check if the context is empty or contains no relevant information.

If context is empty or no relevant documents were found:
- Acknowledge that you don't have the necessary information
- Ask the user to upload relevant documents
- Suggest they clarify or rephrase their question
- Be helpful and guide them on next steps

If context exists and is relevant:
Your response should:
1. Be thorough and detailed, covering all relevant aspects mentioned in the context
2. Include specific examples, data points, or details from the context
3. Explain any processes, relationships, or concepts that are relevant
4. Address potential implications or applications if mentioned in the context
5. Organize the information in a clear, logical structure
6. Reference specific parts of the context to support your points

If the context doesn't contain sufficient information to fully answer the question, please:
- Explain what aspects you can address based on the available context
- Clearly identify what information is missing or would be needed for a complete answer
- Provide the best possible answer with the available information
- Ask for clarification or additional documents

Remember: Base your entire response solely on the provided context. Do not add information from outside sources.
"""

# Template for when context is available
CONTEXT_HUMAN_TEMPLATE = """
CONTEXT INFORMATION:
{context_str}

QUESTION: {query}

ANALYSIS INSTRUCTIONS:
Please provide a comprehensive answer to the question above using the context information provided. 

Your response should:
1. Be thorough and detailed, covering all relevant aspects mentioned in the context
2. Include specific examples, data points, or details from the context
3. Explain any processes, relationships, or concepts that are relevant
4. Address potential implications or applications if mentioned in the context
5. Organize the information in a clear, logical structure
6. Reference specific parts of the context to support your points

If the context doesn't contain sufficient information to fully answer the question, please:
- Explain what aspects you can address based on the available context
- Clearly identify what information is missing or would be needed for a complete answer
- Provide the best possible answer with the available information

Remember: Base your entire response solely on the provided context. Do not add information from outside sources.
""" 