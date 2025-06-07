import re
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from .vector_store import hybrid_search
from .llm_providers import get_llm
from .config import SYSTEM_TEMPLATE, HUMAN_TEMPLATE, NO_CONTEXT_TEMPLATE, CONTEXT_HUMAN_TEMPLATE

# LangGraph State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    provider: str
    model_name: Optional[str]

def search(state: State):
    """Search function for LangGraph"""
    try:
        retrieved_docs = hybrid_search(state["question"], limit=4)
        return {"context": retrieved_docs}
    except Exception as e:
        print(f"Search error: {e}")
        return {"context": []}

def generate(state: State):
    """Generate function for LangGraph"""
    try:
        # Get provider and model from state
        provider = state.get("provider", "ollama")
        model_name = state.get("model_name")
        
        print(f"🔧 Generate function - Provider: {provider}, Model: {model_name}")
        
        # Get LLM instance
        current_llm = get_llm(provider, model_name)
        
        # Check if context is available and has meaningful content
        context_docs = state.get("context", [])
        has_context = bool(context_docs and any(doc.page_content.strip() for doc in context_docs))
        
        if has_context:
            # Use RAG prompt with context
            docs_content = "\n\n".join(doc.page_content for doc in context_docs)
            
            messages = [
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": CONTEXT_HUMAN_TEMPLATE.format(
                    context_str=docs_content, 
                    query=state["question"]
                )},
            ]
        else:
            # Use no-context prompt
            messages = [
                {"role": "system", "content": NO_CONTEXT_TEMPLATE},
                {"role": "user", "content": f"Question: {state['question']}"},
            ]
        
        print(f"📝 Using {'RAG prompt with context' if has_context else 'no-context prompt'}")
        
        response = current_llm.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        print(f"Generate error: {e}")
        return {"answer": f"Error generating response: {str(e)}"}

def extract_after_think(input_text: str) -> str:
    """Extract content after </think> tag"""
    match = re.search(r'</think>(.*)', input_text, re.DOTALL)
    return match.group(1).strip() if match else input_text

# Define the graph
def create_graph():
    """Create and compile the LangGraph workflow"""
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("search", search)
    graph_builder.add_node("generate", generate)
    
    # Add edges
    graph_builder.add_edge("search", "generate")
    graph_builder.add_edge("generate", END)
    
    # Add entrypoint
    graph_builder.add_edge(START, "search")
    
    # Compile the graph
    return graph_builder.compile()

# Create the graph instance
graph = create_graph() 