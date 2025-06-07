import os
from typing import List
from fastapi import HTTPException, UploadFile, File
from .models import QueryRequest, QueryResponse, DocumentRequest
from .config import GROQ_MODEL_CONFIGS, QDRANT_URL, get_ollama_models, OLLAMA_URL
from .vector_store import (
    index_documents_hybrid, 
    hybrid_search, 
    clear_collection, 
    get_collection_info
)
from .document_processing import process_text_document, process_pdf_content
from .graph import graph, extract_after_think
from .llm_providers import default_llm

async def health_check():
    """Health check endpoint"""
    try:
        # Test LLM connection
        test_response = default_llm.invoke([{"role": "user", "content": "Hello"}])
        
        # Test Qdrant connection and get collection info
        collection_info = get_collection_info()
        
        return {
            "status": "healthy",
            "ollama_url": OLLAMA_URL,
            "qdrant_url": QDRANT_URL,
            **collection_info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

async def get_available_models():
    """Get available models from all providers"""
    # Dynamically fetch Ollama models
    ollama_models = get_ollama_models()
    
    return {
        "ollama": ollama_models,
        "groq": GROQ_MODEL_CONFIGS
    }

async def upload_documents(documents: List[DocumentRequest]):
    """Upload and index documents"""
    try:
        all_docs = []
        for doc_req in documents:
            docs = process_text_document(doc_req.content, doc_req.metadata)
            all_docs.extend(docs)
        
        # Index documents
        num_indexed = index_documents_hybrid(all_docs)
        
        return {
            "message": f"Successfully indexed {num_indexed} document chunks",
            "chunks_created": num_indexed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Upload and process PDF files"""
    try:
        all_docs = []
        
        for file in files:
            if file.content_type != "application/pdf":
                continue
                
            # Read PDF content
            content = await file.read()
            docs = process_pdf_content(content, file.filename)
            all_docs.extend(docs)
        
        # Index all documents
        num_indexed = index_documents_hybrid(all_docs)
        
        return {
            "message": f"Successfully processed {len(files)} PDF files and indexed {num_indexed} chunks",
            "files_processed": len(files),
            "chunks_created": num_indexed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def query_documents(request: QueryRequest) -> QueryResponse:
    """Query documents using hybrid search and LLM"""
    try:
        # Use LangGraph to process the query
        response = graph.invoke({
            "question": request.question,
            "provider": request.provider or "ollama",  # Use provider from request
            "model_name": request.model_name,          # Use model from request
            "context": [],         # Will be filled by search node
            "answer": ""           # Will be filled by generate node
        })
        
        answer = response.get("answer", "No answer generated")
        
        # Get sources from context
        sources = []
        if "context" in response:
            sources = [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response["context"]
            ]
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def clear_collection_endpoint():
    """Clear all documents from the collection"""
    try:
        result = clear_collection()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def test_hybrid_search_endpoint(query: str = "AI", limit: int = 4):
    """Test hybrid search functionality"""
    try:
        results = hybrid_search(query, limit=limit)
        
        return {
            "query": query,
            "limit": limit,
            "results": [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def test_retriever_endpoint(query: str = "AI", limit: int = 4):
    """Test basic retriever functionality"""
    try:
        # Use hybrid search as the main retriever
        results = hybrid_search(query, limit=limit)
        
        return {
            "query": query,
            "limit": limit,
            "results": [
                {
                    "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 