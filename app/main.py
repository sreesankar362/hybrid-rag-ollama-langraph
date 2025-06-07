from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .config import API_TITLE, API_DESCRIPTION
from .vector_store import create_hybrid_collection
from .models import QueryRequest, QueryResponse, DocumentRequest
from .endpoints import (
    health_check,
    get_available_models,
    upload_documents,
    upload_pdfs,
    query_documents,
    clear_collection_endpoint,
    test_hybrid_search_endpoint,
    test_retriever_endpoint
)

# Initialize FastAPI app
app = FastAPI(title=API_TITLE, description=API_DESCRIPTION)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    try:
        # Create collection if it doesn't exist
        create_hybrid_collection()
        print("✅ Hybrid RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Startup error: {e}")

# API Endpoints
@app.get("/health")
async def health():
    return await health_check()

@app.get("/models")
async def models():
    return await get_available_models()

@app.post("/upload")
async def upload(documents: List[DocumentRequest]):
    return await upload_documents(documents)

@app.post("/upload-pdfs")
async def upload_pdf_files(files: List[UploadFile] = File(...)):
    return await upload_pdfs(files)

@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    return await query_documents(request)

@app.delete("/clear-collection")
async def clear_collection():
    return await clear_collection_endpoint()

@app.get("/test-hybrid-search")
async def test_hybrid_search(query: str = "AI", limit: int = 4):
    return await test_hybrid_search_endpoint(query, limit)

@app.get("/test-retriever")
async def test_retriever(query: str = "AI", limit: int = 4):
    return await test_retriever_endpoint(query, limit)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 