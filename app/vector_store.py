from typing import List
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, models
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_core.documents import Document
from fastapi import HTTPException

from .config import QDRANT_URL, COLLECTION_NAME

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize embedding models
dense_embedding_model = TextEmbedding("thenlper/gte-large")
sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")

# Global variables
collection_exists = False

def create_hybrid_collection():
    """Create a collection with hybrid vector configuration"""
    global collection_exists
    
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"Collection {COLLECTION_NAME} already exists")
            collection_exists = True
            return True
            
        # Create collection with hybrid vectors
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "thenlper/gte-large": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "miniCOIL": models.SparseVectorParams(modifier=models.Modifier.IDF),
            }
        )
        
        print(f"Created hybrid collection: {COLLECTION_NAME}")
        collection_exists = True
        return True
        
    except Exception as e:
        print(f"Error creating collection: {e}")
        return False

def index_documents_hybrid(documents: List[Document]):
    """Index documents with both dense and sparse embeddings"""
    if not collection_exists:
        if not create_hybrid_collection():
            raise HTTPException(status_code=500, detail="Failed to create collection")
    
    try:
        # Generate embeddings
        texts = [doc.page_content for doc in documents]
        dense_embeddings = list(dense_embedding_model.embed(texts))
        sparse_embeddings = list(sparse_embedding_model.embed(texts))
        
        # Create points for Qdrant
        points = []
        for idx, (dense_emb, sparse_emb, doc) in enumerate(zip(dense_embeddings, sparse_embeddings, documents)):
            point_id = str(uuid.uuid4())
            point = PointStruct(
                id=point_id,
                vector={
                    "thenlper/gte-large": dense_emb,
                    "miniCOIL": sparse_emb.as_object(),
                },
                payload={
                    "document": doc.page_content,
                    "metadata": doc.metadata
                }
            )
            points.append(point)
        
        # Upsert to Qdrant
        operation_info = qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        
        print(f"Indexed {len(points)} documents with hybrid embeddings")
        return len(points)
        
    except Exception as e:
        print(f"Error indexing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to index documents: {str(e)}")

def hybrid_search(query: str, limit: int = 4) -> List[Document]:
    """Perform hybrid search with prefetch"""
    if not collection_exists:
        raise HTTPException(status_code=503, detail="Collection not available")
    
    try:
        # Generate query embeddings
        dense_vector = next(dense_embedding_model.query_embed(query))
        sparse_vector = next(sparse_embedding_model.query_embed(query))
        
        # Create prefetch queries
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="thenlper/gte-large",
                limit=20,
            ),
            models.Prefetch(
                query=models.SparseVector(**sparse_vector.as_object()),
                using="miniCOIL",
                limit=20,
            )
        ]
        
        # Perform hybrid search with re-ranking
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=dense_vector,
            using="thenlper/gte-large",
            with_payload=True,
            limit=limit,
        )
        print(results)
        # Convert to Document objects
        retrieved_docs = []
        for point in results.points:
            doc = Document(
                page_content=point.payload.get("document", ""),
                metadata=point.payload.get("metadata", {})
            )
            retrieved_docs.append(doc)
        
        return retrieved_docs
        
    except Exception as e:
        print(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def clear_collection():
    """Clear all documents from the collection"""
    global collection_exists
    
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            return {
                "message": "Collection does not exist",
                "status": "success"
            }
        
        # Delete the collection
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
        collection_exists = False
        
        # Recreate the collection
        if create_hybrid_collection():
            return {
                "message": f"Successfully cleared collection '{COLLECTION_NAME}' and recreated it",
                "status": "success"
            }
        else:
            return {
                "message": f"Cleared collection '{COLLECTION_NAME}' but failed to recreate it",
                "status": "warning"
            }
            
    except Exception as e:
        print(f"Error clearing collection: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear collection: {str(e)}")

def get_collection_info():
    """Get collection information for health checks"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        return {
            "collections": collection_names,
            "hybrid_collection_exists": COLLECTION_NAME in collection_names
        }
    except Exception as e:
        raise Exception(f"Failed to get collection info: {str(e)}") 