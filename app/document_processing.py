import io
from typing import List
import PyPDF2
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import CHUNK_SIZE, CHUNK_OVERLAP

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def process_text_document(content: str, metadata: dict = None) -> List[Document]:
    """Process a text document into chunks"""
    if metadata is None:
        metadata = {}
    
    # Split into chunks
    chunks = text_splitter.split_text(content)
    docs = []
    
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata=metadata
        )
        docs.append(doc)
    
    return docs

def process_pdf_content(pdf_content: bytes, filename: str) -> List[Document]:
    """Process PDF content into document chunks"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Split into chunks
    chunks = text_splitter.split_text(text)
    docs = []
    
    for chunk in chunks:
        doc = Document(
            page_content=chunk,
            metadata={"source": filename, "type": "pdf"}
        )
        docs.append(doc)
    
    return docs 