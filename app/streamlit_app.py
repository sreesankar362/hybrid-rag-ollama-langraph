import streamlit as st
import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# API endpoint
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="RAG Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = 0

if "show_retriever_logs" not in st.session_state:
    st.session_state.show_retriever_logs = True

if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = "ollama"

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .thinking-process {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-style: italic;
    }
    .retriever-logs {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
        font-size: 0.9em;
    }
    .retrieved-doc {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.85em;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .model-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for document management and system status
with st.sidebar:
    st.title("🤖 RAG Chat Assistant")
    st.caption("Powered by Ollama + Qdrant + LangGraph")
    
    # System Status
    st.header("📊 System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Health Check", use_container_width=True):
            try:
                response = requests.get(f"{API_URL}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    if health_data["status"] == "healthy":
                        st.success("✅ System Online")
                        st.info(f"📚 Collections: {health_data['collections']}")
                    else:
                        st.error("❌ System Issues")
                else:
                    st.error("❌ API Offline")
            except Exception as e:
                st.error(f"❌ Connection Error")
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # Model Selection
    st.header("🤖 Model Selection")
    
    # Provider Selection
    provider = st.radio(
        "Select Provider",
        ["ollama", "groq"],
        index=0 if st.session_state.selected_provider == "ollama" else 1,
        key="provider_selector"
    )
    
    # Get available models
    try:
        models_response = requests.get(f"{API_URL}/models")
        if models_response.status_code == 200:
            available_models = models_response.json()
            
            # Model Selection based on provider
            if provider == "ollama":
                models = available_models["ollama"]
            else:
                models = available_models["groq"]
            
            # Create model selection dropdown
            model_options = {model["name"]: model["tag"] for model in models if model["is_active"]}
            selected_model_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                index=0
            )
            
            # Update session state
            st.session_state.selected_provider = provider
            st.session_state.selected_model = model_options[selected_model_name]
            
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    # Retriever Settings
    st.header("🔧 Retriever Settings")
    
    show_logs = st.checkbox(
        "📋 Show Retriever Logs",
        value=st.session_state.show_retriever_logs,
        help="Show which documents were retrieved for each query"
    )
    st.session_state.show_retriever_logs = show_logs
    
    retriever_limit = st.slider(
        "📄 Documents to Retrieve", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="Number of documents to retrieve for context"
    )
    
    st.divider()
    
    # Document Upload Section
    st.header("📁 Document Management")
    
    # Clear All Documents Button
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear All Documents", use_container_width=True, type="secondary"):
            if st.session_state.documents_uploaded > 0:
                # Show confirmation dialog
                if st.session_state.get("confirm_clear", False):
                    with st.spinner("Clearing all documents..."):
                        try:
                            response = requests.delete(f"{API_URL}/clear-collection")
                            if response.status_code == 200:
                                result = response.json()
                                st.success("✅ All documents cleared!")
                                st.session_state.documents_uploaded = 0
                                # Add system message to chat
                                st.session_state.messages.append({
                                    "role": "system",
                                    "content": "🗑️ All documents have been cleared from the collection."
                                })
                                st.session_state.confirm_clear = False
                                st.rerun()
                            else:
                                st.error(f"❌ Failed to clear documents: {response.json().get('detail', 'Unknown error')}")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                        st.session_state.confirm_clear = False
                else:
                    st.warning("⚠️ This will delete ALL documents. Click again to confirm.")
                    st.session_state.confirm_clear = True
                    st.rerun()
            else:
                st.info("ℹ️ No documents to clear")
    
    with col2:
        st.metric("📄 Total Documents", st.session_state.documents_uploaded)
    
    st.divider()
    
    # PDF Upload
    with st.expander("📄 Upload PDF Files", expanded=True):
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if st.button("📤 Upload PDFs", use_container_width=True):
            if uploaded_files:
                with st.spinner("Uploading and processing PDFs..."):
                    try:
                        files = []
                        for uploaded_file in uploaded_files:
                            files.append(("files", (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")))
                        
                        response = requests.post(f"{API_URL}/upload-pdfs", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Uploaded {result['files_processed']} files!")
                            st.info(f"📄 Created {result['chunks_created']} chunks")
                            st.session_state.documents_uploaded += result['chunks_created']
                            # Add system message to chat
                            st.session_state.messages.append({
                                "role": "system",
                                "content": f"📄 Successfully uploaded {result['files_processed']} PDF files ({result['chunks_created']} chunks). You can now ask questions about them!"
                            })
                            st.rerun()
                        else:
                            st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please select PDF files first")
    
    # Manual Document Input
    with st.expander("✏️ Add Text Document"):
        document_text = st.text_area("Document content", height=150, key="manual_doc")
        metadata_text = st.text_area("Metadata (JSON)", height=80, key="manual_meta", 
                                   placeholder='{"source": "manual", "title": "My Document"}')
        
        if st.button("➕ Add Document", use_container_width=True):
            if document_text.strip():
                try:
                    metadata_dict = json.loads(metadata_text) if metadata_text.strip() else {}
                    document = {
                        "content": document_text,
                        "metadata": metadata_dict
                    }
                    
                    response = requests.post(f"{API_URL}/upload", json=[document])
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Document added!")
                        st.info(f"📄 Created {result['chunks_created']} chunks")
                        st.session_state.documents_uploaded += result['chunks_created']
                        # Add system message to chat
                        st.session_state.messages.append({
                            "role": "system",
                            "content": f"📝 Successfully added a text document ({result['chunks_created']} chunks). You can now ask questions about it!"
                        })
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {response.json().get('detail', 'Unknown error')}")
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON in metadata")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter document content")

# Main chat interface
st.title("💬 Chat with Your Documents")

# Display chat messages
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        # Welcome message
        st.markdown("""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>
            Welcome! I'm your RAG assistant. Upload some documents using the sidebar, then ask me questions about them.
            <br><br>
            <strong>What I can help with:</strong>
            <ul>
                <li>📄 Answer questions about uploaded PDFs</li>
                <li>🔍 Search through document content</li>
                <li>📊 Summarize information</li>
                <li>🔗 Find relationships between concepts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        
        elif message["role"] == "assistant":
            content = message["content"]
            
            # Check if there are retriever logs
            retriever_logs = message.get("retriever_logs", None)
            
            # Check if there's thinking process
            if "<think>" in content and "</think>" in content:
                think_start = content.find("<think>") + len("<think>")
                think_end = content.find("</think>")
                thinking = content[think_start:think_end].strip()
                answer = content[think_end + len("</think>"):].strip()
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    <div class="thinking-process">
                        <strong>🤔 Thinking:</strong> {thinking}
                    </div>
                    {answer}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            
            # Display retriever logs if available and enabled
            if retriever_logs and st.session_state.show_retriever_logs:
                with st.expander("📋 Retriever Logs", expanded=False):
                    st.markdown(f"**🔍 Query:** {retriever_logs['query']}")
                    st.markdown(f"**📊 Retrieved:** {len(retriever_logs['documents'])} documents")
                    
                    for i, doc in enumerate(retriever_logs['documents'], 1):
                        st.markdown(f"""
                        <div class="retrieved-doc">
                            <strong>📄 Document {i}:</strong><br>
                            <strong>Content:</strong> {doc['content'][:300]}{'...' if len(doc['content']) > 300 else ''}<br>
                            <strong>Metadata:</strong> {doc['metadata']}<br>
                            <strong>Score:</strong> {doc.get('score', 'N/A')}
                        </div>
                        """, unsafe_allow_html=True)
        
        elif message["role"] == "system":
            st.markdown(f"""
            <div class="chat-message" style="background-color: #e8f5e8; border-left: 4px solid #4caf50;">
                <strong>🔔 System:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

# Chat input
st.markdown("---")

# Create columns for input and button
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.text_input(
        "Type your message...", 
        key="chat_input",
        placeholder="Ask me anything about your documents...",
        label_visibility="collapsed"
    )

with col2:
    send_button = st.button("📤 Send", use_container_width=True, type="primary")

# Handle message sending
if send_button and user_input.strip():
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show typing indicator
    with st.spinner("🤖 Assistant is thinking..."):
        try:
            # First, get retriever results if logs are enabled
            retriever_logs = None
            if st.session_state.show_retriever_logs:
                try:
                    retriever_response = requests.get(
                        f"{API_URL}/test-hybrid-search", 
                        params={"query": user_input, "limit": retriever_limit}
                    )
                    if retriever_response.status_code == 200:
                        retriever_data = retriever_response.json()
                        retriever_logs = {
                            "query": user_input,
                            "documents": [
                                {
                                    "content": doc["page_content"],
                                    "metadata": doc["metadata"],
                                    "score": doc.get("score", "N/A")
                                }
                                for doc in retriever_data["results"]
                            ]
                        }
                except Exception as e:
                    print(f"Error getting retriever logs: {e}")
            
            # Send query to API with provider and model
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "question": user_input,
                    "provider": st.session_state.selected_provider,
                    "model_name": st.session_state.selected_model
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add assistant response with retriever logs
                assistant_message = {
                    "role": "assistant",
                    "content": result["answer"]
                }
                
                if retriever_logs:
                    assistant_message["retriever_logs"] = retriever_logs
                
                st.session_state.messages.append(assistant_message)
            else:
                st.error("❌ Failed to get response from AI")
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Rerun to update chat
    st.rerun()

# Auto-scroll to bottom (JavaScript injection)
st.markdown("""
<script>
    var element = window.parent.document.querySelector('.main');
    element.scrollTop = element.scrollHeight;
</script>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8em;">
    🤖 RAG Chat Assistant | Powered by Ollama, Qdrant & LangGraph | No API keys required
</div>
""", unsafe_allow_html=True) 