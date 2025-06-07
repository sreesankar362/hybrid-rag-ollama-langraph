# RAG System with FastAPI, Streamlit, and LangGraph

This project implements a complete Retrieval-Augmented Generation (RAG) system using FastAPI, Streamlit, LangGraph, Ollama, and Qdrant with binary quantization. This implementation features **fully local deployment** with **no API keys required**.

## Features

- 🚀 **FastAPI backend** with LangGraph for RAG pipeline orchestration
- 🎨 **Streamlit frontend** for intuitive user interaction
- 📄 **PDF upload and processing** functionality
- 🗄️ **Local Qdrant vector database** with binary quantization
- 🤖 **Local Ollama LLM integration** (no API key required)
- 🔍 **FastEmbed embeddings** for efficient vector search
- 🎯 **Hybrid search** with Maximal Marginal Relevance (MMR)
- 🧠 **Smart context handling** with dynamic prompt selection
- 🔄 **Dynamic model discovery** - automatically detects available Ollama models
- 💬 **Enhanced chat interface** with provider selection
- 📊 **Comprehensive system monitoring** and health checks

## New Features (Latest Updates)

### 🤖 **Local Ollama Deployment**

- Fully containerized Ollama service
- Automatic model pulling on startup
- No external dependencies

### 🔄 **Dynamic Model Discovery**

- Automatically detects all available Ollama models
- No hardcoded model configurations
- Real-time model list updates

### 🧠 **Intelligent Context Handling**

- Programmatically detects document availability
- Uses different prompts based on context presence
- Helpful guidance when no documents are found

### 🎯 **Enhanced Provider Selection**

- Choose between Ollama and Groq providers
- Model-specific selection within each provider
- Proper backend routing based on UI selection

## Prerequisites

- Docker
- Docker Compose

## Setup and Running

1. **Clone the repository:**

```bash
git clone <repository-url>
cd miniCOIL-RAG
```

2. **Start the services:**

```bash
docker-compose up --build
```

This will automatically:

- Start Qdrant vector database
- Deploy Ollama service locally
- Pull the llama3.2 model
- Start the FastAPI backend
- Launch the Streamlit frontend

3. **Access the applications:**

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## Usage

### 1. **System Health Check**

- Use the "🔍 Health Check" button in the sidebar
- Verifies connectivity to Ollama and Qdrant
- Shows system status and available models

### 2. **Model Selection**

- **Provider Selection**: Choose between Ollama (local) or Groq (cloud)
- **Model Selection**: Pick from available models in each provider
- **Dynamic Discovery**: Ollama models are automatically detected

### 3. **Document Management**

**PDF Upload:**

- Use the "📄 Upload PDF Files" section
- Select multiple PDF files
- Automatic processing and indexing

**Manual Document Upload:**

- Use the "✏️ Add Text Document" section
- Enter document text and optional metadata
- JSON format metadata support

**Document Clearing:**

- "🗑️ Clear All Documents" button
- Removes all indexed documents
- Confirmation dialog for safety

### 4. **Chat Interface**

- **Smart Responses**: Different prompts based on document availability
- **Context-Aware**: Uses RAG when documents are available
- **Helpful Guidance**: Provides instructions when no documents are found
- **Source Display**: Shows retrieved document sources
- **Retriever Logs**: Optional display of search results

## Architecture

### **Local-First Design**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │    FastAPI      │    │     Qdrant      │
│   Frontend      │◄──►│    Backend      │◄──►│  Vector Store   │
│  (Port 8501)    │    │  (Port 8000)    │    │  (Port 6333)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Ollama      │
                       │   LLM Service   │
                       │  (Port 11434)   │
                       └─────────────────┘
```

### **LangGraph Pipeline**

- **Search Node**: Hybrid search with MMR for diverse results
- **Generate Node**: Context-aware response generation
- **Smart Routing**: Different prompts based on context availability

### **Components**

- **FastEmbed**: Efficient 1024-dimensional embeddings
- **Binary Quantization**: Optimized vector storage
- **Hybrid Search**: Combines semantic and keyword search
- **Dynamic Models**: Auto-discovery of available models

## API Endpoints

### **Core Endpoints**

- `GET /health` - System health and status
- `GET /models` - Available models from all providers
- `POST /query` - Query documents with provider/model selection
- `POST /upload` - Add documents manually
- `POST /upload-pdfs` - Upload and process PDF files
- `DELETE /clear-collection` - Clear all documents

### **Testing Endpoints**

- `GET /test-hybrid-search` - Test hybrid search functionality
- `GET /test-retriever` - Test basic retrieval

## Configuration

### **Environment Variables**

```bash
# Qdrant Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Ollama Configuration (Local)
OLLAMA_URL=http://ollama:11434

# Groq Configuration (Optional)
GROQ_API_KEY=your_groq_api_key
GROQ_URL=https://api.groq.com/openai/v1
```

### **Docker Services**

```yaml
services:
  app: # FastAPI + Streamlit
  qdrant: # Vector database
  ollama: # Local LLM service
```

## Model Management

### **Ollama Models**

- **Automatic Discovery**: All pulled models are automatically available
- **Dynamic Loading**: No configuration file updates needed
- **Model Pulling**: Add new models with `docker-compose exec ollama ollama pull <model>`

### **Supported Providers**

- **Ollama**: Local inference (recommended)
- **Groq**: Cloud inference (requires API key)

## Smart Context Handling

### **When Documents Are Available**

- Uses comprehensive RAG prompts
- Provides detailed, context-based responses
- Shows source documents and metadata

### **When No Documents Are Found**

- Uses helpful guidance prompts
- Suggests document upload steps
- Asks for question clarification
- Provides clear next steps

## Benefits

### **🔒 Privacy & Security**

- **Local Processing**: Documents never leave your infrastructure
- **No API Keys**: Core functionality works without external services
- **Data Control**: Complete control over your data

### **💰 Cost Effective**

- **No Per-Token Charges**: Local inference is free
- **Scalable**: Add more models without additional costs
- **Efficient**: Optimized with binary quantization

### **🛠️ Developer Friendly**

- **Easy Setup**: Single command deployment
- **Extensible**: Add new providers and models easily
- **Well Documented**: Comprehensive API documentation

## Development

### **File Structure**

```
miniCOIL-RAG/
├── app/
│   ├── config.py          # Configuration and dynamic model loading
│   ├── endpoints.py       # FastAPI route handlers
│   ├── graph.py          # LangGraph pipeline with smart context handling
│   ├── llm_providers.py  # Provider abstraction layer
│   ├── models.py         # Pydantic models
│   ├── streamlit_app.py  # Frontend interface
│   └── vector_store.py   # Qdrant integration
├── docker-compose.yml    # Service orchestration
├── Dockerfile           # Application container
├── requirements.txt     # Python dependencies
├── .dockerignore       # Docker build exclusions
├── .gitignore         # Git exclusions
└── README.md          # This file
```

### **Adding New Models**

1. Pull model in Ollama: `ollama pull <model-name>`
2. Restart the application
3. Model appears automatically in the UI

### **Adding New Providers**

1. Update `llm_providers.py`
2. Add configuration in `config.py`
3. Update UI provider selection

## Troubleshooting

### **Common Issues**

- **Ollama Connection**: Ensure Ollama service is running
- **Model Not Found**: Check if model is pulled in Ollama
- **Memory Issues**: Ensure sufficient RAM for models
- **Port Conflicts**: Check if ports 8000, 8501, 6333, 11434 are available

### **Health Check**

Use the health check endpoint to verify all services are running correctly.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.
