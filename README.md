# 🚀 Production-Ready RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with cutting-edge technologies and production-ready features. Enterprise-grade solution with advanced AI engineering capabilities.

## ✨ Key Features

### 🎯 Advanced Retrieval
- **Hybrid Search**: Combines dense vector search (semantic) with BM25 sparse search (keyword-based)
- **Cross-Encoder Reranking**: Uses state-of-the-art cross-encoder models for improved relevance
- **Metadata Filtering**: Advanced filtering capabilities for precise document retrieval
- **Query Classification**: Automatic query type detection for optimized routing

### 🧠 Quality Assurance
- **Hallucination Detection**: NLI-based detection to identify unsupported claims
- **Context Grounding**: Ensures responses stay grounded in retrieved context
- **Forced Citations**: Every claim is backed by source references with [Source N] format
- **Confidence Scoring**: Multi-signal confidence calculation for answer reliability

### 🔄 LangGraph Orchestration
- **State Management**: Sophisticated workflow with 8-stage pipeline
- **Error Handling**: Graceful degradation and comprehensive error recovery
- **Observability**: Full tracing with LangSmith integration
- **Extensible**: Easy to add new stages or modify workflow

### 🗄️ Production Infrastructure
- **Qdrant Vector DB**: High-performance vector database (not basic FAISS)
- **Redis Caching**: Intelligent caching for improved response times
- **FastAPI Backend**: Modern, async API with automatic documentation
- **Docker Deployment**: Complete containerization with docker-compose

### 📊 Monitoring & Evaluation
- **RAGAS Framework**: Automated evaluation of faithfulness, relevancy, precision, recall
- **Prometheus Metrics**: Real-time system metrics and performance tracking
- **Grafana Dashboards**: Beautiful visualizations for monitoring
- **Custom Metrics**: Query types, confidence scores, processing times

### 🔒 Enterprise Features
- **Rate Limiting**: Prevent API abuse with configurable limits
- **Authentication Ready**: Structure for adding auth middleware
- **Comprehensive Logging**: Structured logging with rotation
- **Health Checks**: Proper health endpoints for orchestration
- **Incremental Indexing**: Update documents without full reindex

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              LangGraph Orchestration                  │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │   │
│  │  │Classify│→ │Retrieve│→ │ Rerank │→ │Generate│    │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │   │
│  │       ↓           ↓           ↓           ↓         │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐    │   │
│  │  │Extract │→ │Halluc. │→ │Ground  │→ │Finalize│    │   │
│  │  │Citations│  │Check   │  │Check   │  │        │    │   │
│  │  └────────┘  └────────┘  └────────┘  └────────┘    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ↓                    ↓                    ↓
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Qdrant  │        │  Redis   │        │  Ollama  │
   │ Vector DB│        │  Cache   │        │   LLM    │
   └──────────┘        └──────────┘        └──────────┘
```

## 📁 Project Structure

```
advanced-rag/
├── src/
│   ├── api/                    # FastAPI endpoints
│   │   ├── main.py            # Main API application
│   │   ├── rate_limiter.py    # Rate limiting
│   │   └── dependencies.py    # Dependency injection
│   ├── core/                   # Core RAG components
│   │   ├── vector_store.py    # Hybrid vector store
│   │   ├── reranker.py        # Cross-encoder reranking
│   │   ├── hallucination_detector.py  # Hallucination detection
│   │   └── langgraph_rag.py   # LangGraph orchestration
│   ├── models/                 # Data models
│   │   └── schemas.py         # Pydantic schemas
│   ├── services/               # Business logic
│   │   ├── document_processor.py  # Document processing
│   │   └── metrics_collector.py   # Metrics collection
│   ├── evaluation/             # Evaluation framework
│   │   └── ragas_evaluator.py # RAGAS integration
│   └── utils/                  # Utilities
│       └── logger.py          # Logging setup
├── config/                     # Configuration
│   └── settings.py            # Settings management
├── tests/                      # Test suite
├── data/                       # Data storage
│   ├── raw/                   # Raw documents
│   ├── processed/             # Processed documents
│   └── vector_store/          # Vector embeddings
├── logs/                       # Application logs
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Multi-container setup
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Ollama (for local LLM) or API access to LLM provider
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
```bash
git clone <your-repo>
cd advanced-rag
```

2. **Set up environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start services with Docker**
```bash
docker-compose up -d
```

5. **Initialize Ollama models** (if using local LLM)
```bash
docker exec -it ollama ollama pull llama3
```

6. **Run the application**
```bash
# Development
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Production
docker-compose up -d rag-api
```

### Access Points
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Grafana**: http://localhost:3000 (admin/admin)
- **Qdrant UI**: http://localhost:6333/dashboard

## 📖 Usage Examples

### Query the RAG System

```python
import requests

# Simple query
response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What are the deployment steps?",
        "top_k": 5,
        "include_sources": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']}")
print(f"Citations: {len(result['citations'])}")
```

### Upload Documents

```python
# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": f},
        data={"metadata": '{"category": "technical"}'}
    )

print(response.json())
```

### Get System Metrics

```python
response = requests.get("http://localhost:8000/metrics")
metrics = response.json()
print(f"Total Queries: {metrics['total_queries']}")
print(f"Avg Response Time: {metrics['average_response_time_ms']}ms")
print(f"Cache Hit Rate: {metrics['cache_hit_rate']:.2%}")
```

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# Retrieval tuning
RETRIEVAL_TOP_K=10              # Initial retrieval count
RERANK_TOP_K=5                  # Final results after reranking
HYBRID_SEARCH_ALPHA=0.5         # 0=BM25 only, 1=vector only

# Quality controls
HALLUCINATION_THRESHOLD=0.8     # Confidence threshold
ENABLE_HALLUCINATION_CHECK=true # Enable/disable detection

# Performance
CHUNK_SIZE=1000                 # Document chunk size
CHUNK_OVERLAP=200               # Overlap between chunks
CACHE_TTL=86400                 # Cache time-to-live (seconds)
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_rag.py -v
```

## 📊 Evaluation

The system includes automated evaluation using RAGAS:

```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

evaluator = RAGASEvaluator()
results = evaluator.evaluate_system(test_queries)

print(f"Faithfulness: {results['faithfulness']:.2f}")
print(f"Answer Relevancy: {results['answer_relevancy']:.2f}")
print(f"Context Precision: {results['context_precision']:.2f}")
```

## 🎯 Advanced Features Implemented

### 1. Hybrid Search
- ✅ Vector similarity (semantic understanding)
- ✅ BM25 keyword search (exact matching)
- ✅ Configurable alpha blending
- ✅ Score normalization

### 2. Reranking
- ✅ Cross-encoder model integration
- ✅ Ensemble scoring (vector + BM25 + cross-encoder)
- ✅ Configurable weights

### 3. Hallucination Control
- ✅ NLI-based entailment checking
- ✅ Claim extraction and verification
- ✅ Context grounding metrics
- ✅ Confidence adjustment

### 4. Citations
- ✅ Automatic source tracking
- ✅ [Source N] format in responses
- ✅ Snippet extraction
- ✅ Relevance scoring per citation

### 5. Observability
- ✅ LangSmith tracing integration
- ✅ Structured logging
- ✅ Prometheus metrics
- ✅ Custom metrics collection

### 6. Production Ready
- ✅ FastAPI backend with async support
- ✅ Docker containerization
- ✅ Health checks
- ✅ Rate limiting
- ✅ Error handling
- ✅ Configuration management

## 🔄 Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f rag-api

# Scale API instances
docker-compose up -d --scale rag-api=3

# Stop services
docker-compose down
```

### Kubernetes (Optional)
```bash
# Apply configurations
kubectl apply -f k8s/

# Check status
kubectl get pods -n rag-system
```

## 📈 Performance Optimization

- **Caching**: Redis-based caching reduces repeated queries
- **Batch Processing**: Efficient document processing
- **Connection Pooling**: Optimized database connections
- **Async Operations**: Non-blocking I/O throughout
- **Resource Limits**: Configured memory and CPU limits

## 🤝 Contributing

This is a portfolio project, but suggestions are welcome!

## 📝 License

MIT License - See LICENSE file for details

## 👤 Author

**Your Name**
- Portfolio: [your-portfolio.com]
- LinkedIn: [your-linkedin]
- GitHub: [@yourusername]

## 🎓 Resume Highlights

This project demonstrates:
- ✅ Advanced RAG architecture beyond basic implementations
- ✅ Production-ready code with proper error handling
- ✅ Modern Python best practices (type hints, async, etc.)
- ✅ Microservices architecture with Docker
- ✅ Comprehensive testing and evaluation
- ✅ Monitoring and observability
- ✅ API design and documentation
- ✅ LangGraph for complex workflows
- ✅ Vector database optimization
- ✅ ML model integration (cross-encoders, NLI)

---

**Note**: This is a production-ready RAG system showcasing advanced features and best practices. It goes far beyond basic RAG implementations with hybrid search, reranking, hallucination detection, comprehensive monitoring, and proper deployment infrastructure.