# 🚀 Quick Start Guide

## Prerequisites
- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM

## Step-by-Step Setup

### 1️⃣ Automated Setup (Recommended)
```bash
# Run the setup script
chmod +x setup.sh
./setup.sh
```

This will:
- ✅ Check Python & Docker
- ✅ Create virtual environment
- ✅ Install dependencies
- ✅ Start Qdrant, Redis, Ollama
- ✅ Pull Ollama llama3 model
- ✅ Create .env file

### 2️⃣ Manual Setup

**Step 1: Clone & Navigate**
```bash
cd /Users/suchi/Desktop/MgenAI-chatbot
```

**Step 2: Create Environment File**
```bash
cp .env.example .env
# Edit .env with your settings (optional: add LANGCHAIN_API_KEY for tracing)
```

**Step 3: Install Python Dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 4: Start Docker Services**
```bash
# Start Qdrant (Vector DB), Redis (Cache), and Ollama (LLM)
docker-compose up -d
```

**Step 5: Pull Ollama Model**
```bash
docker exec ollama ollama pull llama3
```

**Step 6: Start the API**
```bash
# Development mode with auto-reload
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## 🎯 Access Points

Once running, access:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics
- **Qdrant Dashboard**: http://localhost:6333/dashboard

## 📝 First Query

### Using Swagger UI (Browser)
1. Go to http://localhost:8000/docs
2. Click on `/query` endpoint
3. Click "Try it out"
4. Enter your query:
```json
{
  "query": "What is this system about?",
  "top_k": 5,
  "include_sources": true
}
```
5. Click "Execute"

### Using cURL
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is this system about?",
    "top_k": 5,
    "include_sources": true
  }'
```

### Using Python
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is this system about?",
        "top_k": 5,
        "include_sources": True
    }
)

result = response.json()
print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence_score']}")
```

## 📤 Upload Documents

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@/path/to/document.pdf" \
  -F 'metadata={"category": "technical"}'
```

## 🛑 Stop Services

```bash
# Stop all Docker services
docker-compose down

# Stop API (Ctrl+C in terminal)
```

## 🔍 Troubleshooting

### Services not starting?
```bash
# Check Docker status
docker-compose ps

# View logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### Ollama model not found?
```bash
# Pull the model again
docker exec ollama ollama pull llama3

# List available models
docker exec ollama ollama list
```

### Port already in use?
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process or change port in .env
API_PORT=8001
```

## 📊 System Architecture

```
User Request
    ↓
FastAPI (src/api/main.py)
    ↓
LangGraph RAG (src/core/langgraph_rag.py)
    ↓
├── Redis Cache Check
├── Query Rewriting
├── Vector Search (Qdrant)
├── Reranking
├── LLM Generation (Ollama)
├── Hallucination Detection
└── Response Formatting
    ↓
JSON Response
```

## 🎓 Next Steps

1. **Upload your documents**: Use `/documents/upload` endpoint
2. **Configure settings**: Edit `.env` for your needs
3. **Monitor metrics**: Check `/metrics` endpoint
4. **View traces**: Add LANGCHAIN_API_KEY for LangSmith tracing
5. **Scale up**: Use `docker-compose up -d --scale rag-api=3`

## 💡 Tips

- First query may be slow (model loading)
- Subsequent queries use cache (faster)
- Check logs in `logs/` directory
- Use `include_sources: true` for citations
- Adjust `top_k` for more/fewer results

---

**Need help?** Check the main README.md or logs/app.log