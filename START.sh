#!/bin/bash

echo "🚀 Starting MgenAI RAG System..."
echo "================================"

# Check if virtual environment exists
if [ ! -d "virenv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv virenv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source virenv/bin/activate

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies (first time only)..."
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file if needed"
fi

# Check if Docker services are running
echo "Checking Docker services..."

# Check if Redis is already running on port 6379
if lsof -i :6379 > /dev/null 2>&1; then
    echo "✅ Redis already running on port 6379"
else
    echo "Starting Redis..."
    docker-compose up -d redis
fi

# Start Qdrant if not running
if ! docker ps | grep -q qdrant; then
    echo "Starting Qdrant..."
    docker-compose up -d qdrant
fi

# Start Ollama if not running
if ! docker ps | grep -q ollama; then
    echo "Starting Ollama..."
    docker-compose up -d ollama
fi

echo "Waiting for services to be ready..."
sleep 10

# Check if Ollama model exists
if ! docker exec ollama ollama list 2>/dev/null | grep -q llama3; then
    echo "Pulling Ollama llama3 model (this may take a while)..."
    docker exec ollama ollama pull llama3
fi

echo ""
echo "✅ All services ready!"
echo ""
echo "Starting API server..."
echo "Access at: http://localhost:8000/docs"
echo ""

# Start the API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Made with Bob
