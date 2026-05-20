#!/bin/bash

echo "🚀 Starting RAG System with Streamlit UI..."
echo "=========================================="

# Activate virtual environment
source virenv/bin/activate

# Install streamlit if not installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "Installing Streamlit..."
    pip install streamlit pandas
fi

# Check if API is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo ""
    echo "⚠️  API is not running!"
    echo "Please start the API first in another terminal:"
    echo "  source virenv/bin/activate"
    echo "  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    read -p "Press Enter once API is running..."
fi

echo ""
echo "✅ Starting Streamlit UI..."
echo "Access at: http://localhost:8501"
echo ""

# Run Streamlit
streamlit run streamlit_app.py

# Made with Bob
