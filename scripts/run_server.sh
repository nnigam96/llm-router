#!/bin/bash
# Script to run the LLM-Router FastAPI server

set -e  # Exit on error

echo "🚀 Starting LLM Router API Server"
echo "================================="
echo ""

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama server is not running."
    echo "Please start Ollama first:"
    echo "  ollama serve"
    echo ""
    echo "Or run the setup script: ./scripts/setup_ollama.sh"
    exit 1
fi

echo "✅ Ollama server is running"
echo ""

# Check if Python dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ Error: Dependencies not installed."
    echo ""
    echo "Please install dependencies first:"
    echo "  pip install -e ."
    echo "  or"
    echo "  pip install -e '.[dev]'  # with dev dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Run the FastAPI server
echo "🚀 Starting LLM Router API server..."
echo "===================================="
echo ""
echo "Server will be available at:"
echo "  - API: http://localhost:8000"
echo "  - Docs: http://localhost:8000/docs"
echo "  - Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the server with uvicorn
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
