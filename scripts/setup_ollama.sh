#!/bin/bash
# Setup script for initializing Ollama models

set -e  # Exit on error

echo "🚀 LLM Router - Ollama Setup Script"
echo "===================================="
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Error: Ollama is not installed."
    echo ""
    echo "Please install Ollama first:"
    echo "  macOS/Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or visit: https://ollama.com/download"
    exit 1
fi

echo "✅ Ollama is installed"
echo ""

# Check if Ollama server is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama server is not running."
    echo "Starting Ollama server in the background..."
    ollama serve > /dev/null 2>&1 &
    sleep 3  # Give the server time to start

    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "❌ Error: Failed to start Ollama server."
        echo "Please start it manually with: ollama serve"
        exit 1
    fi
fi

echo "✅ Ollama server is running"
echo ""

# Pull the required models
echo "📦 Downloading required models..."
echo ""

echo "1/2 Pulling llama3.2:3b (Professor Expert)..."
if ollama pull llama3.2:3b; then
    echo "   ✅ llama3.2:3b downloaded successfully"
else
    echo "   ❌ Failed to download llama3.2:3b"
    exit 1
fi
echo ""

echo "2/2 Pulling tinyllama (Zoomer Expert)..."
if ollama pull tinyllama; then
    echo "   ✅ tinyllama downloaded successfully"
else
    echo "   ❌ Failed to download tinyllama"
    exit 1
fi
echo ""

# Verify models are available
echo "🔍 Verifying models..."
MODELS=$(ollama list | awk 'NR>1 {print $1}')

if echo "$MODELS" | grep -q "llama3.2:3b" && echo "$MODELS" | grep -q "tinyllama"; then
    echo "✅ All models verified and ready"
else
    echo "⚠️  Warning: Some models may not be available"
    echo "Available models:"
    ollama list
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Install Python dependencies: pip install -e ."
echo "  2. Run the server: ./scripts/run_server.sh"
echo "  3. Test the API: curl http://localhost:8000/health"
echo ""
