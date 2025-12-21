#!/bin/bash
# Setup script for Forge Engine using uv

set -e

echo "Setting up Forge Engine with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed."
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
cd server
uv pip install -r requirements.txt
cd ..

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Build C++ library: cd cpp && ./build.sh"
echo "2. Run server: ./server/run.sh --web"

