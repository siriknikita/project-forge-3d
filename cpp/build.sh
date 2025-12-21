#!/bin/bash
# Build script for Forge Engine C++ library

set -e

echo "Building Forge Engine C++ library..."

# Check if pybind11 is installed (try uv first, then regular python)
# Need to check from server directory where dependencies are installed
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
SERVER_DIR="$PROJECT_ROOT/server"

if command -v uv &> /dev/null; then
    # Check from server directory where uv dependencies are installed
    if ! (cd "$SERVER_DIR" && uv run python -c "import pybind11" 2>/dev/null); then
        echo "Error: pybind11 is not installed in uv environment."
        echo "Please install it first:"
        echo "  cd server && uv pip install -r requirements.txt"
        exit 1
    fi
    PYTHON_CMD="uv run python"
    # Get the actual Python executable path from uv
    UV_PYTHON_EXECUTABLE=$(cd "$SERVER_DIR" && uv run python -c "import sys; print(sys.executable)" 2>/dev/null)
    if [ -n "$UV_PYTHON_EXECUTABLE" ]; then
        echo "Detected Python from uv: $UV_PYTHON_EXECUTABLE"
        CMAKE_PYTHON_ARG="-DPython3_EXECUTABLE=$UV_PYTHON_EXECUTABLE"
    fi
else
    PYTHON_CMD="python3"
    if ! python3 -c "import pybind11" 2>/dev/null; then
        echo "Error: pybind11 is not installed."
        echo "Please install it first:"
        echo "  pip install pybind11"
        exit 1
    fi
    CMAKE_PYTHON_ARG=""
fi

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
# CMakeLists.txt will automatically detect uv and use it from server directory
# Also pass Python executable explicitly to ensure correct version
cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_PYTHON_ARG

# Build
echo "Building..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "Build complete! The module is in: $(pwd)/forge_engine.so"
echo ""
echo "To use it, add this directory to PYTHONPATH:"
echo "  export PYTHONPATH=\$PYTHONPATH:$(pwd)"

