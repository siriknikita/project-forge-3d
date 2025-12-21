#!/bin/bash
# Run script for Forge Engine server

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Add project root to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Add C++ build directory to Python path if it exists
if [ -d "$PROJECT_ROOT/cpp/build" ]; then
    export PYTHONPATH="$PROJECT_ROOT/cpp/build:$PYTHONPATH"
fi

# Change to server directory
cd "$SCRIPT_DIR"

# Run the UI script with uv if available, otherwise use python3
if command -v uv &> /dev/null; then
    uv run python ui.py "$@"
else
    python3 ui.py "$@"
fi

