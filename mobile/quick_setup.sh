#!/bin/bash
# Quick setup: install deps, check setup, and optionally build

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Flutter Mobile Quick Setup"
echo "=============================="
echo ""

# Check Flutter
if ! command -v flutter &> /dev/null; then
    echo "❌ Flutter not found. Install it first:"
    echo "   brew install flutter"
    exit 1
fi

# Initialize project if needed
if [ ! -d "android" ] || [ ! -d "ios" ]; then
    echo "📦 Initializing Flutter project..."
    flutter create .
fi

# Install dependencies
echo "📥 Installing dependencies..."
flutter pub get

# Check devices
echo ""
echo "🔍 Checking for devices..."
flutter devices

echo ""
echo "✅ Setup complete!"
echo ""
echo "Available commands:"
echo "  • ./build_apk.sh           - Build release APK (recommended)"
echo "  • flutter run              - Run on connected device (for development)"
echo "  • flutter build apk        - Build APK file"

