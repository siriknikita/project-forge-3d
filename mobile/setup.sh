#!/bin/bash
# Flutter mobile app setup and build script

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🚀 Setting up Flutter mobile app..."

# Check if Flutter is installed
if ! command -v flutter &> /dev/null; then
    echo "❌ Error: Flutter is not installed."
    echo "Please install Flutter first:"
    echo "  brew install flutter"
    echo "  or visit: https://flutter.dev/docs/get-started/install"
    exit 1
fi

# Check Flutter installation
echo "📱 Checking Flutter installation..."
flutter doctor

# Initialize Flutter project if needed
if [ ! -d "android" ] || [ ! -d "ios" ]; then
    echo "📦 Initializing Flutter project structure..."
    flutter create .
fi

# Install dependencies
echo "📥 Installing Flutter dependencies..."
flutter pub get

# Check for connected devices
echo "🔍 Checking for connected devices..."
DEVICES=$(flutter devices | grep -c "•" || echo "0")
if [ "$DEVICES" -eq "0" ]; then
    echo "⚠️  No devices found. Connect your Android phone via USB or start an emulator."
    echo ""
    echo "To connect Android phone:"
    echo "  1. Enable Developer Options (Settings → About Phone → tap Build Number 7 times)"
    echo "  2. Enable USB Debugging (Settings → Developer Options)"
    echo "  3. Connect via USB"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  • Build APK: ./build_apk.sh"
echo "  • Run on connected device (dev): flutter run"
echo "  • Build APK manually: flutter build apk --release"

