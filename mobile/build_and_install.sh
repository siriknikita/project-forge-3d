#!/bin/bash
# Build Flutter app APK (without installing)

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🔨 Building Flutter app APK..."

# Build release APK
echo "📦 Building release APK..."
flutter build apk --release

APK_PATH="build/app/outputs/flutter-apk/app-release.apk"

if [ -f "$APK_PATH" ]; then
    APK_SIZE=$(du -h "$APK_PATH" | cut -f1)
    echo ""
    echo "✅ APK built successfully!"
    echo ""
    echo "📱 APK location: $APK_PATH"
    echo "📊 APK size: $APK_SIZE"
    echo ""
    echo "To install on your phone:"
    echo "  1. Transfer the APK to your phone (via USB, email, or cloud storage)"
    echo "  2. Enable 'Install from Unknown Sources' in Android settings"
    echo "  3. Open the APK file on your phone and install"
    echo ""
    echo "Or use ADB to install (if phone is connected):"
    echo "  adb install $APK_PATH"
else
    echo "❌ Error: APK not found at expected location"
    exit 1
fi

