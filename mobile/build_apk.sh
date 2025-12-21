#!/bin/bash
# Build Flutter app APK (release version)

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
    APK_ABSPATH=$(cd "$(dirname "$APK_PATH")" && pwd)/$(basename "$APK_PATH")
    
    echo ""
    echo "✅ APK built successfully!"
    echo ""
    echo "📱 APK location: $APK_ABSPATH"
    echo "📊 APK size: $APK_SIZE"
    echo ""
    echo "To install on your phone:"
    echo "  1. Transfer the APK to your phone (via USB, email, or cloud storage)"
    echo "  2. Enable 'Install from Unknown Sources' in Android settings"
    echo "  3. Open the APK file on your phone and install"
    echo ""
    echo "Quick transfer options:"
    echo "  • USB: adb push $APK_PATH /sdcard/Download/"
    echo "  • Or use: adb install $APK_PATH (if phone is connected)"
else
    echo "❌ Error: APK not found at expected location"
    exit 1
fi

