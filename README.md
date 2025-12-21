# Forge Engine 3D Reconstruction Pipeline

High-performance 3D reconstruction pipeline for real-time processing between mobile devices and M4-series Macs. Achieves 1080p @ 30 FPS with sub-100ms end-to-end latency.

## Algorithm Documentation

For detailed information about the core 3D reconstruction algorithm, see [docs/ALGORITHM.md](docs/ALGORITHM.md). This document covers:
- Mathematical foundations (feature detection, color-depth heuristics, composition)
- Algorithm pseudocode and implementation details
- Data structures and performance characteristics
- Parameter configuration and tuning guidelines

## Architecture

- **C++ Core Library**: Accelerate Framework + Grand Central Dispatch for high-speed processing
- **FastAPI Server**: WebSocket binary streaming with device pairing
- **Flutter Mobile Client**: Camera streaming with Dart isolates for smooth UI
- **Zero-Copy Data Flow**: Direct memory sharing between Python and C++

## Features

- 🔐 **Secure Device Pairing**: QR code or 6-digit code pairing before streaming
- ⚡ **High Performance**: Zero-copy data transfer, parallel processing with GCD
- 📱 **Mobile Optimized**: Dart isolates keep UI at 60 FPS while streaming at 30 FPS
- 🎯 **M4 Optimized**: ARMv8.5-a instructions for maximum performance
- 📦 **Model Export**: High-quality PLY/OBJ export for Blender and other 3D software

## Prerequisites

### Mac (Server)
- macOS 13.0+ (Ventura or later)
- M4-series Mac (or compatible Apple Silicon)
- Python 3.10+
- CMake 3.20+
- Xcode Command Line Tools
- pybind11 (installed via pip)

### Mobile Device
- iOS 13.0+ or Android 8.0+
- Flutter 3.0+
- Camera permissions

## Setup Instructions

### Quick Start (with uv)

```bash
# Install all dependencies
./setup.sh

# Build C++ library
cd cpp && ./build.sh

# Run server
./server/run.sh --web
```

### Detailed Setup

### 1. Install Python Dependencies (with uv)

```bash
cd server
uv pip install -r requirements.txt
```

This will install all dependencies including pybind11 needed for the C++ build.

### 2. Build C++ Core Library

**Option A: Using the build script (Recommended)**
```bash
cd cpp
./build.sh
```

The script automatically detects `uv` and uses it for Python commands.

**Option B: Manual build**
```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)
```

The built module will be in `cpp/build/forge_engine.so` (or `.dylib` on macOS).

**Important**: After building, add the build directory to your PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/cpp/build
```

### 4. Setup Flutter Mobile App

```bash
cd mobile
flutter pub get
```

For iOS:
```bash
cd ios
pod install
cd ..
```

### 3. Run the Server

**Option A: Using the run script (Recommended)**
```bash
./server/run.sh --web
```

The script automatically uses `uv run` if available.

**Option B: Using uv directly**
```bash
cd server
uv run python ui.py --web
```

This will:
- Display pairing code in terminal
- Start web UI on http://localhost:8080
- Start FastAPI server on http://0.0.0.0:8000

**Option C: Direct FastAPI**
```bash
cd server
export PYTHONPATH=$PYTHONPATH:$(pwd)/..
uv run python main.py
```

The server will start on `http://0.0.0.0:8000`.

### 6. Configure Mobile App

1. Update the server URL in `mobile/lib/main.dart`:
   ```dart
   final defaultUrl = 'http://YOUR_MAC_IP:8000';
   ```

2. Find your Mac's IP address:
   ```bash
   ifconfig | grep "inet " | grep -v 127.0.0.1
   ```

### 7. Run Mobile App

```bash
cd mobile
flutter run
```

## Usage Workflow

### Device Pairing

1. **Start Server**: Run `python server/ui.py --web` on your Mac
2. **Get Pairing Code**: The terminal will display a 6-digit code and QR code
3. **Open Mobile App**: Launch the Flutter app on your mobile device
4. **Pair Device**: 
   - Scan the QR code, OR
   - Enter the 6-digit code manually
5. **Start Streaming**: Once paired, tap "Start Streaming" in the mobile app

### Model Export

Once frames are processed, export the 3D model:

```bash
# Export as binary PLY (recommended)
curl "http://localhost:8000/export/ply?token=YOUR_SESSION_TOKEN&binary=true" -o model.ply

# Export as OBJ
curl "http://localhost:8000/export/obj?token=YOUR_SESSION_TOKEN" -o model.obj
```

Or use the `/model/status` endpoint to check model progress:

```bash
curl "http://localhost:8000/model/status?token=YOUR_SESSION_TOKEN"
```

## Project Structure

```
project-forge-3d/
├── cpp/                    # C++ Core Library
│   ├── CMakeLists.txt
│   ├── include/
│   │   └── forge_engine/
│   │       ├── FrameProcessor.hpp
│   │       ├── CircularBuffer.hpp
│   │       └── Model3D.hpp
│   └── src/
│       ├── FrameProcessor.cpp
│       ├── Model3D.cpp
│       └── pybind_module.cpp
├── server/                 # FastAPI Server
│   ├── main.py
│   ├── pairing.py
│   ├── ui.py
│   ├── requirements.txt
│   └── templates/
│       └── pairing.html
└── mobile/                 # Flutter Mobile App
    ├── lib/
    │   ├── main.dart
    │   ├── screens/
    │   │   └── pairing_screen.dart
    │   ├── services/
    │   │   ├── pairing_service.dart
    │   │   ├── session_manager.dart
    │   │   └── websocket_service.dart
    │   └── isolates/
    │       └── camera_stream_isolate.dart
    └── pubspec.yaml
```

## Performance Targets

- **Frame Rate**: Steady 30 FPS at 1920x1080
- **Latency**: <100ms end-to-end
- **CPU Usage**: Low (due to C++ optimization and M4 SIMD)
- **Memory**: Zero-copy between network and processing

## Security

- Pairing codes expire after 5 minutes
- Session tokens expire after 24 hours
- WebSocket connections require valid session token
- No broadcast discovery - explicit pairing required
- Only paired devices can connect

## Troubleshooting

### C++ Module Not Found

If you see `Warning: forge_engine module not found`:
1. Ensure you've built the C++ library (see Setup step 2)
2. Add the build directory to `PYTHONPATH`:
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)/cpp/build
   ```
3. Or use the provided run script: `./server/run.sh` (it sets PYTHONPATH automatically)

### pybind11 Not Found During CMake

If CMake can't find pybind11:
1. Install dependencies: `cd server && uv pip install -r requirements.txt`
2. Verify installation with uv: `uv run python -c "import pybind11; print(pybind11.get_cmake_dir())"`
3. Or with regular python: `python3 -c "import pybind11; print(pybind11.get_cmake_dir())"`
4. If the command above works, CMake should find it automatically (it tries uv first, then regular python)

### Camera Not Working

- **iOS**: Add camera permissions to `ios/Runner/Info.plist`
- **Android**: Add camera permissions to `android/app/src/main/AndroidManifest.xml`

### WebSocket Connection Failed

- Ensure both devices are on the same WiFi network
- Check firewall settings on Mac
- Verify server URL in mobile app matches Mac's IP address

## Development

### Adding 3D Reconstruction Algorithm

The placeholder reconstruction algorithm is in `cpp/src/FrameProcessor.cpp` in the `performReconstruction()` method. Replace this with your chosen algorithm:

- Structure from Motion (SfM)
- Stereo Vision
- Monocular Depth Estimation
- SLAM

### Building for Production

```bash
# C++ with optimizations
cd cpp/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=armv8.5-a"
make -j$(sysctl -n hw.ncpu)

# Flutter release build
cd mobile
flutter build apk --release  # Android
flutter build ios --release  # iOS
```

## License

[Your License Here]

## Contributing

[Contributing Guidelines Here]
