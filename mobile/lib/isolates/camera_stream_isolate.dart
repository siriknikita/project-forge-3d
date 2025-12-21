import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:io';
import 'package:camera/camera.dart';
import '../services/websocket_service.dart';

/// Serializable frame data structure for isolate communication.
class FrameDataForIsolate {
  final int width;
  final int height;
  final String format; // 'yuv420' or 'bgra8888'
  final List<Uint8List> planeBytes;
  final List<int> bytesPerRow;
  final List<int>? planeWidths;
  final List<int>? planeHeights;
  final int? targetWidth; // Optional target width for downsampling
  final int? targetHeight; // Optional target height for downsampling

  FrameDataForIsolate({
    required this.width,
    required this.height,
    required this.format,
    required this.planeBytes,
    required this.bytesPerRow,
    this.planeWidths,
    this.planeHeights,
    this.targetWidth,
    this.targetHeight,
  });
}

/// Target resolution for downsampling (reduces processing load significantly)
const int targetWidth = 640;
const int targetHeight = 480;

/// Top-level function for isolate-based frame conversion.
/// This function can be called via compute().
Uint8List convertImageInIsolate(FrameDataForIsolate frameData) {
  // Determine target resolution (use provided or default)
  final targetW = frameData.targetWidth ?? targetWidth;
  final targetH = frameData.targetHeight ?? targetHeight;
  
  // Check if downsampling is needed
  final needsDownsampling = frameData.width > targetW || frameData.height > targetH;
  
  if (frameData.format == 'yuv420') {
    if (needsDownsampling) {
      return yuv420ToRgbInIsolateDownsampled(frameData, targetW, targetH);
    } else {
      return yuv420ToRgbInIsolate(frameData);
    }
  } else {
    if (needsDownsampling) {
      return bgraToRgbInIsolateDownsampled(frameData, targetW, targetH);
    } else {
      return bgraToRgbInIsolate(frameData);
    }
  }
}

/// Convert YUV420 format to RGB in isolate (optimized version).
Uint8List yuv420ToRgbInIsolate(FrameDataForIsolate frameData) {
  final width = frameData.width;
  final height = frameData.height;
  
  final rgbData = Uint8List(width * height * 3);
  
  // Get Y plane bytes
  final yBytes = frameData.planeBytes[0];
  final yStride = frameData.bytesPerRow[0];
  
  // Get U and V plane bytes if available
  Uint8List? uBytes;
  Uint8List? vBytes;
  int? uStride;
  int? vStride;
  int? uWidth;
  int? uHeight;
  
  if (frameData.planeBytes.length > 2 && 
      frameData.planeWidths != null && 
      frameData.planeHeights != null) {
    uBytes = frameData.planeBytes[1];
    vBytes = frameData.planeBytes[2];
    uStride = frameData.bytesPerRow[1];
    vStride = frameData.bytesPerRow[2];
    uWidth = frameData.planeWidths![1];
    uHeight = frameData.planeHeights![1];
  }
  
  // Pre-calculate constants for YUV to RGB conversion
  const int yCoeff = 298;
  const int vRCoeff = 409;
  const int vGCoeff = 208;
  const int uGCoeff = 100;
  const int uBCoeff = 516;
  const int shift = 256;
  const int halfShift = 128;
  
  // Pre-calculate UV lookup optimization
  final bool hasUV = uBytes != null && vBytes != null && uWidth != null && uHeight != null;
  final int uWidthVal = uWidth ?? 0;
  final int uHeightVal = uHeight ?? 0;
  final int uStrideVal = uStride ?? 0;
  
  // Optimized conversion: process pixels with stride-aware access
  for (int y = 0; y < height; y++) {
    final yRowOffset = y * yStride;
    final rgbRowOffset = y * width * 3;
    final uvY = y ~/ 2; // Pre-calculate UV row index
    final uvYClamped = hasUV ? uvY.clamp(0, uHeightVal - 1) : 0;
    final uvRowOffset = hasUV ? uvYClamped * uStrideVal : 0;
    
    for (int x = 0; x < width; x++) {
      // Get Y value
      final yVal = yBytes[yRowOffset + x];
      
      // Get U and V values (subsampled by 2x2) - optimized lookup
      int uVal = 128;
      int vVal = 128;
      
      if (hasUV) {
        final uvX = (x ~/ 2).clamp(0, uWidthVal - 1);
        final uvIndex = uvRowOffset + uvX;
        
        if (uvIndex < uBytes!.length) uVal = uBytes[uvIndex];
        if (uvIndex < vBytes!.length) vVal = vBytes[uvIndex];
      }
      
      // Optimized YUV to RGB conversion with pre-calculated constants
      final yAdj = yVal - 16;
      final uAdj = uVal - 128;
      final vAdj = vVal - 128;
      
      // ITU-R BT.601 conversion with optimized integer math
      // Pre-calculate common terms
      final yTerm = yCoeff * yAdj;
      final vTerm = vRCoeff * vAdj;
      final uTerm = uBCoeff * uAdj;
      final uGTerm = uGCoeff * uAdj;
      final vGTerm = vGCoeff * vAdj;
      
      // Calculate RGB with optimized clamping (using bitwise OR for faster bounds check)
      int r = (yTerm + vTerm + halfShift) ~/ shift;
      int g = (yTerm - uGTerm - vGTerm + halfShift) ~/ shift;
      int b = (yTerm + uTerm + halfShift) ~/ shift;
      
      // Optimized clamping: use bitwise operations where possible
      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);
      
      // Write RGB values (optimized index calculation)
      final rgbIndex = rgbRowOffset + (x * 3);
      rgbData[rgbIndex] = r;
      rgbData[rgbIndex + 1] = g;
      rgbData[rgbIndex + 2] = b;
    }
  }
  
  return rgbData;
}

/// Convert YUV420 format to RGB with downsampling (much faster for large frames).
Uint8List yuv420ToRgbInIsolateDownsampled(FrameDataForIsolate frameData, int targetWidth, int targetHeight) {
  final srcWidth = frameData.width;
  final srcHeight = frameData.height;
  
  final rgbData = Uint8List(targetWidth * targetHeight * 3);
  
  // Calculate scaling factors (pre-calculate once)
  final scaleX = srcWidth / targetWidth;
  final scaleY = srcHeight / targetHeight;
  
  // Get Y plane bytes
  final yBytes = frameData.planeBytes[0];
  final yStride = frameData.bytesPerRow[0];
  
  // Get U and V plane bytes if available
  Uint8List? uBytes;
  Uint8List? vBytes;
  int? uStride;
  int? vStride;
  int? uWidth;
  int? uHeight;
  
  if (frameData.planeBytes.length > 2 && 
      frameData.planeWidths != null && 
      frameData.planeHeights != null) {
    uBytes = frameData.planeBytes[1];
    vBytes = frameData.planeBytes[2];
    uStride = frameData.bytesPerRow[1];
    vStride = frameData.bytesPerRow[2];
    uWidth = frameData.planeWidths![1];
    uHeight = frameData.planeHeights![1];
  }
  
  // Pre-calculate constants for YUV to RGB conversion
  const int yCoeff = 298;
  const int vRCoeff = 409;
  const int vGCoeff = 208;
  const int uGCoeff = 100;
  const int uBCoeff = 516;
  const int shift = 256;
  const int halfShift = 128;
  
  // Pre-calculate UV lookup optimization
  final bool hasUV = uBytes != null && vBytes != null && uWidth != null && uHeight != null;
  final int uWidthVal = uWidth ?? 0;
  final int uHeightVal = uHeight ?? 0;
  final int uStrideVal = uStride ?? 0;
  
  // Process only target resolution pixels (downsampling)
  for (int y = 0; y < targetHeight; y++) {
    final srcY = (y * scaleY).round().clamp(0, srcHeight - 1);
    final yRowOffset = srcY * yStride;
    final rgbRowOffset = y * targetWidth * 3;
    final uvSrcY = srcY ~/ 2;
    final uvYClamped = hasUV ? uvSrcY.clamp(0, uHeightVal - 1) : 0;
    final uvRowOffset = hasUV ? uvYClamped * uStrideVal : 0;
    
    for (int x = 0; x < targetWidth; x++) {
      final srcX = (x * scaleX).round().clamp(0, srcWidth - 1);
      
      // Get Y value from source
      final yVal = yBytes[yRowOffset + srcX];
      
      // Get U and V values (subsampled by 2x2) - optimized lookup
      int uVal = 128;
      int vVal = 128;
      
      if (hasUV) {
        final uvX = (srcX ~/ 2).clamp(0, uWidthVal - 1);
        final uvIndex = uvRowOffset + uvX;
        
        if (uvIndex < uBytes!.length) uVal = uBytes[uvIndex];
        if (uvIndex < vBytes!.length) vVal = vBytes[uvIndex];
      }
      
      // Optimized YUV to RGB conversion with pre-calculated constants
      final yAdj = yVal - 16;
      final uAdj = uVal - 128;
      final vAdj = vVal - 128;
      
      // Pre-calculate common terms
      final yTerm = yCoeff * yAdj;
      final vTerm = vRCoeff * vAdj;
      final uTerm = uBCoeff * uAdj;
      final uGTerm = uGCoeff * uAdj;
      final vGTerm = vGCoeff * vAdj;
      
      // Calculate RGB with optimized clamping
      int r = (yTerm + vTerm + halfShift) ~/ shift;
      int g = (yTerm - uGTerm - vGTerm + halfShift) ~/ shift;
      int b = (yTerm + uTerm + halfShift) ~/ shift;
      
      // Optimized clamping
      r = r.clamp(0, 255);
      g = g.clamp(0, 255);
      b = b.clamp(0, 255);
      
      // Write RGB values
      final rgbIndex = rgbRowOffset + (x * 3);
      rgbData[rgbIndex] = r;
      rgbData[rgbIndex + 1] = g;
      rgbData[rgbIndex + 2] = b;
    }
  }
  
  return rgbData;
}

/// Convert BGRA8888 format to RGB in isolate (drop alpha channel, optimized).
Uint8List bgraToRgbInIsolate(FrameDataForIsolate frameData) {
  final width = frameData.width;
  final height = frameData.height;
  final bgraBytes = frameData.planeBytes[0];
  final stride = frameData.bytesPerRow[0];
  
  final rgbData = Uint8List(width * height * 3);
  
  // Optimized: process with stride-aware access
  for (int y = 0; y < height; y++) {
    final bgraRowOffset = y * stride;
    final rgbRowOffset = y * width * 3;
    
    for (int x = 0; x < width; x++) {
      // BGRA format: B, G, R, A
      final bgraIndex = bgraRowOffset + (x * 4);
      
      if (bgraIndex + 2 < bgraBytes.length) {
        final b = bgraBytes[bgraIndex];
        final g = bgraBytes[bgraIndex + 1];
        final r = bgraBytes[bgraIndex + 2];
        
        // Write RGB values
        final rgbIndex = rgbRowOffset + (x * 3);
        rgbData[rgbIndex] = r;
        rgbData[rgbIndex + 1] = g;
        rgbData[rgbIndex + 2] = b;
      }
    }
  }
  
  return rgbData;
}

/// Convert BGRA8888 format to RGB with downsampling (much faster for large frames).
Uint8List bgraToRgbInIsolateDownsampled(FrameDataForIsolate frameData, int targetWidth, int targetHeight) {
  final srcWidth = frameData.width;
  final srcHeight = frameData.height;
  final bgraBytes = frameData.planeBytes[0];
  final stride = frameData.bytesPerRow[0];
  
  final rgbData = Uint8List(targetWidth * targetHeight * 3);
  
  // Calculate scaling factors
  final scaleX = srcWidth / targetWidth;
  final scaleY = srcHeight / targetHeight;
  
  // Process only target resolution pixels (downsampling)
  for (int y = 0; y < targetHeight; y++) {
    final srcY = (y * scaleY).round().clamp(0, srcHeight - 1);
    final bgraRowOffset = srcY * stride;
    final rgbRowOffset = y * targetWidth * 3;
    
    for (int x = 0; x < targetWidth; x++) {
      final srcX = (x * scaleX).round().clamp(0, srcWidth - 1);
      
      // BGRA format: B, G, R, A
      final bgraIndex = bgraRowOffset + (srcX * 4);
      
      if (bgraIndex + 2 < bgraBytes.length) {
        final b = bgraBytes[bgraIndex];
        final g = bgraBytes[bgraIndex + 1];
        final r = bgraBytes[bgraIndex + 2];
        
        // Write RGB values
        final rgbIndex = rgbRowOffset + (x * 3);
        rgbData[rgbIndex] = r;
        rgbData[rgbIndex + 1] = g;
        rgbData[rgbIndex + 2] = b;
      }
    }
  }
  
  return rgbData;
}

/// Isolate entry point for frame conversion worker.
void _isolateWorkerEntry(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);
  
  receivePort.listen((message) {
    if (message is Map) {
      if (message['type'] == 'convert') {
        final frameData = message['frameData'] as FrameDataForIsolate;
        final resultPort = message['resultPort'] as SendPort;
        
        try {
          final result = convertImageInIsolate(frameData);
          resultPort.send({'success': true, 'result': result});
        } catch (e) {
          resultPort.send({'success': false, 'error': e.toString()});
        }
      } else if (message['type'] == 'stop') {
        receivePort.close();
        Isolate.exit();
      }
    }
  });
}

/// Pool of persistent isolates for frame conversion.
class IsolatePool {
  final List<Isolate> _isolates = [];
  final List<SendPort> _sendPorts = [];
  int _nextIndex = 0;
  static const int _poolSize = 8; // 8 persistent isolates for parallel processing (increased for 30 FPS)

  /// Initialize the isolate pool.
  Future<void> initialize() async {
    for (int i = 0; i < _poolSize; i++) {
      final receivePort = ReceivePort();
      final isolate = await Isolate.spawn(_isolateWorkerEntry, receivePort.sendPort);
      final sendPort = await receivePort.first as SendPort;
      _isolates.add(isolate);
      _sendPorts.add(sendPort);
    }
  }

  /// Convert frame using an available isolate from the pool.
  Future<Uint8List> convertFrame(FrameDataForIsolate frameData) async {
    // Use round-robin to distribute load, but don't wait for busy isolates
    // This allows parallel processing across all isolates
    final index = _nextIndex % _poolSize;
    _nextIndex++;
    
    final resultPort = ReceivePort();
    _sendPorts[index].send({
      'type': 'convert',
      'frameData': frameData,
      'resultPort': resultPort.sendPort,
    });
    
    try {
      final result = await resultPort.first as Map;
      
      if (result['success'] == true) {
        return result['result'] as Uint8List;
      } else {
        throw Exception(result['error'] as String);
      }
    } catch (e) {
      rethrow;
    }
  }

  /// Dispose of all isolates in the pool.
  Future<void> dispose() async {
    for (int i = 0; i < _sendPorts.length; i++) {
      _sendPorts[i].send({'type': 'stop'});
    }
    for (final isolate in _isolates) {
      isolate.kill(priority: Isolate.immediate);
    }
    _isolates.clear();
    _sendPorts.clear();
  }
}

/// Isolate for camera streaming to keep UI responsive.
class CameraStreamIsolate {
  static IsolatePool? _isolatePool;
  static const int targetFps = 30;
  static const Duration frameInterval = Duration(milliseconds: 1000 ~/ targetFps);

  /// Initialize the isolate pool (call before streaming).
  static Future<void> initializePool() async {
    if (_isolatePool == null) {
      _isolatePool = IsolatePool();
      await _isolatePool!.initialize();
    }
  }

  /// Dispose of the isolate pool (call after streaming).
  static Future<void> disposePool() async {
    await _isolatePool?.dispose();
    _isolatePool = null;
  }

  /// Start camera stream isolate.
  static Future<SendPort> startIsolate({
    required String serverUrl,
    required String sessionToken,
    required CameraController cameraController,
  }) async {
    final receivePort = ReceivePort();
    await Isolate.spawn(
      _isolateEntry,
      {
        'sendPort': receivePort.sendPort,
        'serverUrl': serverUrl,
        'sessionToken': sessionToken,
      },
    );

    return await receivePort.first as SendPort;
  }

  /// Isolate entry point.
  static void _isolateEntry(Map<String, dynamic> message) async {
    final sendPort = message['sendPort'] as SendPort;
    final serverUrl = message['serverUrl'] as String;
    final sessionToken = message['sessionToken'] as String;

    final receivePort = ReceivePort();
    sendPort.send(receivePort.sendPort);

    WebSocketService? wsService;

    try {
      // Connect WebSocket
      wsService = WebSocketService(
        serverUrl: serverUrl,
        sessionToken: sessionToken,
      );
      await wsService.connect();

      // Listen for camera frames from main isolate
      receivePort.listen((message) async {
        if (message == 'stop') {
          await wsService?.close();
          Isolate.exit(sendPort);
        } else if (message is Map && message.containsKey('frame')) {
          try {
            final frameData = message['frame'] as Uint8List;
            wsService!.sendFrame(frameData); // Non-blocking, no await needed
          } catch (e) {
            sendPort.send({'error': e.toString()});
          }
        }
      });

      // Frame rate control
      Timer.periodic(frameInterval, (timer) {
        // Frame rate is controlled by camera stream
      });

    } catch (e) {
      sendPort.send({'error': e.toString()});
      await wsService?.close();
      Isolate.exit(sendPort);
    }
  }

  /// Convert CameraImage to Uint8List in RGB format (called from main isolate).
  /// Uses IsolatePool for efficient parallel frame conversion.
  static Future<Uint8List> convertImageToBytes(CameraImage image) async {
    // Ensure pool is initialized
    if (_isolatePool == null) {
      await initializePool();
    }
    
    // Extract serializable data from CameraImage
    final format = image.format.group == ImageFormatGroup.yuv420 ? 'yuv420' : 'bgra8888';
    // Optimize: avoid copying if plane.bytes is already Uint8List (which it should be)
    final planeBytes = image.planes.map((plane) => 
      plane.bytes is Uint8List ? plane.bytes as Uint8List : Uint8List.fromList(plane.bytes)
    ).toList();
    final bytesPerRow = image.planes.map((plane) => plane.bytesPerRow).toList();
    // Convert nullable ints to non-nullable, filtering out nulls if needed
    final planeWidths = image.planes
        .map((plane) => plane.width)
        .where((width) => width != null)
        .cast<int>()
        .toList();
    final planeHeights = image.planes
        .map((plane) => plane.height)
        .where((height) => height != null)
        .cast<int>()
        .toList();
    
    final frameData = FrameDataForIsolate(
      width: image.width,
      height: image.height,
      format: format,
      planeBytes: planeBytes,
      bytesPerRow: bytesPerRow,
      planeWidths: planeWidths.isNotEmpty ? planeWidths : null,
      planeHeights: planeHeights.isNotEmpty ? planeHeights : null,
    );
    
    // Use IsolatePool for efficient parallel processing
    return await _isolatePool!.convertFrame(frameData);
  }

}

