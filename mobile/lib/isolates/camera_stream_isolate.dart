import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
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
  
  // Optimized conversion: process pixels with stride-aware access
  for (int y = 0; y < height; y++) {
    final yRowOffset = y * yStride;
    final rgbRowOffset = y * width * 3;
    
    for (int x = 0; x < width; x++) {
      // Get Y value
      final yVal = yBytes[yRowOffset + x];
      
      // Get U and V values (subsampled by 2x2)
      int uVal = 128;
      int vVal = 128;
      
      if (uBytes != null && vBytes != null && uWidth != null && uHeight != null) {
        final uvX = (x ~/ 2).clamp(0, uWidth - 1);
        final uvY = (y ~/ 2).clamp(0, uHeight - 1);
        final uvIndex = (uvY * uStride!) + uvX;
        
        if (uvIndex < uBytes.length) uVal = uBytes[uvIndex];
        if (uvIndex < vBytes.length) vVal = vBytes[uvIndex];
      }
      
      // Optimized YUV to RGB conversion using integer math where possible
      final yAdj = yVal - 16;
      final uAdj = uVal - 128;
      final vAdj = vVal - 128;
      
      // ITU-R BT.601 conversion with integer optimization
      int r = (298 * yAdj + 409 * vAdj + 128) ~/ 256;
      int g = (298 * yAdj - 100 * uAdj - 208 * vAdj + 128) ~/ 256;
      int b = (298 * yAdj + 516 * uAdj + 128) ~/ 256;
      
      // Clamp values
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

/// Convert YUV420 format to RGB with downsampling (much faster for large frames).
Uint8List yuv420ToRgbInIsolateDownsampled(FrameDataForIsolate frameData, int targetWidth, int targetHeight) {
  final srcWidth = frameData.width;
  final srcHeight = frameData.height;
  
  final rgbData = Uint8List(targetWidth * targetHeight * 3);
  
  // Calculate scaling factors
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
  
  // Process only target resolution pixels (downsampling)
  for (int y = 0; y < targetHeight; y++) {
    final srcY = (y * scaleY).round().clamp(0, srcHeight - 1);
    final yRowOffset = srcY * yStride;
    final rgbRowOffset = y * targetWidth * 3;
    
    for (int x = 0; x < targetWidth; x++) {
      final srcX = (x * scaleX).round().clamp(0, srcWidth - 1);
      
      // Get Y value from source
      final yVal = yBytes[yRowOffset + srcX];
      
      // Get U and V values (subsampled by 2x2)
      int uVal = 128;
      int vVal = 128;
      
      if (uBytes != null && vBytes != null && uWidth != null && uHeight != null) {
        final uvX = (srcX ~/ 2).clamp(0, uWidth - 1);
        final uvY = (srcY ~/ 2).clamp(0, uHeight - 1);
        final uvIndex = (uvY * uStride!) + uvX;
        
        if (uvIndex < uBytes.length) uVal = uBytes[uvIndex];
        if (uvIndex < vBytes.length) vVal = vBytes[uvIndex];
      }
      
      // Optimized YUV to RGB conversion
      final yAdj = yVal - 16;
      final uAdj = uVal - 128;
      final vAdj = vVal - 128;
      
      int r = (298 * yAdj + 409 * vAdj + 128) ~/ 256;
      int g = (298 * yAdj - 100 * uAdj - 208 * vAdj + 128) ~/ 256;
      int b = (298 * yAdj + 516 * uAdj + 128) ~/ 256;
      
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

/// Isolate for camera streaming to keep UI responsive.
class CameraStreamIsolate {
  static const int targetFps = 30;
  static const Duration frameInterval = Duration(milliseconds: 1000 ~/ targetFps);

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
      // Note: Camera initialization in isolate is complex
      // For now, we'll receive camera frames from main isolate
      // This is a simplified version - in production, you'd pass camera controller
      // or use a different approach for isolate-based streaming

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
  /// Uses compute() to offload conversion to an isolate for non-blocking processing.
  static Future<Uint8List> convertImageToBytes(CameraImage image) async {
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
    
    // Use compute() to run conversion in isolate
    return await compute(convertImageInIsolate, frameData);
  }

}

