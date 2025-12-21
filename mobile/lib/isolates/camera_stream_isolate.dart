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

  FrameDataForIsolate({
    required this.width,
    required this.height,
    required this.format,
    required this.planeBytes,
    required this.bytesPerRow,
    this.planeWidths,
    this.planeHeights,
  });
}

/// Top-level function for isolate-based frame conversion.
/// This function can be called via compute().
Uint8List convertImageInIsolate(FrameDataForIsolate frameData) {
  if (frameData.format == 'yuv420') {
    return yuv420ToRgbInIsolate(frameData);
  } else {
    return bgraToRgbInIsolate(frameData);
  }
}

/// Convert YUV420 format to RGB in isolate.
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
  
  // Convert YUV to RGB
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // Get Y value
      final yIndex = (y * yStride) + x;
      final yVal = yBytes[yIndex];
      
      // Get U and V values (subsampled by 2x2)
      int uVal = 128;
      int vVal = 128;
      
      if (uBytes != null && vBytes != null && uWidth != null && uHeight != null) {
        final uvX = (x ~/ 2).clamp(0, uWidth - 1);
        final uvY = (y ~/ 2).clamp(0, uHeight - 1);
        final uIndex = (uvY * uStride!) + uvX;
        final vIndex = (uvY * vStride!) + uvX;
        
        if (uIndex < uBytes.length) uVal = uBytes[uIndex];
        if (vIndex < vBytes.length) vVal = vBytes[vIndex];
      }
      
      // Convert YUV to RGB using ITU-R BT.601 conversion
      final yFloat = (yVal - 16) / 219.0;
      final uFloat = (uVal - 128) / 224.0;
      final vFloat = (vVal - 128) / 224.0;
      
      int r = ((yFloat + 1.402 * vFloat) * 255).round().clamp(0, 255);
      int g = ((yFloat - 0.344 * uFloat - 0.714 * vFloat) * 255).round().clamp(0, 255);
      int b = ((yFloat + 1.772 * uFloat) * 255).round().clamp(0, 255);
      
      // Write RGB values
      final rgbIndex = (y * width + x) * 3;
      rgbData[rgbIndex] = r;
      rgbData[rgbIndex + 1] = g;
      rgbData[rgbIndex + 2] = b;
    }
  }
  
  return rgbData;
}

/// Convert BGRA8888 format to RGB in isolate (drop alpha channel).
Uint8List bgraToRgbInIsolate(FrameDataForIsolate frameData) {
  final width = frameData.width;
  final height = frameData.height;
  final bgraBytes = frameData.planeBytes[0];
  final stride = frameData.bytesPerRow[0];
  
  final rgbData = Uint8List(width * height * 3);
  
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // BGRA format: B, G, R, A
      final bgraIndex = (y * stride) + (x * 4);
      
      if (bgraIndex + 2 < bgraBytes.length) {
        final b = bgraBytes[bgraIndex];
        final g = bgraBytes[bgraIndex + 1];
        final r = bgraBytes[bgraIndex + 2];
        // Skip alpha (bgraIndex + 3)
        
        // Write RGB values
        final rgbIndex = (y * width + x) * 3;
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

