import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';
import 'dart:io';
import 'package:camera/camera.dart';
import '../services/websocket_service.dart';

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
            await wsService!.sendFrame(frameData);
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
  static Uint8List convertImageToBytes(CameraImage image) {
    if (image.format.group == ImageFormatGroup.yuv420) {
      // YUV420 format (Android)
      // Convert YUV420 to RGB
      return _yuv420ToRgb(image);
    } else {
      // BGRA8888 format (iOS)
      // Extract RGB channels (drop alpha)
      return _bgraToRgb(image);
    }
  }

  /// Convert YUV420 format to RGB.
  static Uint8List _yuv420ToRgb(CameraImage image) {
    final width = image.width;
    final height = image.height;
    
    // YUV420 has 3 planes: Y (luma), U (chroma), V (chroma)
    final yPlane = image.planes[0];
    final uPlane = image.planes.length > 1 ? image.planes[1] : null;
    final vPlane = image.planes.length > 2 ? image.planes[2] : null;
    
    final rgbData = Uint8List(width * height * 3);
    
    // Get Y plane bytes
    final yBytes = yPlane.bytes;
    final yStride = yPlane.bytesPerRow;
    
    // Get U and V plane bytes if available
    Uint8List? uBytes;
    Uint8List? vBytes;
    int? uStride;
    int? vStride;
    int? uWidth;
    int? uHeight;
    
    if (uPlane != null && vPlane != null) {
      uBytes = uPlane.bytes;
      vBytes = vPlane.bytes;
      uStride = uPlane.bytesPerRow;
      vStride = vPlane.bytesPerRow;
      uWidth = uPlane.width;
      uHeight = uPlane.height;
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

  /// Convert BGRA8888 format to RGB (drop alpha channel).
  static Uint8List _bgraToRgb(CameraImage image) {
    final width = image.width;
    final height = image.height;
    final plane = image.planes[0];
    final bgraBytes = plane.bytes;
    final stride = plane.bytesPerRow;
    
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
}

