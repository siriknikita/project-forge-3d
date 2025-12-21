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

  /// Convert CameraImage to Uint8List (called from main isolate).
  static Uint8List convertImageToBytes(CameraImage image) {
    if (image.format.group == ImageFormatGroup.yuv420) {
      // YUV420 format (Android)
      // For simplicity, we'll send Y plane only
      // In production, you'd want to convert YUV to RGB
      return Uint8List.fromList(image.planes[0].bytes);
    } else {
      // BGRA8888 format (iOS)
      return Uint8List.fromList(image.planes[0].bytes);
    }
  }
}

