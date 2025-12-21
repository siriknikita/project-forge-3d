import 'dart:async';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';

/// WebSocket service for binary frame streaming.
class WebSocketService {
  WebSocketChannel? _channel;
  final String serverUrl;
  final String sessionToken;
  StreamController<Uint8List>? _frameController;

  WebSocketService({
    required this.serverUrl,
    required this.sessionToken,
  });

  /// Connect to WebSocket server.
  Future<void> connect() async {
    try {
      final wsUrl = serverUrl.replaceFirst('http://', 'ws://').replaceFirst('https://', 'wss://');
      final uri = Uri.parse('$wsUrl/ws/stream?token=$sessionToken');
      
      _channel = WebSocketChannel.connect(uri);
      _frameController = StreamController<Uint8List>.broadcast();
    } catch (e) {
      throw Exception('WebSocket connection error: $e');
    }
  }

  /// Send binary frame data.
  Future<void> sendFrame(Uint8List frameData) async {
    if (_channel == null) {
      throw Exception('WebSocket not connected');
    }

    try {
      _channel!.sink.add(frameData);
    } catch (e) {
      throw Exception('Failed to send frame: $e');
    }
  }

  /// Close WebSocket connection.
  Future<void> close() async {
    await _channel?.sink.close();
    await _frameController?.close();
    _channel = null;
    _frameController = null;
  }

  /// Check if connected.
  bool get isConnected => _channel != null;
}

