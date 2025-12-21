import 'dart:async';
import 'dart:typed_data';
import 'package:web_socket_channel/web_socket_channel.dart';

/// WebSocket service for binary frame streaming.
class WebSocketService {
  WebSocketChannel? _channel;
  final String serverUrl;
  final String sessionToken;
  StreamController<Uint8List>? _frameController;
  StreamSubscription? _closeSubscription;
  bool _isConnected = false;

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
      
      // Listen for connection close events
      _closeSubscription = _channel!.stream.listen(
        (data) {
          // Handle incoming data if needed
        },
        onError: (error) {
          _isConnected = false;
          throw Exception('WebSocket error: $error');
        },
        onDone: () {
          _isConnected = false;
        },
        cancelOnError: true,
      );
      
      // Wait a bit to check if connection was established
      await Future.delayed(const Duration(milliseconds: 100));
      
      // Check if connection is still open
      if (_channel != null) {
        _isConnected = true;
      }
    } catch (e) {
      _isConnected = false;
      // Provide more specific error messages
      if (e.toString().contains('Invalid or expired session token') ||
          e.toString().contains('Missing session token')) {
        throw Exception('Authentication failed: Session token is invalid or expired. Please pair your device again.');
      } else if (e.toString().contains('Connection refused') ||
                 e.toString().contains('Failed host lookup')) {
        throw Exception('Cannot connect to server. Please check if the server is running at $serverUrl');
      } else {
        throw Exception('WebSocket connection error: $e');
      }
    }
  }

  /// Send binary frame data.
  Future<void> sendFrame(Uint8List frameData) async {
    if (_channel == null || !_isConnected) {
      throw Exception('WebSocket not connected');
    }

    try {
      _channel!.sink.add(frameData);
    } catch (e) {
      _isConnected = false;
      throw Exception('Failed to send frame: $e');
    }
  }

  /// Close WebSocket connection.
  Future<void> close() async {
    _isConnected = false;
    await _closeSubscription?.cancel();
    await _channel?.sink.close();
    await _frameController?.close();
    _channel = null;
    _frameController = null;
    _closeSubscription = null;
  }

  /// Check if connected.
  bool get isConnected => _isConnected && _channel != null;
}

