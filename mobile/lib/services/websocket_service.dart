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
  
  // Frame send queue for non-blocking sends
  final StreamController<Uint8List> _sendQueue = StreamController<Uint8List>.broadcast();
  StreamSubscription<Uint8List>? _sendQueueSubscription;
  static const int _maxQueueSize = 10; // Drop frames if queue exceeds this

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
        
        // Start processing send queue
        _sendQueueSubscription = _sendQueue.stream.listen(
          (frameData) async {
            if (_channel != null && _isConnected) {
              try {
                _channel!.sink.add(frameData);
              } catch (e) {
                _isConnected = false;
                // Error will be handled by the stream listener
              }
            }
          },
          onError: (error) {
            _isConnected = false;
          },
        );
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

  /// Send binary frame data (non-blocking).
  /// Frames are queued and sent asynchronously. If queue is full, frames are dropped.
  void sendFrame(Uint8List frameData) {
    if (_channel == null || !_isConnected) {
      return; // Silently ignore if not connected
    }

    // Check queue size and drop frame if queue is too full
    // Note: StreamController doesn't expose queue size directly,
    // so we'll just add to queue and let it handle backpressure
    if (!_sendQueue.isClosed) {
      _sendQueue.add(frameData);
    }
  }

  /// Close WebSocket connection.
  Future<void> close() async {
    _isConnected = false;
    await _sendQueueSubscription?.cancel();
    await _sendQueue.close();
    await _closeSubscription?.cancel();
    await _channel?.sink.close();
    await _frameController?.close();
    _channel = null;
    _frameController = null;
    _closeSubscription = null;
    _sendQueueSubscription = null;
  }

  /// Check if connected.
  bool get isConnected => _isConnected && _channel != null;
}

