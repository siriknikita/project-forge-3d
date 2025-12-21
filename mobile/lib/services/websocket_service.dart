import 'dart:async';
import 'dart:collection';
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
  static const int _maxQueueSize = 30; // Drop frames if queue exceeds this
  
  // Bounded queue with size tracking
  final Queue<Uint8List> _frameQueue = Queue<Uint8List>();
  int _queueSize = 0;
  bool _isProcessingQueue = false;

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
                // Add to bounded queue (FIFO drop if full)
                _addToQueue(frameData);
                
                // Process queue asynchronously
                _processQueue();
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

  /// Add frame to bounded queue (FIFO drop if full).
  void _addToQueue(Uint8List frameData) {
    if (_queueSize >= _maxQueueSize) {
      // Queue is full, drop oldest frame (FIFO)
      if (_frameQueue.isNotEmpty) {
        _frameQueue.removeFirst();
        _queueSize--;
      }
    }
    _frameQueue.add(frameData);
    _queueSize++;
  }

  /// Process queue asynchronously (one frame at a time).
  Future<void> _processQueue() async {
    if (_isProcessingQueue || !_isConnected) {
      return;
    }

    _isProcessingQueue = true;
    
    // Process all frames in queue
    while (_frameQueue.isNotEmpty && _isConnected) {
      final queuedFrame = _frameQueue.removeFirst();
      _queueSize--;
      
      try {
        if (_channel != null && _isConnected) {
          _channel!.sink.add(queuedFrame);
        }
      } catch (e) {
        _isConnected = false;
        break;
      }
      
      // No artificial delay - process frames as fast as possible
    }
    
    _isProcessingQueue = false;
    
    // If there are more frames, process them (handles case where frames arrived while processing)
    if (_frameQueue.isNotEmpty && _isConnected) {
      _processQueue();
    }
  }

  /// Send binary frame data (non-blocking).
  /// Frames are queued and sent asynchronously. If queue is full, frames are dropped.
  void sendFrame(Uint8List frameData) {
    if (_channel == null || !_isConnected) {
      return; // Silently ignore if not connected
    }

    if (!_sendQueue.isClosed) {
      _sendQueue.add(frameData);
    }
  }

  /// Check if queue is full or near capacity.
  bool isQueueFull() {
    return _queueSize >= _maxQueueSize;
  }

  /// Get current queue size.
  int getQueueSize() {
    return _queueSize;
  }

  /// Check if queue is at or above threshold (e.g., 80% capacity).
  bool isQueueNearFull() {
    return _queueSize >= (_maxQueueSize * 0.8).round();
  }

  /// Get maximum queue size.
  int getMaxQueueSize() {
    return _maxQueueSize;
  }

  /// Close WebSocket connection.
  Future<void> close() async {
    _isConnected = false;
    await _sendQueueSubscription?.cancel();
    await _sendQueue.close();
    await _closeSubscription?.cancel();
    await _channel?.sink.close();
    await _frameController?.close();
    _frameQueue.clear();
    _queueSize = 0;
    _channel = null;
    _frameController = null;
    _closeSubscription = null;
    _sendQueueSubscription = null;
  }

  /// Check if connected.
  bool get isConnected => _isConnected && _channel != null;
}

