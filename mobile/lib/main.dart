import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'package:http/http.dart' as http;
import 'screens/pairing_screen.dart';
import 'services/session_manager.dart';
import 'services/websocket_service.dart';
import 'services/pairing_service.dart';
import 'isolates/camera_stream_isolate.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Initialize cameras
  final cameras = await availableCameras();
  if (cameras.isEmpty) {
    throw Exception('No cameras available');
  }

  runApp(Forge3DApp(cameras: cameras));
}

class Forge3DApp extends StatelessWidget {
  final List<CameraDescription> cameras;

  const Forge3DApp({Key? key, required this.cameras}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Forge 3D',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: MainScreen(cameras: cameras),
    );
  }
}

class MainScreen extends StatefulWidget {
  final List<CameraDescription> cameras;

  const MainScreen({Key? key, required this.cameras}) : super(key: key);

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  String? _sessionToken;
  String? _serverUrl;
  bool _isStreaming = false;
  bool _isExporting = false;
  CameraController? _cameraController;
  WebSocketService? _wsService;
  SendPort? _isolateSendPort;
  Isolate? _streamIsolate;
  int _pendingConversions = 0; // Track pending conversions for parallel processing
  static const int _maxPendingConversions = 8; // Allow up to 8 parallel conversions (increased to reduce drops)
  
  // Frame rate monitoring with rolling window
  final List<DateTime> _frameTimestamps = [];
  int _droppedFrameCount = 0;
  double _currentFps = 0.0;
  Timer? _fpsUpdateTimer;
  DateTime? _lastFpsUpdate;
  
  // Time-based frame rate limiting (disabled - rely on queue and pending conversions)
  DateTime? _lastProcessedFrameTime;

  @override
  void initState() {
    super.initState();
    _loadSavedSession();
  }

  Future<void> _loadSavedSession() async {
    final token = await SessionManager.getSessionToken();
    final url = await SessionManager.getServerUrl();
    
    if (token != null && url != null) {
      setState(() {
        _sessionToken = token;
        _serverUrl = url;
      });
    } else {
      // Default server URL - user should configure this
      setState(() {
        _serverUrl = 'http://192.168.31.128:8000'; // Update with your Mac's IP
      });
    }
  }

  Future<void> _initializeCamera() async {
    if (widget.cameras.isEmpty) return;

    _cameraController = CameraController(
      widget.cameras[0],
      ResolutionPreset.medium, // Keep medium - low might cause issues with frame rate
      enableAudio: false,
    );

    await _cameraController!.initialize();
  }

  Future<void> _startStreaming() async {
    if (_sessionToken == null || _serverUrl == null) {
      _showError('Please pair device first');
      return;
    }

    try {
      // Validate token before attempting connection
      final pairingService = PairingService(_serverUrl!);
      try {
        final validationResult = await pairingService.validateSessionToken(_sessionToken!);
        final expiresInHours = validationResult['expires_in_hours'] as double?;
        if (expiresInHours != null && expiresInHours < 0.1) {
          _showError('Session token expires soon. Please re-pair your device.');
          return;
        }
      } catch (e) {
        // Token is invalid or expired
        _showError('Session token is invalid or expired. Please pair your device again.');
        // Clear invalid token
        await SessionManager.clearSessionToken();
        setState(() {
          _sessionToken = null;
        });
        return;
      }

      await _initializeCamera();
      
      _wsService = WebSocketService(
        serverUrl: _serverUrl!,
        sessionToken: _sessionToken!,
      );
      await _wsService!.connect();

      // Verify connection was established
      if (!_wsService!.isConnected) {
        _showError('Failed to establish WebSocket connection');
        await _wsService!.close();
        return;
      }

      // Reset frame rate limiting
      _lastProcessedFrameTime = null;

      // Start FPS update timer
      _fpsUpdateTimer = Timer.periodic(const Duration(milliseconds: 150), (timer) {
        if (mounted && _isStreaming) {
          _updateFps();
        }
      });

      // Start camera stream
      await _cameraController!.startImageStream((CameraImage image) {
        if (_wsService != null && _wsService!.isConnected) {
          final now = DateTime.now();
          
          // Frame dropping: skip if too many conversions are pending (prevent memory buildup)
          // Check this first since it's the most likely bottleneck
          if (_pendingConversions >= _maxPendingConversions) {
            _droppedFrameCount++;
            return; // Skip this frame to prevent too many parallel conversions
          }
          
          // Queue-aware frame dropping: only skip if queue is actually full (not near full)
          // User reported queue is not full all the time, so be less aggressive
          if (_wsService!.isQueueFull()) {
            _droppedFrameCount++;
            return; // Skip this frame only if queue is completely full
          }
          
          // Time-based limiting removed - rely on queue and pending conversions only
          // This allows maximum throughput while still preventing memory issues
          
          // Increment pending conversions counter
          _pendingConversions++;
          
          // Track frame timestamp for FPS calculation
          _frameTimestamps.add(now);
          
          // Clean up old timestamps (keep only last 2 seconds)
          final cutoffTime = now.subtract(const Duration(seconds: 2));
          _frameTimestamps.removeWhere((timestamp) => timestamp.isBefore(cutoffTime));
          
          // Convert frame in isolate (non-blocking, allows parallel processing)
          CameraStreamIsolate.convertImageToBytes(image).then((frameData) {
            // Send frame non-blocking (queued)
            _wsService!.sendFrame(frameData);
            // Update last processed frame time only after frame is sent
            _lastProcessedFrameTime = DateTime.now();
            _pendingConversions--;
          }).catchError((e) {
            // Handle conversion errors without blocking the stream
            _pendingConversions--;
            if (mounted) {
              // Only show error for critical failures, not for every frame
              print('Frame conversion error: $e');
            }
          });
        }
      });

      setState(() {
        _isStreaming = true;
      });
    } catch (e) {
      String errorMessage = 'Streaming error: $e';
      if (e.toString().contains('Authentication failed') ||
          e.toString().contains('invalid or expired')) {
        errorMessage = 'Session expired. Please pair your device again.';
        // Clear invalid token
        await SessionManager.clearSessionToken();
        setState(() {
          _sessionToken = null;
        });
      }
      _showError(errorMessage);
    }
  }

  /// Update FPS based on rolling window of frame timestamps.
  void _updateFps() {
    final now = DateTime.now();
    final cutoffTime = now.subtract(const Duration(seconds: 1));
    
    // Count frames in the last second
    final framesInLastSecond = _frameTimestamps.where((timestamp) => 
      timestamp.isAfter(cutoffTime)
    ).length;
    
    // Calculate FPS
    final newFps = framesInLastSecond.toDouble();
    
    // Throttle UI updates (only update if FPS changed significantly or enough time passed)
    if (_lastFpsUpdate == null || 
        now.difference(_lastFpsUpdate!).inMilliseconds >= 150 ||
        (newFps - _currentFps).abs() >= 1.0) {
      if (mounted) {
        setState(() {
          _currentFps = newFps;
          _lastFpsUpdate = now;
        });
      }
    }
  }

  Future<void> _stopStreaming() async {
    _fpsUpdateTimer?.cancel();
    _fpsUpdateTimer = null;
    await _cameraController?.stopImageStream();
    await _wsService?.close();
    await _cameraController?.dispose();
    
    setState(() {
      _isStreaming = false;
      _cameraController = null;
      _wsService = null;
      _pendingConversions = 0;
      _frameTimestamps.clear();
      _droppedFrameCount = 0;
      _currentFps = 0.0;
      _lastFpsUpdate = null;
      _lastProcessedFrameTime = null;
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  void _showSuccess(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.green,
      ),
    );
  }

  Future<void> _exportModel() async {
    if (_sessionToken == null || _serverUrl == null) {
      _showError('Please pair device first');
      return;
    }

    if (_isExporting) {
      return; // Already exporting
    }

    setState(() {
      _isExporting = true;
    });

    try {
      // Check model status before attempting export
      final pairingService = PairingService(_serverUrl!);
      final modelStatus = await pairingService.getModelStatus(_sessionToken!);
      
      final status = modelStatus['status'] as String?;
      final vertexCount = modelStatus['vertex_count'] as int? ?? 0;
      final framesProcessed = modelStatus['frames_processed'] as int? ?? 0;
      final framesRejected = modelStatus['frames_rejected'] as int? ?? 0;
      
      if (status == 'no_model' || vertexCount == 0) {
        String message = 'Model is empty. ';
        if (framesProcessed == 0 && framesRejected > 0) {
          message += 'No frames were processed successfully. '
              'Please check that video streaming is active and frames are being sent correctly.';
        } else if (framesProcessed == 0) {
          message += 'No frames have been processed yet. '
              'Please start video streaming and record some video before exporting.';
        } else {
          message += 'Frames were processed but no vertices were generated. '
              'Please ensure you are recording video with the camera.';
        }
        _showError(message);
        setState(() {
          _isExporting = false;
        });
        return;
      }
      
      // Model has vertices, proceed with export
      final uri = Uri.parse('$_serverUrl/export/obj/save?token=$_sessionToken');
      final response = await http
          .post(uri, headers: {'Content-Type': 'application/json'})
          .timeout(const Duration(seconds: 30));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['success'] == true) {
          _showSuccess('Model exported successfully to /tmp/model.obj');
        } else {
          _showError('Export failed: ${data['message'] ?? 'Unknown error'}');
        }
      } else {
        try {
          final errorData = json.decode(response.body);
          _showError(errorData['detail'] ?? 'Export failed');
        } catch (_) {
          _showError('Export failed: ${response.statusCode} ${response.reasonPhrase}');
        }
      }
    } on SocketException catch (e) {
      _showError('Network error: Cannot connect to server. ${e.message}');
    } on TimeoutException {
      _showError('Export timeout: Server did not respond in time');
    } catch (e) {
      _showError('Export error: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isExporting = false;
        });
      }
    }
  }

  Future<void> _handlePairingSuccess(String sessionToken) async {
    setState(() {
      _sessionToken = sessionToken;
    });
    Navigator.pop(context);
  }

  void _showPairingScreen() {
    if (_serverUrl == null) {
      _showError('Server URL not configured');
      return;
    }

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => PairingScreen(
          serverUrl: _serverUrl!,
          onPairingSuccess: _handlePairingSuccess,
        ),
      ),
    );
  }

  @override
  void dispose() {
    _fpsUpdateTimer?.cancel();
    _stopStreaming();
    super.dispose();
  }

  /// Build FPS indicator with color coding.
  Widget _buildFpsIndicator(double fps) {
    Color color;
    if (fps >= 25) {
      color = Colors.green;
    } else if (fps >= 15) {
      color = Colors.orange;
    } else {
      color = Colors.red;
    }
    
    return Container(
      width: 12,
      height: 12,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Forge 3D'),
        actions: [
          if (_sessionToken != null)
            IconButton(
              icon: const Icon(Icons.check_circle, color: Colors.green),
              onPressed: () {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Device paired')),
                );
              },
            ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Server URL Configuration
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Server Configuration',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text('Server: ${_serverUrl ?? "Not configured"}'),
                    const SizedBox(height: 8),
                    Text(
                      'Status: ${_sessionToken != null ? "Paired" : "Not paired"}',
                      style: TextStyle(
                        color: _sessionToken != null ? Colors.green : Colors.orange,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            const SizedBox(height: 24),
            
            // Pairing Button
            if (_sessionToken == null)
              ElevatedButton.icon(
                onPressed: _showPairingScreen,
                icon: const Icon(Icons.qr_code_scanner),
                label: const Text('Pair Device'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
              ),
            
            // Streaming Controls
            if (_sessionToken != null) ...[
              if (!_isStreaming)
                ElevatedButton.icon(
                  onPressed: _startStreaming,
                  icon: const Icon(Icons.videocam),
                  label: const Text('Start Streaming'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    backgroundColor: Colors.green,
                  ),
                )
              else
                ElevatedButton.icon(
                  onPressed: _stopStreaming,
                  icon: const Icon(Icons.stop),
                  label: const Text('Stop Streaming'),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    backgroundColor: Colors.red,
                  ),
                ),
              
              const SizedBox(height: 16),
              
              // Export Model Button
              ElevatedButton.icon(
                onPressed: _isExporting ? null : _exportModel,
                icon: _isExporting
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : const Icon(Icons.download),
                label: Text(_isExporting ? 'Exporting...' : 'Export Model'),
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  backgroundColor: Colors.blue,
                  disabledBackgroundColor: Colors.grey,
                ),
              ),
              
              const SizedBox(height: 16),
              
              // FPS Counter (only show when streaming)
              if (_isStreaming)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Row(
                      mainAxisAlignment: MainAxisAlignment.spaceAround,
                      children: [
                        _buildFpsIndicator(_currentFps),
                        Text(
                          'FPS: ${_currentFps.toStringAsFixed(1)}',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                        Text(
                          'Dropped: $_droppedFrameCount',
                          style: TextStyle(
                            fontSize: 14,
                            color: _droppedFrameCount > 0 ? Colors.orange : Colors.grey,
                          ),
                        ),
                        if (_wsService != null)
                          Text(
                            'Queue: ${_wsService!.getQueueSize()}/${_wsService!.getMaxQueueSize()}',
                            style: TextStyle(
                              fontSize: 12,
                              color: _wsService!.isQueueNearFull() ? Colors.orange : Colors.grey,
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              
              const SizedBox(height: 16),
              
              // Camera Preview
              if (_isStreaming && _cameraController != null)
                Expanded(
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: CameraPreview(_cameraController!),
                  ),
                ),
            ],
          ],
        ),
      ),
    );
  }
}
