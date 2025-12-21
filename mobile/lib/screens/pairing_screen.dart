import 'package:flutter/material.dart';
import 'package:qr_code_scanner/qr_code_scanner.dart';
import '../services/pairing_service.dart';
import '../services/session_manager.dart';

/// Screen for device pairing with QR code scanner and manual code entry.
class PairingScreen extends StatefulWidget {
  final String serverUrl;
  final Function(String sessionToken) onPairingSuccess;

  const PairingScreen({
    Key? key,
    required this.serverUrl,
    required this.onPairingSuccess,
  }) : super(key: key);

  @override
  State<PairingScreen> createState() => _PairingScreenState();
}

class _PairingScreenState extends State<PairingScreen> {
  final TextEditingController _codeController = TextEditingController();
  final GlobalKey _qrKey = GlobalKey(debugLabel: 'QR');
  QRViewController? _qrController;
  bool _isScanning = true;
  bool _isVerifying = false;
  String? _errorMessage;
  bool? _serverReachable;
  bool _isCheckingConnection = false;

  @override
  void initState() {
    super.initState();
    _checkServerConnection();
  }

  @override
  void dispose() {
    _codeController.dispose();
    _qrController?.dispose();
    super.dispose();
  }

  Future<void> _checkServerConnection() async {
    setState(() {
      _isCheckingConnection = true;
    });

    try {
      final pairingService = PairingService(widget.serverUrl);
      final isReachable = await pairingService.checkServerHealth();
      setState(() {
        _serverReachable = isReachable;
        _isCheckingConnection = false;
      });
    } catch (e) {
      setState(() {
        _serverReachable = false;
        _isCheckingConnection = false;
      });
    }
  }

  Future<void> _verifyPairingCode(String code) async {
    if (code.length != 6) {
      setState(() {
        _errorMessage = 'Pairing code must be 6 digits';
      });
      return;
    }

    setState(() {
      _isVerifying = true;
      _errorMessage = null;
    });

    try {
      final pairingService = PairingService(widget.serverUrl);
      
      // Check server connection first
      if (_serverReachable == false) {
        final isReachable = await pairingService.checkServerHealth();
        if (!isReachable) {
          setState(() {
            _errorMessage =
                'Cannot connect to server at ${widget.serverUrl}.\n'
                'Please check:\n'
                '1. Server is running\n'
                '2. Correct IP address and port\n'
                '3. Device and server are on the same network\n'
                '4. Firewall allows connections';
            _isVerifying = false;
            _serverReachable = false;
          });
          return;
        } else {
          setState(() {
            _serverReachable = true;
          });
        }
      }
      
      final sessionToken = await pairingService.verifyPairingCode(code);
      
      // Save session token
      await SessionManager.saveSessionToken(sessionToken);
      await SessionManager.saveServerUrl(widget.serverUrl);

      if (mounted) {
        widget.onPairingSuccess(sessionToken);
      }
    } catch (e) {
      setState(() {
        _errorMessage = e.toString().replaceAll('Exception: ', '');
        _isVerifying = false;
      });
    }
  }

  void _onQRViewCreated(QRViewController controller) {
    setState(() {
      _qrController = controller;
    });

    controller.scannedDataStream.listen((scanData) {
      if (scanData.code != null && _isScanning) {
        setState(() {
          _isScanning = false;
        });
        _verifyPairingCode(scanData.code!);
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pair Device'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Scan QR Code or Enter Pairing Code',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            
            // Server Connection Status
            if (_isCheckingConnection)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                    SizedBox(width: 8),
                    Text('Checking server connection...'),
                  ],
                ),
              )
            else if (_serverReachable != null)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      _serverReachable! ? Icons.check_circle : Icons.error,
                      color: _serverReachable! ? Colors.green : Colors.red,
                      size: 20,
                    ),
                    const SizedBox(width: 8),
                    Text(
                      _serverReachable!
                          ? 'Server connected'
                          : 'Server unreachable',
                      style: TextStyle(
                        color: _serverReachable! ? Colors.green : Colors.red,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    if (!_serverReachable!)
                      IconButton(
                        icon: const Icon(Icons.refresh, size: 20),
                        onPressed: _checkServerConnection,
                        tooltip: 'Retry connection',
                      ),
                  ],
                ),
              ),
            
            const SizedBox(height: 8),
            
            // QR Code Scanner
            if (_isScanning)
              Container(
                height: 300,
                decoration: BoxDecoration(
                  border: Border.all(color: Colors.grey),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: QRView(
                    key: _qrKey,
                    onQRViewCreated: _onQRViewCreated,
                    overlay: QrScannerOverlayShape(
                      borderColor: Colors.blue,
                      borderRadius: 10,
                      borderLength: 30,
                      borderWidth: 10,
                      cutOutSize: 250,
                    ),
                  ),
                ),
              ),
            
            const SizedBox(height: 24),
            
            // Manual Code Entry
            TextField(
              controller: _codeController,
              decoration: const InputDecoration(
                labelText: 'Or Enter 6-Digit Code',
                border: OutlineInputBorder(),
                hintText: '000000',
              ),
              keyboardType: TextInputType.number,
              maxLength: 6,
              enabled: !_isVerifying,
            ),
            
            const SizedBox(height: 16),
            
            // Verify Button
            ElevatedButton(
              onPressed: _isVerifying
                  ? null
                  : () {
                      final code = _codeController.text;
                      if (code.length == 6) {
                        _verifyPairingCode(code);
                      }
                    },
              child: _isVerifying
                  ? const SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Verify Code'),
            ),
            
            // Error Message
            if (_errorMessage != null)
              Padding(
                padding: const EdgeInsets.only(top: 16),
                child: Text(
                  _errorMessage!,
                  style: const TextStyle(color: Colors.red),
                  textAlign: TextAlign.center,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

