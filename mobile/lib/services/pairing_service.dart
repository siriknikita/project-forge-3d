import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

/// Service for device pairing with the server.
class PairingService {
  final String serverUrl;
  static const Duration _defaultTimeout = Duration(seconds: 10);

  PairingService(this.serverUrl);

  /// Check if server is reachable.
  Future<bool> checkServerHealth() async {
    try {
      final response = await http
          .get(Uri.parse('$serverUrl/health'))
          .timeout(_defaultTimeout);

      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }

  /// Start pairing process and get pairing code.
  Future<PairingResponse> startPairing() async {
    try {
      final response = await http
          .post(
            Uri.parse('$serverUrl/pair/start'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(_defaultTimeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return PairingResponse.fromJson(data);
      } else {
        throw Exception('Failed to start pairing: ${response.statusCode}');
      }
    } on SocketException catch (e) {
      throw Exception(
          'Network error: Cannot connect to server. Please check if the server is running and reachable at $serverUrl. Error: ${e.message}');
    } on HttpException catch (e) {
      throw Exception('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw Exception('Invalid server response: ${e.message}');
    } on TimeoutException catch (e) {
      throw Exception(
          'Connection timeout: Server did not respond within ${_defaultTimeout.inSeconds} seconds. Please check if the server is running at $serverUrl');
    } catch (e) {
      if (e.toString().contains('timeout') ||
          e.toString().contains('timed out')) {
        throw Exception(
            'Connection timeout: Server did not respond. Please check if the server is running at $serverUrl');
      }
      throw Exception('Pairing error: $e');
    }
  }

  /// Verify pairing code and get session token.
  Future<String> verifyPairingCode(String pairingCode) async {
    try {
      final response = await http
          .post(
            Uri.parse('$serverUrl/pair/verify?pairing_code=$pairingCode'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(_defaultTimeout);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['session_token'] as String;
      } else {
        try {
          final errorData = json.decode(response.body);
          throw Exception(errorData['detail'] ?? 'Invalid pairing code');
        } catch (_) {
          throw Exception(
              'Server returned error: ${response.statusCode} ${response.reasonPhrase}');
        }
      }
    } on SocketException catch (e) {
      throw Exception(
          'Network error: Cannot connect to server at $serverUrl. Please check:\n'
          '1. Server is running\n'
          '2. Correct IP address and port\n'
          '3. Device and server are on the same network\n'
          '4. Firewall allows connections\n'
          'Error: ${e.message}');
    } on HttpException catch (e) {
      throw Exception('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw Exception('Invalid server response: ${e.message}');
    } on TimeoutException catch (e) {
      throw Exception(
          'Connection timeout: Server at $serverUrl did not respond within ${_defaultTimeout.inSeconds} seconds.\n'
          'Please check if the server is running and accessible.');
    } catch (e) {
      // Check if it's already a formatted error message
      if (e.toString().contains('Network error') ||
          e.toString().contains('Connection timeout') ||
          e.toString().contains('HTTP error')) {
        rethrow;
      }
      // Check for timeout in error message
      if (e.toString().contains('timeout') ||
          e.toString().contains('timed out') ||
          e.toString().contains('Connection timed out')) {
        throw Exception(
            'Connection timeout: Server at $serverUrl did not respond.\n'
            'Please check if the server is running and accessible.');
      }
      throw Exception('Verification error: $e');
    }
  }

  /// Validate session token with server.
  Future<Map<String, dynamic>> validateSessionToken(String token) async {
    try {
      final response = await http
          .get(
            Uri.parse('$serverUrl/session/validate?token=$token'),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(_defaultTimeout);

      if (response.statusCode == 200) {
        return json.decode(response.body) as Map<String, dynamic>;
      } else {
        try {
          final errorData = json.decode(response.body);
          throw Exception(errorData['detail'] ?? 'Invalid session token');
        } catch (_) {
          throw Exception(
              'Server returned error: ${response.statusCode} ${response.reasonPhrase}');
        }
      }
    } on SocketException catch (e) {
      throw Exception(
          'Network error: Cannot connect to server at $serverUrl. Error: ${e.message}');
    } on HttpException catch (e) {
      throw Exception('HTTP error: ${e.message}');
    } on FormatException catch (e) {
      throw Exception('Invalid server response: ${e.message}');
    } on TimeoutException catch (e) {
      throw Exception(
          'Connection timeout: Server at $serverUrl did not respond within ${_defaultTimeout.inSeconds} seconds.');
    } catch (e) {
      if (e.toString().contains('Invalid or expired session token')) {
        throw Exception('Session token is invalid or expired. Please pair your device again.');
      }
      throw Exception('Token validation error: $e');
    }
  }
}

/// Response from pairing start endpoint.
class PairingResponse {
  final String pairingCode;
  final String sessionToken;
  final String qrCode;
  final int expiresInMinutes;

  PairingResponse({
    required this.pairingCode,
    required this.sessionToken,
    required this.qrCode,
    required this.expiresInMinutes,
  });

  factory PairingResponse.fromJson(Map<String, dynamic> json) {
    return PairingResponse(
      pairingCode: json['pairing_code'] as String,
      sessionToken: json['session_token'] as String,
      qrCode: json['qr_code'] as String,
      expiresInMinutes: json['expires_in_minutes'] as int,
    );
  }
}

