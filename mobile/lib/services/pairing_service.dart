import 'dart:convert';
import 'package:http/http.dart' as http;

/// Service for device pairing with the server.
class PairingService {
  final String serverUrl;

  PairingService(this.serverUrl);

  /// Start pairing process and get pairing code.
  Future<PairingResponse> startPairing() async {
    try {
      final response = await http.post(
        Uri.parse('$serverUrl/pair/start'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return PairingResponse.fromJson(data);
      } else {
        throw Exception('Failed to start pairing: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Pairing error: $e');
    }
  }

  /// Verify pairing code and get session token.
  Future<String> verifyPairingCode(String pairingCode) async {
    try {
      final response = await http.post(
        Uri.parse('$serverUrl/pair/verify?pairing_code=$pairingCode'),
        headers: {'Content-Type': 'application/json'},
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        return data['session_token'] as String;
      } else {
        final errorData = json.decode(response.body);
        throw Exception(errorData['detail'] ?? 'Invalid pairing code');
      }
    } catch (e) {
      throw Exception('Verification error: $e');
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

