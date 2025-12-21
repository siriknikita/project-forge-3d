import 'package:shared_preferences/shared_preferences.dart';

/// Manages session token storage and retrieval.
class SessionManager {
  static const String _sessionTokenKey = 'forge_session_token';
  static const String _serverUrlKey = 'forge_server_url';

  /// Save session token securely.
  static Future<void> saveSessionToken(String token) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_sessionTokenKey, token);
  }

  /// Get stored session token.
  static Future<String?> getSessionToken() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_sessionTokenKey);
  }

  /// Clear session token.
  static Future<void> clearSessionToken() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_sessionTokenKey);
  }

  /// Save server URL.
  static Future<void> saveServerUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_serverUrlKey, url);
  }

  /// Get stored server URL.
  static Future<String?> getServerUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_serverUrlKey);
  }

  /// Check if session token exists.
  static Future<bool> hasSessionToken() async {
    final token = await getSessionToken();
    return token != null && token.isNotEmpty;
  }
}

