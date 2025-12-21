"""
Device pairing and session management for secure WebSocket connections.
"""
import secrets
import time
import uuid
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import qrcode
from io import BytesIO
import base64


class PairingManager:
    """Manages device pairing codes and session tokens."""
    
    PAIRING_CODE_EXPIRY_MINUTES = 5
    SESSION_TOKEN_EXPIRY_HOURS = 24
    
    def __init__(self):
        # Store active pairing codes: code -> (expiry_time, session_token)
        self._pairing_codes: Dict[str, Tuple[float, str]] = {}
        # Store active sessions: token -> (created_time, device_id)
        self._sessions: Dict[str, Tuple[float, Optional[str]]] = {}
    
    def generate_pairing_code(self) -> Tuple[str, str, str]:
        """
        Generate a new pairing code and session token.
        
        Returns:
            Tuple of (pairing_code, session_token, qr_code_base64)
        """
        # Generate 6-digit pairing code
        pairing_code = f"{secrets.randbelow(1000000):06d}"
        
        # Generate session token (UUID)
        session_token = str(uuid.uuid4())
        
        # Set expiry time
        expiry_time = time.time() + (self.PAIRING_CODE_EXPIRY_MINUTES * 60)
        
        # Store pairing code
        self._pairing_codes[pairing_code] = (expiry_time, session_token)
        
        # Store session
        self._sessions[session_token] = (time.time(), None)
        
        # Generate QR code
        qr_code_base64 = self._generate_qr_code(pairing_code)
        
        return pairing_code, session_token, qr_code_base64
    
    def verify_pairing_code(self, pairing_code: str) -> Optional[str]:
        """
        Verify a pairing code and return the session token if valid.
        
        Args:
            pairing_code: The 6-digit pairing code
            
        Returns:
            Session token if valid, None otherwise
        """
        # Clean up expired codes
        self._cleanup_expired_codes()
        
        if pairing_code not in self._pairing_codes:
            return None
        
        expiry_time, session_token = self._pairing_codes[pairing_code]
        
        # Check if expired
        if time.time() > expiry_time:
            del self._pairing_codes[pairing_code]
            return None
        
        # Remove pairing code (one-time use)
        del self._pairing_codes[pairing_code]
        
        return session_token
    
    def validate_session_token(self, session_token: str) -> bool:
        """
        Validate a session token.
        
        Args:
            session_token: The session token to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Clean up expired sessions
        self._cleanup_expired_sessions()
        
        if session_token not in self._sessions:
            return False
        
        created_time, _ = self._sessions[session_token]
        expiry_time = created_time + (self.SESSION_TOKEN_EXPIRY_HOURS * 3600)
        
        if time.time() > expiry_time:
            del self._sessions[session_token]
            return False
        
        return True
    
    def get_session_info(self, session_token: str) -> Optional[Dict]:
        """
        Get information about a session.
        
        Args:
            session_token: The session token
            
        Returns:
            Session info dict or None if invalid
        """
        if not self.validate_session_token(session_token):
            return None
        
        created_time, device_id = self._sessions[session_token]
        expiry_time = created_time + (self.SESSION_TOKEN_EXPIRY_HOURS * 3600)
        
        return {
            "token": session_token,
            "created_at": datetime.fromtimestamp(created_time).isoformat(),
            "expires_at": datetime.fromtimestamp(expiry_time).isoformat(),
            "device_id": device_id
        }
    
    def _generate_qr_code(self, pairing_code: str) -> str:
        """Generate QR code as base64 string."""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(pairing_code)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        return f"data:image/png;base64,{img_base64}"
    
    def _cleanup_expired_codes(self):
        """Remove expired pairing codes."""
        current_time = time.time()
        expired_codes = [
            code for code, (expiry, _) in self._pairing_codes.items()
            if current_time > expiry
        ]
        for code in expired_codes:
            del self._pairing_codes[code]
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired_tokens = [
            token for token, (created, _) in self._sessions.items()
            if current_time > (created + self.SESSION_TOKEN_EXPIRY_HOURS * 3600)
        ]
        for token in expired_tokens:
            del self._sessions[token]

