"""
Device pairing and session management for secure WebSocket connections.
"""
import secrets
import time
import uuid
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import qrcode
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


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
        created_time = time.time()
        self._sessions[session_token] = (created_time, None)
        
        # Generate QR code
        qr_code_base64 = self._generate_qr_code(pairing_code)
        
        logger.info(
            f"Generated pairing code: {pairing_code}, "
            f"session_token: {session_token[:8]}... "
            f"(expires in {self.PAIRING_CODE_EXPIRY_MINUTES} minutes, "
            f"session expires in {self.SESSION_TOKEN_EXPIRY_HOURS} hours)"
        )
        
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
            logger.debug(f"Pairing code '{pairing_code}' not found in active codes")
            return None
        
        expiry_time, session_token = self._pairing_codes[pairing_code]
        current_time = time.time()
        
        # Check if expired
        if current_time > expiry_time:
            logger.debug(
                f"Pairing code '{pairing_code}' expired "
                f"(expired {current_time - expiry_time:.1f} seconds ago)"
            )
            del self._pairing_codes[pairing_code]
            return None
        
        # Remove pairing code (one-time use)
        del self._pairing_codes[pairing_code]
        
        # Log successful verification
        if session_token in self._sessions:
            created_time, _ = self._sessions[session_token]
            token_age = current_time - created_time
            logger.info(
                f"Pairing code '{pairing_code}' verified successfully, "
                f"session_token: {session_token[:8]}... "
                f"(token age: {token_age:.1f} seconds)"
            )
        
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
        current_time = time.time()
        
        if current_time > expiry_time:
            del self._sessions[session_token]
            return False
        
        return True
    
    def get_token_age_seconds(self, session_token: str) -> Optional[float]:
        """
        Get the age of a session token in seconds.
        
        Args:
            session_token: The session token
            
        Returns:
            Token age in seconds, or None if token doesn't exist
        """
        if session_token not in self._sessions:
            return None
        
        created_time, _ = self._sessions[session_token]
        return time.time() - created_time
    
    def get_token_expiry_remaining_seconds(self, session_token: str) -> Optional[float]:
        """
        Get the remaining time until token expiry in seconds.
        
        Args:
            session_token: The session token
            
        Returns:
            Remaining seconds until expiry, or None if token doesn't exist or is expired
        """
        if session_token not in self._sessions:
            return None
        
        created_time, _ = self._sessions[session_token]
        expiry_time = created_time + (self.SESSION_TOKEN_EXPIRY_HOURS * 3600)
        current_time = time.time()
        
        if current_time > expiry_time:
            return None
        
        return expiry_time - current_time
    
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
        if expired_codes:
            logger.debug(f"Cleaning up {len(expired_codes)} expired pairing codes")
            for code in expired_codes:
                del self._pairing_codes[code]
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        current_time = time.time()
        expired_tokens = [
            token for token, (created, _) in self._sessions.items()
            if current_time > (created + self.SESSION_TOKEN_EXPIRY_HOURS * 3600)
        ]
        if expired_tokens:
            logger.debug(f"Cleaning up {len(expired_tokens)} expired session tokens")
            for token in expired_tokens:
                del self._sessions[token]

