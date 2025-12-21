"""
FastAPI server for high-performance 3D reconstruction pipeline.
Handles device pairing, WebSocket binary streaming, and model export.
"""
from __future__ import annotations  # Add this at the very top

import asyncio
import sys
import os
import logging
from typing import Optional, TYPE_CHECKING
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import numpy as np
import uvloop
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.pairing import PairingManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import the C++ module (will fail if not built yet)
try:
    import forge_engine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False
    print("Warning: forge_engine module not found. Build the C++ library first.")
    # Create a dummy type for type hints when module is not available
    class forge_engine:
        class FrameProcessor:
            pass

app = FastAPI(title="Forge Engine 3D Reconstruction Server")


# Request logging middleware
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        
        # Log incoming request
        logger.info(
            f"→ {request.method} {request.url.path} "
            f"[Client: {client_ip}] "
            f"[Query: {dict(request.query_params)}]"
        )
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"← {request.method} {request.url.path} "
                f"[Status: {response.status_code}] "
                f"[Time: {process_time:.3f}s]"
            )
            
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"✗ {request.method} {request.url.path} "
                f"[Error: {str(e)}] "
                f"[Time: {process_time:.3f}s]"
            )
            raise


# Add middleware (order matters - logging should be first)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pairing manager
pairing_manager = PairingManager()

# Global frame processor (initialized when first frame arrives)
# Use string annotation to avoid evaluation at import time
if TYPE_CHECKING:
    from forge_engine import FrameProcessor

frame_processor: Optional["FrameProcessor"] = None
FRAME_CONFIG = {
    "width": 1920,
    "height": 1080,
    "channels": 3  # RGB
}


def get_session_token(websocket: WebSocket) -> Optional[str]:
    """Extract session token from WebSocket query params or headers."""
    # Try query parameter first
    token = websocket.query_params.get("token")
    if token:
        return token
    
    # Try headers
    token = websocket.headers.get("X-Session-Token")
    if token:
        return token
    
    return None


def validate_session(websocket: WebSocket) -> str:
    """Validate session token and return it, or raise exception."""
    token = get_session_token(websocket)
    if not token:
        raise HTTPException(status_code=401, detail="Missing session token")
    
    if not pairing_manager.validate_session_token(token):
        raise HTTPException(status_code=401, detail="Invalid or expired session token")
    
    return token


@app.get("/health")
async def health_check():
    """Health check endpoint (public)."""
    return {
        "status": "healthy",
        "engine_available": ENGINE_AVAILABLE
    }


@app.post("/pair/start")
async def start_pairing():
    """
    Start pairing process. Generates pairing code and QR code.
    Returns pairing code, session token, and QR code image.
    """
    pairing_code, session_token, qr_code_base64 = pairing_manager.generate_pairing_code()
    
    return {
        "pairing_code": pairing_code,
        "session_token": session_token,
        "qr_code": qr_code_base64,
        "expires_in_minutes": PairingManager.PAIRING_CODE_EXPIRY_MINUTES
    }


@app.post("/pair/verify")
async def verify_pairing(pairing_code: str = Query(..., description="6-digit pairing code")):
    """
    Verify pairing code and get session token.
    
    Args:
        pairing_code: 6-digit pairing code (query parameter or form data)
        
    Returns:
        Session token if valid
    """
    logger.info(f"Pairing verification attempt for code: {pairing_code}")
    
    try:
        session_token = pairing_manager.verify_pairing_code(pairing_code)
        
        if not session_token:
            logger.warning(f"Pairing verification failed: Invalid or expired code '{pairing_code}'")
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired pairing code"
            )
        
        logger.info(f"Pairing verification successful for code: {pairing_code} (session token: {session_token[:8]}...)")
        
        return {
            "session_token": session_token,
            "expires_in_hours": PairingManager.SESSION_TOKEN_EXPIRY_HOURS
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during pairing verification for code '{pairing_code}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error during pairing verification"
        )


@app.get("/model/status")
async def get_model_status(token: str = Query(..., description="Session token")):
    """
    Get current model status.
    
    Args:
        token: Session token
        
    Returns:
        Model statistics
    """
    if not pairing_manager.validate_session_token(token):
        raise HTTPException(status_code=401, detail="Invalid session token")
    
    if not ENGINE_AVAILABLE or frame_processor is None:
        return {
            "status": "no_model",
            "message": "No model has been created yet"
        }
    
    stats = frame_processor.getStats()
    model_stats = frame_processor.getModel().getStatistics()
    
    return {
        "status": "active",
        "frames_processed": stats.frames_processed,
        "avg_processing_time_ms": stats.avg_processing_time_ms,
        "vertex_count": model_stats.vertex_count,
        "face_count": model_stats.face_count,
        "bounding_box": {
            "min": [model_stats.min_x, model_stats.min_y, model_stats.min_z],
            "max": [model_stats.max_x, model_stats.max_y, model_stats.max_z]
        }
    }


@app.get("/export/ply")
async def export_ply(
    token: str = Query(..., description="Session token"),
    binary: bool = Query(True, description="Export as binary PLY")
):
    """
    Export model as PLY file.
    
    Args:
        token: Session token
        binary: Export as binary (True) or ASCII (False)
        
    Returns:
        PLY file download
    """
    if not pairing_manager.validate_session_token(token):
        raise HTTPException(status_code=401, detail="Invalid session token")
    
    if not ENGINE_AVAILABLE or frame_processor is None:
        raise HTTPException(status_code=404, detail="No model available")
    
    model = frame_processor.getModel()
    if model.getVertexCount() == 0:
        raise HTTPException(status_code=404, detail="Model is empty")
    
    filename = f"model_{int(asyncio.get_event_loop().time())}.ply"
    filepath = f"/tmp/{filename}"
    
    success = model.exportPLY(filepath, binary=binary)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to export model")
    
    return FileResponse(
        filepath,
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/export/obj")
async def export_obj(token: str = Query(..., description="Session token")):
    """
    Export model as OBJ file.
    
    Args:
        token: Session token
        
    Returns:
        OBJ file download
    """
    if not pairing_manager.validate_session_token(token):
        raise HTTPException(status_code=401, detail="Invalid session token")
    
    if not ENGINE_AVAILABLE or frame_processor is None:
        raise HTTPException(status_code=404, detail="No model available")
    
    model = frame_processor.getModel()
    if model.getVertexCount() == 0:
        raise HTTPException(status_code=404, detail="Model is empty")
    
    filename = f"model_{int(asyncio.get_event_loop().time())}.obj"
    filepath = f"/tmp/{filename}"
    
    success = model.exportOBJ(filepath)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to export model")
    
    return FileResponse(
        filepath,
        media_type="application/octet-stream",
        filename=filename
    )


@app.get("/session/validate")
async def validate_session_endpoint(token: str = Query(..., description="Session token")):
    """
    Validate session token and return token information.
    Useful for checking token validity before attempting WebSocket connection.
    
    Args:
        token: Session token to validate
        
    Returns:
        Token validation result with expiry information
    """
    session_info = pairing_manager.get_session_info(token)
    
    if not session_info:
        # Token is invalid or expired
        logger.info(f"Session validation failed for token '{token[:8]}...'")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session token"
        )
    
    # Calculate time remaining
    expires_at = datetime.fromisoformat(session_info['expires_at'])
    time_remaining_seconds = (expires_at - datetime.now()).total_seconds()
    time_remaining_hours = time_remaining_seconds / 3600
    
    logger.info(
        f"Session validation successful for token '{token[:8]}...' "
        f"(expires in {time_remaining_hours:.2f} hours)"
    )
    
    return {
        "valid": True,
        "token": token[:8] + "...",  # Only return partial token for security
        "created_at": session_info["created_at"],
        "expires_at": session_info["expires_at"],
        "expires_in_hours": round(time_remaining_hours, 2),
        "expires_in_seconds": int(time_remaining_seconds)
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for binary frame streaming.
    Requires valid session token in query params or headers.
    """
    global frame_processor
    
    # Validate session token BEFORE accepting connection
    token = get_session_token(websocket)
    client_ip = websocket.client.host if websocket.client else 'unknown'
    
    if not token:
        logger.warning(f"WebSocket connection rejected: Missing session token from {client_ip}")
        # Accept connection first, then close with error code
        await websocket.accept()
        await websocket.close(code=1002, reason="Missing session token")
        return
    
    # Validate token and get detailed info for logging
    token_valid = pairing_manager.validate_session_token(token)
    if not token_valid:
        # Get token info for detailed logging
        session_info = pairing_manager.get_session_info(token)
        if session_info:
            # Token exists but expired
            logger.warning(
                f"WebSocket connection rejected: Expired session token '{token[:8]}...' "
                f"from {client_ip} (expired at {session_info.get('expires_at', 'unknown')})"
            )
        else:
            # Token doesn't exist
            logger.warning(
                f"WebSocket connection rejected: Invalid session token '{token[:8]}...' "
                f"from {client_ip} (token not found in active sessions)"
            )
        # Accept connection first, then close with error code
        await websocket.accept()
        await websocket.close(code=1008, reason="Invalid or expired session token")
        return
    
    session_token = token
    await websocket.accept()
    logger.info(f"WebSocket connection accepted: session_token={session_token[:8]}... from {client_ip}")
    
    # Initialize frame processor if needed
    if ENGINE_AVAILABLE and frame_processor is None:
        config = forge_engine.FrameConfig()
        config.width = FRAME_CONFIG["width"]
        config.height = FRAME_CONFIG["height"]
        config.channels = FRAME_CONFIG["channels"]
        frame_processor = forge_engine.FrameProcessor(config)
    
    if not ENGINE_AVAILABLE:
        await websocket.send_text("ERROR: C++ engine not available")
        await websocket.close()
        return
    
    try:
        while True:
            # Receive binary frame data
            data = await websocket.receive_bytes()
            
            # Convert to numpy array (zero-copy with pybind11)
            frame_array = np.frombuffer(data, dtype=np.uint8)
            
            # Process frame (zero-copy to C++)
            frame_processor.processFrame(frame_array)
            
            # Optional: Send acknowledgment
            # await websocket.send_text("OK")
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_token}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011, reason=str(e))


if __name__ == "__main__":
    # Use uvloop for better performance
    try:
        uvloop.install()
        loop_type = "uvloop"
    except Exception:
        loop_type = "auto"
    
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        loop=loop_type
    )

