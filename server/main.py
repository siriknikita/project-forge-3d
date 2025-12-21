"""
FastAPI server for high-performance 3D reconstruction pipeline.
Handles device pairing, WebSocket binary streaming, and model export.
"""
from __future__ import annotations  # Add this at the very top

import asyncio
import sys
import os
from typing import Optional, TYPE_CHECKING
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvloop

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.pairing import PairingManager

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

# CORS middleware for development
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
    session_token = pairing_manager.verify_pairing_code(pairing_code)
    
    if not session_token:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired pairing code"
        )
    
    return {
        "session_token": session_token,
        "expires_in_hours": PairingManager.SESSION_TOKEN_EXPIRY_HOURS
    }


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


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for binary frame streaming.
    Requires valid session token in query params or headers.
    """
    global frame_processor
    
    # Validate session token
    try:
        session_token = validate_session(websocket)
    except HTTPException as e:
        await websocket.close(code=1008, reason=e.detail)
        return
    
    await websocket.accept()
    
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

