"""
MVLM Web UI API - FastAPI backend for tracker control and management

Usage:
    python tracking/web/api.py --port 8080

Or for development with auto-reload:
    uvicorn tracking.web.api:app --reload --port 8080
"""

import asyncio
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any

import cv2 as cv
import numpy as np
import torch
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Form,
    HTTPException,
    BackgroundTasks,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.tracker import Tracker
from lib.config.mvlm.config import cfg, update_config_from_file
from tracking.web.server import MVLMTrackerService


# ============================================================================
# Pydantic Models for API
# ============================================================================

class ControlRequest(BaseModel):
    action: str  # 'pause', 'resume', 'quit'


class TargetSwitchRequest(BaseModel):
    text: str
    bbox: Optional[List[int]] = None  # [x, y, w, h]


class ModelLoadRequest(BaseModel):
    config: str
    checkpoint: str
    dataset: Optional[str] = "TNL2K"


class ParameterUpdateRequest(BaseModel):
    parameters: Dict[str, Any]


class VideoSourceRequest(BaseModel):
    source_type: str  # 'file', 'url', 'webcam'
    path: Optional[str] = None
    url: Optional[str] = None
    device_id: Optional[int] = 0


# ============================================================================
# Global State
# ============================================================================

# Global tracker service instance
tracker_service: Optional[MVLMTrackerService] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass


manager = ConnectionManager()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global tracker_service
    tracker_service = MVLMTrackerService()
    print("MVLM Web UI server started")
    yield
    # Shutdown
    if tracker_service:
        await tracker_service.cleanup()
    print("MVLM Web UI server stopped")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="MVLM Tracker WebUI",
    description="Web interface for MVLM multi-modal object tracking",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Vue Frontend Static Files
# ============================================================================

FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial state
        if tracker_service:
            await websocket.send_json({
                "type": "status",
                "data": tracker_service.get_status()
            })

        # Keep connection alive and send updates
        while True:
            # Send periodic status updates
            if tracker_service:
                status = tracker_service.get_status()
                await websocket.send_json({
                    "type": "status_update",
                    "data": status
                })
            await asyncio.sleep(0.1)  # 10 Hz update rate
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Serve the Vue frontend index.html"""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "message": "MVLM Web UI API - Frontend not built. Run: cd tracking/web/frontend && npm install && npm run build",
        "version": "2.0.0",
    }


@app.get("/api/status")
async def get_status():
    """Get current tracker status"""
    if not tracker_service:
        return JSONResponse(
            status_code=503,
            content={"error": "Tracker service not initialized"}
        )
    return tracker_service.get_status()


@app.post("/api/control/pause")
async def pause_tracking():
    """Pause/resume tracking"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    if tracker_service.is_paused:
        tracker_service.resume()
        return {"status": "ok", "message": "Resumed", "paused": False}
    else:
        tracker_service.pause()
        return {"status": "ok", "message": "Paused", "paused": True}


@app.post("/api/control/quit")
async def quit_tracker():
    """Stop tracker and cleanup"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    await tracker_service.stop()
    return {"status": "ok", "message": "Tracker stopped"}


@app.post("/api/target/switch")
async def switch_target(request: TargetSwitchRequest):
    """Switch to a new target with optional bbox"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    try:
        result = await tracker_service.switch_target(
            text=request.text,
            bbox=request.bbox
        )
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/current_target")
async def get_current_target():
    """Get current target information"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    return {
        "text": tracker_service.current_text,
        "paused": tracker_service.is_paused
    }


@app.get("/api/models")
async def list_models():
    """List available trained models"""
    from tracking.web.models import list_available_models
    models = list_available_models()
    return {"models": models}


@app.get("/api/configs")
async def list_configs():
    """List available experiment config YAML files"""
    from tracking.web.models import list_available_configs
    configs = list_available_configs()
    return {"configs": configs}


@app.post("/api/model/load")
async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """Load a specific model checkpoint"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    # Load in background
    def load_task():
        try:
            asyncio.run(tracker_service.load_model(
                config_name=request.config,
                checkpoint_path=request.checkpoint,
                dataset_name=request.dataset
            ))
        except Exception as e:
            print(f"Error loading model: {e}")

    background_tasks.add_task(load_task)
    return {"status": "ok", "message": "Model loading started"}


@app.get("/api/parameters")
async def get_parameters():
    """Get current tracker parameters"""
    from tracking.web.config.web_ui_config import PARAMETER_GROUPS
    return PARAMETER_GROUPS


@app.post("/api/parameters")
async def update_parameters(request: ParameterUpdateRequest):
    """Update tracker parameters"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    try:
        tracker_service.update_parameters(request.parameters)
        return {"status": "ok", "message": "Parameters updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for tracking"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    # Save uploaded file (use temp dir under project root for cross-platform support)
    import tempfile
    upload_dir = Path(tempfile.gettempdir()) / "mvlm_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "status": "ok",
        "message": "Video uploaded",
        "path": str(file_path)
    }


@app.post("/api/video/load")
async def load_video(request: VideoSourceRequest):
    """Load video from file path or URL"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    try:
        if request.source_type == "file" and request.path:
            await tracker_service.load_video(request.path)
        elif request.source_type == "url" and request.url:
            await tracker_service.load_video_from_url(request.url)
        elif request.source_type == "webcam":
            await tracker_service.load_webcam(request.device_id or 0)
        else:
            raise HTTPException(status_code=400, detail="Invalid source")

        return {"status": "ok", "message": "Video loaded"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/video/start")
async def start_tracking(request: TargetSwitchRequest):
    """Start tracking with the given target description"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    try:
        await tracker_service.start_tracking(
            text=request.text,
            bbox=request.bbox
        )
        return {"status": "ok", "message": "Tracking started"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/video/first_frame")
async def get_first_frame():
    """Get the first frame of the loaded video as JPEG (for preview / bbox selection)"""
    if not tracker_service or tracker_service._first_frame is None:
        raise HTTPException(status_code=404, detail="No video loaded or first frame not available")
    _, jpeg = cv.imencode('.jpg', tracker_service._first_frame, [cv.IMWRITE_JPEG_QUALITY, 85])
    return StreamingResponse(io.BytesIO(jpeg.tobytes()), media_type="image/jpeg")


@app.get("/api/video/seek/{frame_number}")
async def seek_frame(frame_number: int):
    """Seek to a specific frame"""
    if not tracker_service:
        raise HTTPException(status_code=503, detail="Tracker service not initialized")

    try:
        await tracker_service.seek_frame(frame_number)
        return {"status": "ok", "message": f"Seeked to frame {frame_number}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# MJPEG Stream Endpoint (legacy compatibility)
# ============================================================================

async def generate_mjpeg():
    """Generator for MJPEG streaming"""
    while tracker_service and tracker_service.is_running:
        frame = tracker_service.get_latest_frame()
        if frame is not None:
            _, jpeg = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 80])
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        await asyncio.sleep(0.033)  # ~30 FPS


@app.get("/stream")
async def video_stream():
    """MJPEG video stream endpoint"""
    if not tracker_service or not tracker_service.is_running:
        return StreamingResponse(
            iter([b'']),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# Legacy Endpoints (for backward compatibility with old HTML UI)
# ============================================================================

@app.post("/control")
async def control_legacy(request: ControlRequest):
    """Legacy control endpoint for old UI"""
    if not tracker_service:
        return JSONResponse(
            status_code=503,
            content={"error": "Tracker service not initialized"}
        )

    action = request.action
    if action == 'pause':
        tracker_service.pause()
        return {"status": "ok", "message": "Paused"}
    elif action == 'resume':
        tracker_service.resume()
        return {"status": "ok", "message": "Resumed"}
    elif action == 'quit':
        await tracker_service.stop()
        return {"status": "ok", "message": "Quit signal sent"}
    else:
        return JSONResponse(status_code=400, content={"error": "Unknown action"})


@app.post("/new_target")
async def new_target_legacy(request: TargetSwitchRequest):
    """Legacy target switch endpoint for old UI"""
    if not tracker_service:
        return JSONResponse(
            status_code=503,
            content={"error": "Tracker service not initialized"}
        )

    try:
        result = await tracker_service.switch_target(
            text=request.text,
            bbox=request.bbox
        )
        return {"status": "queued", **result}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ============================================================================
# Mount Vue frontend static assets (must be after all API routes)
# ============================================================================

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="frontend-assets")


# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MVLM Web UI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--device", type=str, default=None, help='Force device: "cpu" or "cuda" (default: auto-detect)')

    args = parser.parse_args()

    # Store device choice for tracker service to pick up
    if args.device:
        os.environ['MVLM_DEVICE'] = args.device

    print("=" * 60)
    print("MVLM Web UI Server")
    print("=" * 60)
    print(f"Device: {args.device or 'auto'}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"WebSocket: ws://{args.host}:{args.port}/ws")
    print(f"Stream: http://{args.host}:{args.port}/stream")
    print("=" * 60)

    uvicorn.run(
        "tracking.web.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
