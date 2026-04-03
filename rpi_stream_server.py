"""
Raspberry Pi Video Streaming Server
=====================================
Streams camera feed over the network via HTTP.
Anyone on the network can view the stream in a browser.

Endpoints:
- /              â†’ Web page with video player
- /video_feed    â†’ MJPEG stream (for browsers/VLC)
- /snapshot      â†’ Single JPEG image
- /api/frame     â†’ Raw frame bytes for API consumption

Run: python rpi_stream_server.py --port 8080
View: http://RASPBERRY_PI_IP:8080
"""

import io
import time
import threading
import argparse
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# === Configuration ===

@dataclass
class StreamConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    quality: int = 85  # JPEG quality (1-100)
    port: int = 8080
    camera_index: int = 0  # USB camera index


# === Camera Manager ===

class CameraStream:
    """
    Thread-safe camera capture with frame buffering.
    Provides latest frame to multiple clients.
    """
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.camera = None
        self.frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        self.fps_actual = 0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
        self._init_camera()
        self._start_capture_thread()
    
    def _init_camera(self):
        """Initialize USB webcam."""
        print(f"Initializing USB Webcam (index {self.config.camera_index})...")
        
        # Try camera indices starting from config
        camera_found = False
        for idx in range(self.config.camera_index, self.config.camera_index + 3):
            self.camera = cv2.VideoCapture(idx)
            if self.camera.isOpened():
                # Test if we can actually read a frame
                ret, test_frame = self.camera.read()
                if ret and test_frame is not None:
                    camera_found = True
                    self.config.camera_index = idx
                    print(f"âœ“ USB Webcam found and working (index {idx})")
                    break
                else:
                    self.camera.release()
                    print(f"Camera index {idx} opened but cannot read frames")
            else:
                if idx == self.config.camera_index:
                    print(f"Camera index {idx} not available, trying next...")
        
        if not camera_found:
            raise RuntimeError(f"Failed to initialize camera. Tried indices {self.config.camera_index} to {self.config.camera_index + 2}")
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.config.fps)
        
        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        time.sleep(2)  # Camera warm-up
        
        # Test capture
        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise RuntimeError("Camera initialized but cannot capture frames")
        
        print(f"âœ“ USB Webcam ready ({actual_width}x{actual_height})")
        print(f"âœ“ Camera test capture successful")
    
    def _capture_loop(self):
        """Continuous capture loop running in background thread."""
        while self.running:
            try:
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                # Add timestamp overlay
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, timestamp, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update frame buffer
                with self.frame_lock:
                    self.frame = frame
                    self.frame_count += 1
                
                # Calculate FPS
                self._fps_frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self.fps_actual = self._fps_frame_count / (now - self._last_fps_time)
                    self._fps_frame_count = 0
                    self._last_fps_time = now
                
                # Limit frame rate
                time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)
    
    def _start_capture_thread(self):
        """Start background capture thread."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("âœ“ Capture thread started")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (thread-safe)."""
        with self.frame_lock:
            if self.frame is not None:
                return self.frame.copy()
        return None
    
    def get_jpeg(self) -> Optional[bytes]:
        """Get latest frame as JPEG bytes."""
        frame = self.get_frame()
        if frame is None:
            return None
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return buffer.tobytes()
    
    def stop(self):
        """Stop capture and release USB webcam."""
        self.running = False
        time.sleep(0.2)
        self.camera.release()
        print("USB Webcam released")


# === FastAPI Server ===

app = FastAPI(
    title="Raspberry Pi USB Camera Stream",
    description="Live video streaming from USB webcam on Raspberry Pi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global camera instance
camera_stream: Optional[CameraStream] = None


def generate_mjpeg():
    """Generator for MJPEG stream."""
    while True:
        jpeg = camera_stream.get_jpeg()
        if jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        time.sleep(1.0 / 30)  # ~30 FPS


# === Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def index():
    """Web page with embedded video player."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Camera</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', Tahoma, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                color: #eee;
            }
            h1 {
                margin-bottom: 20px;
                font-size: 2em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .container {
                background: rgba(255,255,255,0.1);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                backdrop-filter: blur(10px);
            }
            img {
                border-radius: 10px;
                max-width: 100%;
                height: auto;
            }
            .info {
                margin-top: 20px;
                padding: 15px;
                background: rgba(0,0,0,0.3);
                border-radius: 10px;
                font-family: monospace;
            }
            .info p { margin: 5px 0; }
            .endpoint {
                color: #4ade80;
                background: rgba(74, 222, 128, 0.1);
                padding: 2px 8px;
                border-radius: 4px;
            }
            .controls {
                margin-top: 15px;
                display: flex;
                gap: 10px;
            }
            button {
                padding: 10px 20px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                transition: all 0.2s;
            }
            button:hover { transform: translateY(-2px); }
            .btn-snapshot {
                background: #4ade80;
                color: #1a1a2e;
            }
            .btn-fullscreen {
                background: #60a5fa;
                color: white;
            }
        </style>
    </head>
    <body>
        <h1>ðŸŽ¥ Raspberry Pi USB Camera Stream</h1>
        <div class="container">
            <img id="stream" src="/video_feed" alt="Camera Stream">
            <div class="controls">
                <button class="btn-snapshot" onclick="takeSnapshot()">ðŸ“· Snapshot</button>
                <button class="btn-fullscreen" onclick="toggleFullscreen()">â›¶ Fullscreen</button>
            </div>
        </div>
        <div class="info">
            <p><strong>Endpoints:</strong></p>
            <p><span class="endpoint">GET /video_feed</span> â†’ MJPEG stream (for browsers/VLC)</p>
            <p><span class="endpoint">GET /snapshot</span> â†’ Single JPEG image</p>
            <p><span class="endpoint">GET /api/frame</span> â†’ Raw frame for API</p>
            <p><span class="endpoint">GET /api/status</span> â†’ Camera status</p>
        </div>
        <script>
            function takeSnapshot() {
                window.open('/snapshot', '_blank');
            }
            function toggleFullscreen() {
                const img = document.getElementById('stream');
                if (img.requestFullscreen) {
                    img.requestFullscreen();
                }
            }
        </script>
    </body>
    </html>
    """
    return html


@app.get("/video_feed")
async def video_feed():
    """
    MJPEG video stream.
    View in browser or use with VLC: vlc http://PI_IP:8080/video_feed
    """
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/snapshot")
async def snapshot():
    """Get single JPEG snapshot."""
    jpeg = camera_stream.get_jpeg()
    if jpeg:
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=snapshot.jpg"}
        )
    return Response(status_code=503, content="Camera not ready")


@app.get("/api/frame")
async def get_frame():
    """
    Get raw JPEG frame for API consumption.
    Use this endpoint from the vision server.
    """
    jpeg = camera_stream.get_jpeg()
    if jpeg:
        return Response(content=jpeg, media_type="image/jpeg")
    return Response(status_code=503, content="Camera not ready")


@app.get("/api/status")
async def get_status():
    """Get camera status."""
    return {
        "status": "streaming" if camera_stream and camera_stream.running else "stopped",
        "resolution": f"{camera_stream.config.width}x{camera_stream.config.height}",
        "fps_target": camera_stream.config.fps,
        "fps_actual": round(camera_stream.fps_actual, 1),
        "frame_count": camera_stream.frame_count,
        "quality": camera_stream.config.quality
    }


# === Main ===

def main():
    global camera_stream
    
    parser = argparse.ArgumentParser(description="Raspberry Pi Camera Stream Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (1-100)")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0)")
    
    args = parser.parse_args()
    
    config = StreamConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality,
        port=args.port,
        camera_index=args.camera
    )
    
    print("\n" + "=" * 50)
    print("  RASPBERRY PI USB CAMERA STREAM SERVER")
    print("=" * 50)
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  Target FPS: {config.fps}")
    print(f"  Quality: {config.quality}")
    print(f"  Camera: USB Webcam (index {config.camera_index})")
    print("=" * 50 + "\n")
    
    # Initialize camera
    camera_stream = CameraStream(config)
    
    print(f"\nðŸŒ Stream available at:")
    print(f"   http://YOUR_PI_IP:{config.port}/")
    print(f"   http://YOUR_PI_IP:{config.port}/video_feed")
    print()
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.port)
    finally:
        if camera_stream:
            camera_stream.stop()


if __name__ == "__main__":
    main()
