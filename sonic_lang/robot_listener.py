"""
Robot Listener - FastAPI server for Raspberry Pi
This runs on the Pi and receives commands over WiFi to control the robot via Arduino serial.
Also provides camera streaming.

Run on Raspberry Pi:
    python robot_listener.py
    # or
    uvicorn robot_listener:app --host 0.0.0.0 --port 8080
"""

import os
import glob
import time
import threading
import base64
from typing import Optional
from contextlib import asynccontextmanager

import cv2
import serial
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# =========================
# CONFIGURATION
# =========================
_DEFAULT_SERIAL_BY_ID = (
    "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
)


def resolve_robot_serial_port() -> str:
    for key in ("ROBOT_SERIAL_PORT", "SERIAL_PORT"):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    if os.path.exists(_DEFAULT_SERIAL_BY_ID):
        return _DEFAULT_SERIAL_BY_ID
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return _DEFAULT_SERIAL_BY_ID


SERIAL_PORT = resolve_robot_serial_port()
BAUD_RATE = int(os.getenv("ROBOT_BAUD_RATE", "115200"))
CAMERA_INDEX = int(os.getenv("ROBOT_CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.getenv("ROBOT_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("ROBOT_CAMERA_HEIGHT", "480"))
SERVER_PORT = int(os.getenv("ROBOT_SERVER_PORT", "8080"))

# =========================
# SERIAL CONTROLLER
# =========================
class ArduinoController:
    """Handles serial communication with Arduino"""
    
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.serial: Optional[serial.Serial] = None
        self.lock = threading.Lock()
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to Arduino via serial"""
        try:
            with self.lock:
                if self.serial and self.serial.is_open:
                    self.serial.close()
                
                self.serial = serial.Serial(self.port, self.baud, timeout=1)
                time.sleep(2)  # Wait for Arduino to reset
                self.connected = True
                print(f"Connected to Arduino on {self.port}")
                return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        with self.lock:
            if self.serial and self.serial.is_open:
                self.send_command("S:0")  # Stop before disconnecting
                self.serial.close()
            self.connected = False
    
    def send_command(self, cmd: str) -> bool:
        """Send a command to Arduino"""
        try:
            with self.lock:
                if not self.serial or not self.serial.is_open:
                    if not self.connect():
                        return False
                
                self.serial.write((cmd + "\n").encode())
                print(f"Sent command: {cmd}")
                return True
        except Exception as e:
            print(f"Failed to send command: {e}")
            self.connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Arduino"""
        return self.connected and self.serial and self.serial.is_open


# =========================
# CAMERA CONTROLLER
# =========================
class CameraController:
    """Handles camera capture and streaming"""
    
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera: Optional[cv2.VideoCapture] = None
        self.lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize the camera"""
        try:
            with self.lock:
                if self.camera is not None:
                    self.camera.release()
                
                self.camera = cv2.VideoCapture(self.camera_index)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                
                if not self.camera.isOpened():
                    print("Failed to open camera")
                    return False
                    
                print(f"Camera initialized: {self.width}x{self.height}")
                return True
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def release(self):
        """Release camera resources"""
        with self.lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
    
    def capture_frame(self) -> Optional[bytes]:
        """Capture a single JPEG frame"""
        try:
            with self.lock:
                if self.camera is None or not self.camera.isOpened():
                    if not self.initialize():
                        return None
                
                ret, frame = self.camera.read()
                if not ret:
                    return None
                
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buffer.tobytes()
        except Exception as e:
            print(f"Failed to capture frame: {e}")
            return None
    
    def generate_mjpeg_stream(self):
        """Generator for MJPEG video stream"""
        while True:
            frame = self.capture_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS


# =========================
# GLOBAL INSTANCES
# =========================
arduino = ArduinoController(SERIAL_PORT, BAUD_RATE)
camera = CameraController(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)


# =========================
# LIFESPAN HANDLER
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("Starting Robot Listener...")
    arduino.connect()
    camera.initialize()
    # Center the servo on startup
    arduino.send_command("P:90")
    
    yield
    
    # Shutdown
    print("Shutting down Robot Listener...")
    arduino.send_command("S:0")  # Stop motors
    arduino.disconnect()
    camera.release()


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Robot Controller API",
    description="Control the robot via WiFi commands",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - Allow all origins for easy mobile access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODELS
# =========================
class MoveCommand(BaseModel):
    direction: str  # "forward", "backward", "left", "right", "stop"
    speed: int = 150  # 0-255

class ServoCommand(BaseModel):
    angle: int  # 0-180


# =========================
# API ENDPOINTS
# =========================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve a simple status page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Robot Controller</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }
            h1 { color: #00d9ff; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .connected { background: #2d5a27; }
            .disconnected { background: #5a2727; }
            a { color: #00d9ff; }
        </style>
    </head>
    <body>
        <h1>Robot Controller API</h1>
        <p>The robot listener is running!</p>
        <h3>Endpoints:</h3>
        <ul>
            <li><a href="/status">/status</a> - Check robot status</li>
            <li><a href="/camera/stream">/camera/stream</a> - Live camera feed</li>
            <li><a href="/docs">/docs</a> - API documentation</li>
            <li><a href="/control">/control</a> - Web control interface</li>
        </ul>
    </body>
    </html>
    """


@app.get("/status")
async def get_status():
    """Get robot connection status"""
    return {
        "arduino_connected": arduino.is_connected(),
        "camera_available": camera.camera is not None and camera.camera.isOpened(),
        "serial_port": SERIAL_PORT,
        "camera_index": CAMERA_INDEX
    }


@app.post("/robot/move")
async def move_robot(command: MoveCommand):
    """Send movement command to robot"""
    direction = command.direction.lower()
    speed = max(0, min(255, command.speed))  # Clamp speed to 0-255
    
    cmd_map = {
        "forward": f"F:{speed}",
        "backward": f"B:{speed}",
        "left": f"L:{speed}",
        "right": f"R:{speed}",
        "stop": "S:0"
    }
    
    if direction not in cmd_map:
        raise HTTPException(status_code=400, detail=f"Invalid direction: {direction}")
    
    cmd = cmd_map[direction]
    success = arduino.send_command(cmd)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send command to Arduino")
    
    return {"success": True, "command": cmd, "direction": direction, "speed": speed}


@app.post("/robot/servo")
async def set_servo(command: ServoCommand):
    """Set servo angle"""
    angle = max(0, min(180, command.angle))  # Clamp angle to 0-180
    cmd = f"P:{angle}"
    
    success = arduino.send_command(cmd)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send command to Arduino")
    
    return {"success": True, "command": cmd, "angle": angle}


@app.post("/robot/stop")
async def stop_robot():
    """Emergency stop - stop all motors"""
    success = arduino.send_command("S:0")
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send stop command")
    
    return {"success": True, "command": "S:0"}


@app.get("/camera/stream")
async def camera_stream():
    """MJPEG camera stream"""
    return StreamingResponse(
        camera.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/camera/capture")
async def camera_capture():
    """Capture single image from camera"""
    frame = camera.capture_frame()
    
    if frame is None:
        raise HTTPException(status_code=500, detail="Failed to capture image")
    
    # Return as base64 encoded image
    image_base64 = base64.b64encode(frame).decode('utf-8')
    
    return {
        "success": True,
        "image_base64": image_base64,
        "content_type": "image/jpeg"
    }


@app.post("/robot/reconnect")
async def reconnect_arduino():
    """Attempt to reconnect to Arduino"""
    success = arduino.connect()
    return {"success": success, "connected": arduino.is_connected()}


# =========================
# WEB CONTROL INTERFACE
# =========================
@app.get("/control", response_class=HTMLResponse)
async def control_page():
    """Serve the web control interface"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Robot Control</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
            touch-action: manipulation;
        }
        
        .container {
            max-width: 500px;
            margin: 0 auto;
            padding: 15px;
        }
        
        header {
            text-align: center;
            padding: 15px 0;
        }
        
        h1 {
            font-size: 1.5em;
            color: #00d9ff;
            margin-bottom: 5px;
        }
        
        .status {
            font-size: 0.85em;
            padding: 5px 12px;
            border-radius: 20px;
            display: inline-block;
        }
        
        .status.connected { background: #2d5a27; }
        .status.disconnected { background: #5a2727; }
        
        .camera-feed {
            width: 100%;
            aspect-ratio: 4/3;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            margin: 15px 0;
            position: relative;
        }
        
        .camera-feed img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .camera-feed .placeholder {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-size: 1.2em;
        }
        
        .controls {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(3, 1fr);
            gap: 10px;
            max-width: 300px;
            margin: 0 auto 20px;
        }
        
        .control-btn {
            aspect-ratio: 1;
            border: none;
            border-radius: 12px;
            font-size: 2em;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            justify-content: center;
            -webkit-tap-highlight-color: transparent;
            user-select: none;
            touch-action: none;
        }
        
        .control-btn:active {
            transform: scale(0.95);
        }
        
        .direction-btn {
            background: linear-gradient(145deg, #2d4a7c, #1e3a5f);
            color: #fff;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        }
        
        .direction-btn:active {
            background: linear-gradient(145deg, #3d6aac, #2e5a8f);
        }
        
        .stop-btn {
            background: linear-gradient(145deg, #c0392b, #962d22);
            color: #fff;
            box-shadow: 0 4px 15px rgba(192, 57, 43, 0.4);
        }
        
        .stop-btn:active {
            background: linear-gradient(145deg, #e74c3c, #c0392b);
        }
        
        .empty-cell {
            visibility: hidden;
        }
        
        .mic-section {
            text-align: center;
            margin: 20px 0;
        }
        
        .mic-btn {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(145deg, #00d9ff, #00a8cc);
            color: #1a1a2e;
            font-size: 2em;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(0, 217, 255, 0.4);
            transition: all 0.2s;
        }
        
        .mic-btn:active, .mic-btn.listening {
            transform: scale(1.1);
            box-shadow: 0 4px 30px rgba(0, 217, 255, 0.6);
        }
        
        .mic-btn.listening {
            animation: pulse 1s infinite;
            background: linear-gradient(145deg, #ff6b6b, #ee5a5a);
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 4px 20px rgba(255, 107, 107, 0.4); }
            50% { box-shadow: 0 4px 40px rgba(255, 107, 107, 0.8); }
        }
        
        .voice-text {
            margin-top: 15px;
            min-height: 24px;
            font-size: 1em;
            color: #aaa;
        }
        
        .servo-section {
            margin: 20px 0;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }
        
        .servo-section label {
            display: block;
            margin-bottom: 10px;
            color: #00d9ff;
        }
        
        .servo-slider {
            width: 100%;
            height: 40px;
            -webkit-appearance: none;
            background: #2d4a7c;
            border-radius: 20px;
            outline: none;
        }
        
        .servo-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 30px;
            height: 30px;
            background: #00d9ff;
            border-radius: 50%;
            cursor: pointer;
        }
        
        .speed-section {
            margin: 20px 0;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
        }
        
        .speed-section label {
            display: block;
            margin-bottom: 10px;
            color: #00d9ff;
        }
        
        .speed-value {
            color: #fff;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Robot Control</h1>
            <span id="status" class="status disconnected">Connecting...</span>
        </header>
        
        <div class="camera-feed">
            <img id="camera" src="/camera/stream" alt="Camera Feed" onerror="this.style.display='none'; document.getElementById('camera-placeholder').style.display='flex';">
            <div id="camera-placeholder" class="placeholder" style="display:none;">Camera unavailable</div>
        </div>
        
        <div class="controls">
            <div class="empty-cell"></div>
            <button class="control-btn direction-btn" data-direction="forward">&#9650;</button>
            <div class="empty-cell"></div>
            
            <button class="control-btn direction-btn" data-direction="left">&#9664;</button>
            <button class="control-btn stop-btn" data-direction="stop">&#9632;</button>
            <button class="control-btn direction-btn" data-direction="right">&#9654;</button>
            
            <div class="empty-cell"></div>
            <button class="control-btn direction-btn" data-direction="backward">&#9660;</button>
            <div class="empty-cell"></div>
        </div>
        
        <div class="speed-section">
            <label>Speed: <span id="speed-value" class="speed-value">150</span></label>
            <input type="range" id="speed-slider" class="servo-slider" min="50" max="255" value="150">
        </div>
        
        <div class="servo-section">
            <label>Camera Pan: <span id="servo-value">90</span>&deg;</label>
            <input type="range" id="servo-slider" class="servo-slider" min="0" max="180" value="90">
        </div>
        
        <div class="mic-section">
            <button id="mic-btn" class="mic-btn">&#127908;</button>
            <p id="voice-text" class="voice-text">Tap microphone to speak</p>
        </div>
    </div>
    
    <script>
        const API_BASE = '';
        const HOLD_REPEAT_MS = 120;
        let currentSpeed = 150;
        let isListening = false;
        let recognition = null;
        let holdRepeatTimer = null;
        let activeHeldDirection = null;
        
        // Initialize Speech Recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onresult = (event) => {
                const transcript = Array.from(event.results)
                    .map(result => result[0].transcript)
                    .join('');
                document.getElementById('voice-text').textContent = transcript;
                
                if (event.results[event.results.length - 1].isFinal) {
                    processVoiceCommand(transcript.toLowerCase());
                }
            };
            
            recognition.onend = () => {
                isListening = false;
                document.getElementById('mic-btn').classList.remove('listening');
            };
            
            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                isListening = false;
                document.getElementById('mic-btn').classList.remove('listening');
                document.getElementById('voice-text').textContent = 'Error: ' + event.error;
            };
        }
        
        // Voice command processing
        function processVoiceCommand(text) {
            console.log('Processing command:', text);
            
            if (text.includes('forward') || text.includes('go') || text.includes('ahead') || text.includes('move')) {
                sendMove('forward');
            } else if (text.includes('back') || text.includes('reverse')) {
                sendMove('backward');
            } else if (text.includes('left')) {
                if (text.includes('look')) {
                    setServo(45);
                } else {
                    sendMove('left');
                }
            } else if (text.includes('right')) {
                if (text.includes('look')) {
                    setServo(135);
                } else {
                    sendMove('right');
                }
            } else if (text.includes('stop') || text.includes('halt') || text.includes('freeze')) {
                sendMove('stop');
            } else if (text.includes('center') || text.includes('straight')) {
                setServo(90);
            } else if (text.includes('faster')) {
                currentSpeed = Math.min(255, currentSpeed + 30);
                document.getElementById('speed-slider').value = currentSpeed;
                document.getElementById('speed-value').textContent = currentSpeed;
            } else if (text.includes('slower')) {
                currentSpeed = Math.max(50, currentSpeed - 30);
                document.getElementById('speed-slider').value = currentSpeed;
                document.getElementById('speed-value').textContent = currentSpeed;
            }
        }
        
        // API functions
        async function sendMove(direction) {
            try {
                const response = await fetch(`${API_BASE}/robot/move`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ direction, speed: currentSpeed })
                });
                const data = await response.json();
                console.log('Move response:', data);
            } catch (error) {
                console.error('Move error:', error);
            }
        }

        function stopDirectionHold() {
            if (holdRepeatTimer) {
                clearInterval(holdRepeatTimer);
                holdRepeatTimer = null;
            }
            if (activeHeldDirection) {
                sendMove('stop');
                activeHeldDirection = null;
            }
        }

        function startDirectionHold(direction) {
            stopDirectionHold();
            activeHeldDirection = direction;
            sendMove(direction);
            holdRepeatTimer = setInterval(() => sendMove(direction), HOLD_REPEAT_MS);
        }
        
        async function setServo(angle) {
            try {
                const response = await fetch(`${API_BASE}/robot/servo`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ angle })
                });
                const data = await response.json();
                document.getElementById('servo-slider').value = angle;
                document.getElementById('servo-value').textContent = angle;
                console.log('Servo response:', data);
            } catch (error) {
                console.error('Servo error:', error);
            }
        }
        
        async function checkStatus() {
            try {
                const response = await fetch(`${API_BASE}/status`);
                const data = await response.json();
                const statusEl = document.getElementById('status');
                if (data.arduino_connected) {
                    statusEl.textContent = 'Connected';
                    statusEl.className = 'status connected';
                } else {
                    statusEl.textContent = 'Disconnected';
                    statusEl.className = 'status disconnected';
                }
            } catch (error) {
                document.getElementById('status').textContent = 'Offline';
                document.getElementById('status').className = 'status disconnected';
            }
        }
        
        // Direction buttons: pointer events + repeat while held
        document.querySelectorAll('.control-btn').forEach(btn => {
            const direction = btn.dataset.direction;
            let capturedPointerId = null;

            btn.addEventListener('pointerdown', (e) => {
                if (e.button !== 0) return;
                e.preventDefault();
                if (direction === 'stop') {
                    stopDirectionHold();
                    sendMove('stop');
                    return;
                }
                startDirectionHold(direction);
                try {
                    btn.setPointerCapture(e.pointerId);
                    capturedPointerId = e.pointerId;
                } catch (err) {
                    capturedPointerId = e.pointerId;
                }
            });

            const onPointerEnd = (e) => {
                if (capturedPointerId !== null && e.pointerId === capturedPointerId) {
                    try {
                        btn.releasePointerCapture(e.pointerId);
                    } catch (err) { /* ignore */ }
                    capturedPointerId = null;
                }
                if (direction !== 'stop' && activeHeldDirection === direction) {
                    stopDirectionHold();
                }
            };

            btn.addEventListener('pointerup', onPointerEnd);
            btn.addEventListener('pointercancel', onPointerEnd);
        });
        
        // Mic button
        document.getElementById('mic-btn').addEventListener('click', () => {
            if (!recognition) {
                alert('Speech recognition not supported in this browser');
                return;
            }
            
            if (isListening) {
                recognition.stop();
            } else {
                recognition.start();
                isListening = true;
                document.getElementById('mic-btn').classList.add('listening');
                document.getElementById('voice-text').textContent = 'Listening...';
            }
        });
        
        // Speed slider
        document.getElementById('speed-slider').addEventListener('input', (e) => {
            currentSpeed = parseInt(e.target.value);
            document.getElementById('speed-value').textContent = currentSpeed;
        });
        
        // Servo slider
        document.getElementById('servo-slider').addEventListener('change', (e) => {
            const angle = parseInt(e.target.value);
            document.getElementById('servo-value').textContent = angle;
            setServo(angle);
        });
        
        // Check status periodically
        checkStatus();
        setInterval(checkStatus, 5000);
    </script>
</body>
</html>
    """


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    print(f"Starting Robot Listener on port {SERVER_PORT}")
    print(f"Serial Port: {SERIAL_PORT}")
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"\nAccess the control interface at: http://<pi-ip>:{SERVER_PORT}/control")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
