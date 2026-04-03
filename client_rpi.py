"""
Updated
Robot Vision Client - Raspberry Pi
====================================
Captures images, sends to FastAPI server, receives commands, controls motors.

Modes:
1. Real-time Tracking - Continuous single-image detection with live stream
2. Multi-view Finding - 3-view scan with servo control
3. Manual Control - Poll for commands from web UI

Features:
- Integrated stream server with object detection visualization
- Live video feed showing bounding boxes, labels, and tracking info
- Real-time detection results overlaid on video stream

Run: python client_rpi.py --server http://SERVER_IP:8000
"""

import io
import time
import argparse
import requests
import serial
import threading
import socket
import subprocess
import sys
import os
import glob
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# === Configuration ===

@dataclass
class Config:
    server_url: str = "http://localhost:8000"
    serial_port: str = "/dev/ttyACM0"  # Arduino Uno R3 uses ACM, not USB
    baudrate: int = 9600
    camera_width: int = 640
    camera_height: int = 480
    target_class: str = "person"
    confidence: float = 0.4
    mode: str = "realtime"  # "realtime" or "multiview"
    servo_settle_time: float = 0.5
    movement_duration: float = 1.5
    camera_index: int = 0  # USB camera index (0, 1, 2...)
    stream_port: int = 8080  # Port for local stream server
    use_stream: bool = True  # Use video stream for realtime mode


# === Motor Controller ===

class MotorController:
    """Controls motors via Arduino serial."""
    
    COMMANDS = {
        "far_left": "MFL",
        "left": "ML",
        "center": "MC",
        "right": "MR",
        "far_right": "MFR",
        "stop": "ST",
        "rotate_180": "RB",
        "servo_left": "SL",
        "servo_center": "SC",
        "servo_right": "SR",
    }
    
    def _find_arduino_port(self, preferred_port: str = None):
        """Try to find Arduino port automatically."""
        import glob
        
        # Common Arduino port patterns
        possible_ports = []
        
        # If preferred port is specified, try it first
        if preferred_port:
            possible_ports.append(preferred_port)
        
        # Try ACM ports first (Arduino Uno R3 uses CDC ACM)
        possible_ports.extend(sorted(glob.glob('/dev/ttyACM*')))
        # Then try USB serial ports
        possible_ports.extend(sorted(glob.glob('/dev/ttyUSB*')))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_ports = []
        for port in possible_ports:
            if port not in seen:
                seen.add(port)
                unique_ports.append(port)
        
        return unique_ports
    
    def __init__(self, port: str, baudrate: int):
        self.ser = None
        self.connected = False
        self.rotation_ack = threading.Event()
        
        # Try to find Arduino port if default doesn't work
        ports_to_try = self._find_arduino_port(port)
        
        print(f"ðŸ”Œ Attempting to connect to Arduino via USB...")
        print(f"   Trying ports: {', '.join(ports_to_try)}")
        print(f"   Baudrate: {baudrate}")
        
        last_error = None
        for attempt_port in ports_to_try:
            try:
                print(f"   Trying {attempt_port}...")
                self.ser = serial.Serial(port=attempt_port, baudrate=baudrate, timeout=1)
                time.sleep(2)  # Wait for Arduino to initialize
                
                # Test connection by checking if port is open
                if self.ser.is_open:
                    print(f"âœ“ USB Serial port opened successfully on {attempt_port}")
                    # Try to flush any existing data
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    print(f"âœ“ Input/output buffers cleared")
                    
                    self.connected = True
                    print(f"âœ“ Arduino connected via USB on {attempt_port} (baudrate: {baudrate})")
                    print(f"   Waiting for Arduino messages...")
                    
                    self._listener = threading.Thread(target=self._listen, daemon=True)
                    self._listener.start()
                    print(f"âœ“ USB listener thread started")
                    return  # Successfully connected, exit
                else:
                    self.ser.close()
                    raise Exception("Serial port opened but is not open")
            except serial.SerialException as e:
                print(f"   âœ— Failed: {e}")
                last_error = e
                if self.ser:
                    try:
                        self.ser.close()
                    except:
                        pass
                self.ser = None
                continue  # Try next port
            except Exception as e:
                print(f"   âœ— Failed: {e}")
                last_error = e
                if self.ser:
                    try:
                        self.ser.close()
                    except:
                        pass
                self.ser = None
                continue  # Try next port
        
        # If we get here, all ports failed
        print(f"âš  Failed to connect to Arduino on any port")
        print(f"   Last error: {last_error}")
        print(f"   Check:")
        print(f"   1. Arduino is connected via USB")
        print(f"   2. Arduino is powered on")
        print(f"   3. Check available ports: ls -l /dev/ttyACM* /dev/ttyUSB*")
        print(f"   4. No other program is using the port (try: lsof | grep tty)")
        print(f"   5. User has permissions (may need: sudo usermod -a -G dialout $USER)")
        print(f"   6. Try specifying port manually: --port /dev/ttyACM0")
        print("  Running in SIMULATION mode (no motors)")
    
    def _listen(self):
        """Listen for Arduino messages via USB serial."""
        print(f"ðŸ“¡ USB listener started - waiting for Arduino messages...")
        while True:
            if self.ser and self.ser.is_open:
                try:
                    if self.ser.in_waiting > 0:
                        line = self.ser.readline().decode('utf-8', errors='replace').strip()
                        if line:
                            print(f"ðŸ“¥ [USB RX] {line}")
                            if "ROTATION_ACK" in line:
                                self.rotation_ack.set()
                                print(f"   âœ“ ROTATION_ACK received - servo movement complete")
                    else:
                        # No data available, just wait
                        pass
                except UnicodeDecodeError as e:
                    print(f"âš  [USB RX Decode Error] {e} - received invalid data")
                except Exception as e:
                    print(f"âš  [USB RX Error] {e}")
            else:
                # Serial port not available
                if self.ser is None:
                    break
            time.sleep(0.05)
    
    def send(self, cmd: str):
        """Send raw command via USB serial."""
        print(f"ðŸ“¤ [USB TX] {cmd}")
        if self.ser and self.ser.is_open:
            try:
                cmd_bytes = (cmd + "\n").encode('utf-8')
                bytes_written = self.ser.write(cmd_bytes)
                self.ser.flush()
                print(f"   âœ“ Sent {bytes_written} bytes")
            except serial.SerialTimeoutException:
                print(f"âš  [USB TX Timeout] Command '{cmd}' timed out")
            except serial.SerialException as e:
                print(f"âš  [USB TX Serial Error] {e}")
            except Exception as e:
                print(f"âš  [USB TX Error] {e}")
        else:
            print(f"âš  [USB TX] Cannot send - serial port not open (SIMULATION MODE)")
    
    def move(self, direction: str, duration: float = 0):
        """Move in direction."""
        cmd = self.COMMANDS.get(direction, "ST")
        self.send(cmd)
        if duration > 0:
            time.sleep(duration)
            self.send("ST")
    
    def rotate_servo(self, position: str, wait_ack: bool = True, timeout: float = 3.0) -> bool:
        """Rotate servo to position."""
        cmd = self.COMMANDS.get(f"servo_{position}", "SC")
        self.send(cmd)
        
        if wait_ack and self.connected:
            if self.rotation_ack.wait(timeout):
                self.rotation_ack.clear()
                self.send("PHOTO_ACK")
                return True
            print("  âš  Rotation timeout")
            return False
        
        time.sleep(0.5)  # Fallback delay
        return True
    
    def rotate_180(self):
        """Rotate robot 180 degrees."""
        self.send("RB")
        time.sleep(3.0)
    
    def stop(self):
        self.send("ST")
    
    def close(self):
        if self.ser:
            self.ser.close()


# === Camera ===

class Camera:
    """Handles image capture."""
    
    def __init__(self, config: Config):
        self.config = config
        self.camera = None
        self._init_camera()
    
    def _list_video_devices(self):
        """List available video devices."""
        video_devices = []
        for device_path in glob.glob('/dev/video*'):
            if os.path.exists(device_path):
                video_devices.append(device_path)
        return sorted(video_devices)
    
    def _init_camera(self):
        """Initialize USB webcam."""
        print(f"Initializing USB Webcam (index {self.config.camera_index})...")
        
        # First, check available video devices
        video_devices = self._list_video_devices()
        if video_devices:
            print(f"Found video devices: {', '.join(video_devices)}")
        else:
            print("âš  No /dev/video* devices found!")
            print("  Troubleshooting:")
            print("    1. Check camera is connected: lsusb")
            print("    2. Check permissions: ls -l /dev/video*")
            print("    3. Add user to video group: sudo usermod -a -G video $USER")
            print("    4. Reboot or logout/login after adding to group")
        
        camera_found = False
        
        # Method 1: Try by device path first (more reliable)
        if video_devices:
            print("\nTrying to open by device path...")
            for device_path in video_devices:
                try:
                    # Try opening with device path
                    self.camera = cv2.VideoCapture(device_path)
                    if self.camera.isOpened():
                        ret, test_frame = self.camera.read()
                        if ret and test_frame is not None:
                            camera_found = True
                            print(f"âœ“ Camera opened successfully: {device_path}")
                            break
                        else:
                            self.camera.release()
                            print(f"  {device_path} opened but cannot read frames")
                    else:
                        self.camera.release()
                except Exception as e:
                    print(f"  Error opening {device_path}: {e}")
        
        # Method 2: Try by index if device path failed
        if not camera_found:
            print("\nTrying to open by index...")
            for idx in range(self.config.camera_index, self.config.camera_index + 5):
                try:
                    self.camera = cv2.VideoCapture(idx)
                    if self.camera.isOpened():
                        ret, test_frame = self.camera.read()
                        if ret and test_frame is not None:
                            camera_found = True
                            self.config.camera_index = idx
                            print(f"âœ“ USB Webcam found and working (index {idx})")
                            break
                        else:
                            self.camera.release()
                            print(f"  Camera index {idx} opened but cannot read frames")
                    else:
                        if idx == self.config.camera_index:
                            print(f"  Camera index {idx} not available")
                except Exception as e:
                    print(f"  Error with index {idx}: {e}")
        
        if not camera_found:
            error_msg = "\n" + "=" * 60
            error_msg += "\nâŒ CAMERA INITIALIZATION FAILED"
            error_msg += "\n" + "=" * 60
            error_msg += "\n\nTroubleshooting Steps:"
            error_msg += "\n1. Check camera connection:"
            error_msg += "\n   lsusb | grep -i camera"
            error_msg += "\n   lsusb | grep -i video"
            error_msg += "\n\n2. Check video devices:"
            error_msg += "\n   ls -l /dev/video*"
            error_msg += "\n\n3. Fix permissions (if needed):"
            error_msg += "\n   sudo usermod -a -G video $USER"
            error_msg += "\n   # Then logout and login again"
            error_msg += "\n\n4. Test camera manually:"
            error_msg += "\n   v4l2-ctl --list-devices"
            error_msg += "\n   v4l2-ctl --device=/dev/video0 --all"
            error_msg += "\n\n5. Try different camera:"
            error_msg += "\n   python client_rpi.py --camera 1"
            error_msg += "\n\n6. Check if another process is using camera:"
            error_msg += "\n   lsof | grep video"
            error_msg += "\n" + "=" * 60
            raise RuntimeError(error_msg)
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify settings
        actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Resolution: {actual_width}x{actual_height}")
        time.sleep(2)  # Camera warm-up
        
        # Test capture
        ret, frame = self.camera.read()
        if not ret or frame is None:
            raise RuntimeError("Camera initialized but cannot capture frames")
        print(f"âœ“ Camera test capture successful")
    
    def capture(self) -> Optional[np.ndarray]:
        """Capture a frame from USB webcam."""
        if self.camera is None or not self.camera.isOpened():
            print("âš  Camera not initialized or closed")
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            print("âš  Failed to read frame from camera")
            return None
        
        if frame is None or frame.size == 0:
            print("âš  Camera returned empty frame")
            return None
        
        return frame
    
    def capture_jpeg(self) -> Optional[bytes]:
        """Capture frame as JPEG bytes."""
        frame = self.capture()
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()
    
    def close(self):
        """Release USB webcam."""
        self.camera.release()


# === API Client ===

class VisionClient:
    """Communicates with the FastAPI server."""
    
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self._check_connection()
    
    def _check_connection(self):
        """Check server connection."""
        try:
            resp = self.session.get(f"{self.server_url}/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                print(f"âœ“ Connected to server: {data['status']}")
                print(f"  Model loaded: {data['model_loaded']}")
            else:
                print(f"âš  Server returned status {resp.status_code}")
        except Exception as e:
            print(f"âš  Cannot connect to server: {e}")
    
    def detect_realtime(self, image_bytes: bytes, target_class: str = "person",
                        confidence: float = 0.4, track: bool = True) -> Dict[str, Any]:
        """Send single image for real-time detection."""
        try:
            files = {"image": ("frame.jpg", image_bytes, "image/jpeg")}
            data = {
                "target_class": target_class,
                "confidence": confidence,
                "track": track
            }
            
            resp = self.session.post(
                f"{self.server_url}/detect/realtime",
                files=files,
                data=data,
                timeout=10
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": resp.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def detect_multiview(self, images: Dict[str, bytes], target_class: str = "person",
                         confidence: float = 0.4) -> Dict[str, Any]:
        """Send 3 images for multi-view detection."""
        try:
            files = {
                "image_left": ("left.jpg", images["left"], "image/jpeg"),
                "image_center": ("center.jpg", images["center"], "image/jpeg"),
                "image_right": ("right.jpg", images["right"], "image/jpeg"),
            }
            data = {
                "target_class": target_class,
                "confidence": confidence
            }
            
            resp = self.session.post(
                f"{self.server_url}/detect/multiview",
                files=files,
                data=data,
                timeout=15
            )
            
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": resp.text}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def reset_tracker(self):
        """Reset server-side tracker."""
        try:
            resp = self.session.post(f"{self.server_url}/tracker/reset", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def poll_control_command(self) -> Optional[str]:
        """Poll server for manual control command."""
        try:
            resp = self.session.get(f"{self.server_url}/control/poll", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("has_command"):
                    return data.get("command")
            return None
        except Exception:
            return None
    
    def start_stream(self, stream_url: str, target_class: str = "person", 
                     confidence: float = 0.4, interval_ms: int = 100):
        """Start server-side stream processing."""
        try:
            data = {
                "stream_url": stream_url,
                "target_class": target_class,
                "confidence": confidence,
                "interval_ms": interval_ms
            }
            resp = self.session.post(f"{self.server_url}/stream/start", json=data, timeout=5)
            return resp.status_code == 200
        except Exception as e:
            print(f"Failed to start stream: {e}")
            return False
    
    def stop_stream(self):
        """Stop server-side stream processing."""
        try:
            resp = self.session.post(f"{self.server_url}/stream/stop", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False
    
    def detect_realtime_stream(self) -> Dict[str, Any]:
        """Get real-time detection result from stream."""
        try:
            resp = self.session.get(f"{self.server_url}/detect/realtime/stream", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"success": False, "error": resp.text}
        except Exception as e:
            return {"success": False, "error": str(e)}


# === Integrated Stream Server ===

class IntegratedStreamServer:
    """Integrated stream server with detection visualization."""
    
    def __init__(self, camera: Camera, vision_client: VisionClient, config: Config):
        self.camera = camera
        self.vision_client = vision_client
        self.config = config
        self.frame: Optional[np.ndarray] = None
        self.detection_result: Optional[Dict[str, Any]] = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        self.fps_actual = 0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        
        # FastAPI app for streaming
        self.app = FastAPI(
            title="Raspberry Pi Camera Stream with Detection",
            description="Live video streaming with object detection visualization",
            version="2.0.0"
        )
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self._setup_routes()
        # Don't start capture thread yet - wait for stream processor to start
        # self._start_capture_thread() will be called after stream processor starts
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def index():
            """Web page with video player."""
            pi_ip = self._get_local_ip()
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Live Detection Stream</title>
                <style>
                    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                    body {{
                        font-family: 'Segoe UI', Tahoma, sans-serif;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        min-height: 100vh;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        padding: 20px;
                        color: #eee;
                    }}
                    h1 {{ margin-bottom: 20px; font-size: 2em; }}
                    .container {{
                        background: rgba(255,255,255,0.1);
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
                    }}
                    img {{
                        border-radius: 10px;
                        max-width: 100%;
                        height: auto;
                        border: 2px solid #667eea;
                    }}
                </style>
            </head>
            <body>
                <h1>ðŸŽ¥ Live Detection Stream</h1>
                <div class="container">
                    <img src="/video_feed" alt="Live Stream">
                </div>
            </body>
            </html>
            """
            return html
        
        def _generate_video_feed():
            """Internal generator for video feed."""
            frame_count = 0
            while True:
                try:
                    frame = self.get_annotated_frame()
                    if frame is not None:
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if buffer is not None and len(buffer) > 0:
                            frame_count += 1
                            if frame_count % 30 == 0:  # Log every 30 frames
                                print(f"ðŸ“¹ Streaming frame {frame_count}")
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        # No frame available yet, send a placeholder or wait
                        if frame_count == 0:
                            print("âš  No frame available yet, waiting for camera...")
                        time.sleep(0.1)
                except Exception as e:
                    print(f"âš  Video feed generation error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
                time.sleep(1.0 / 30)
        
        @self.app.get("/video_feed")
        async def video_feed():
            """MJPEG video stream with detection overlay."""
            print("ðŸ“¹ Video feed endpoint accessed")
            return StreamingResponse(
                _generate_video_feed(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        @self.app.get("//video_feed")
        async def video_feed_double_slash():
            """Handle double slash case - serve stream directly."""
            print("ðŸ“¹ Video feed endpoint accessed (double slash)")
            return StreamingResponse(
                _generate_video_feed(),
                media_type="multipart/x-mixed-replace; boundary=frame"
            )
        
        @self.app.get("/api/frame")
        async def get_frame():
            """Get raw frame for API."""
            frame = self.get_annotated_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                return Response(content=buffer.tobytes(), media_type="image/jpeg")
            return Response(status_code=503, content="Camera not ready")
        
        @self.app.get("/api/status")
        async def get_status():
            """Get stream status."""
            return {
                "status": "streaming" if self.running else "stopped",
                "resolution": f"{self.config.camera_width}x{self.config.camera_height}",
                "fps_actual": round(self.fps_actual, 1),
                "frame_count": self.frame_count,
                "has_frame": self.frame is not None,
                "capture_running": self.running
            }
        
        @self.app.get("/test")
        async def test():
            """Test endpoint to verify server is running."""
            return {
                "status": "ok",
                "message": "Stream server is running",
                "routes": ["/", "/video_feed", "/api/frame", "/api/status", "/test"]
            }
        
        # Handle double slash case (redirect)
        @self.app.get("//video_feed")
        async def video_feed_double_slash():
            """Redirect double slash to single slash."""
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/video_feed", status_code=301)
    
    def _draw_detections(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """Draw detection bounding boxes on frame."""
        if not detection_result.get("success") or not detection_result.get("target_found"):
            # Still draw timestamp and FPS even if no detection
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show appropriate message based on mode
            message = detection_result.get("message", "Live Feed Active")
            if "waiting for detection" in message.lower() or "live feed active" in message.lower():
                cv2.putText(frame, "LIVE FEED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Manual Control Mode", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(frame, f"Target: {self.config.target_class}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Searching...", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Draw all detections
        all_detections = detection_result.get("all_detections", [])
        for det in all_detections:
            bbox = det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                class_name = det.get("class_name", "unknown")
                confidence = det.get("confidence", 0.0)
                is_target = det.get("class_name") == self.config.target_class
                
                # Color: green for target, blue for others
                color = (0, 255, 0) if is_target else (255, 0, 0)
                thickness = 3 if is_target else 2
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Highlight target detection
        target_det = detection_result.get("target_detection")
        if target_det:
            bbox = target_det.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                # Draw thicker box for target
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 255, 255), 3)
                
                # Draw direction indicator
                direction = detection_result.get("direction", "none")
                centroid = target_det.get("centroid", [x1 + (x2-x1)//2, y1 + (y2-y1)//2])
                cx, cy = centroid
                
                # Draw direction arrow
                if direction == "left":
                    cv2.arrowedLine(frame, (cx, cy), (cx - 50, cy), (0, 255, 255), 3, tipLength=0.3)
                elif direction == "right":
                    cv2.arrowedLine(frame, (cx, cy), (cx + 50, cy), (0, 255, 255), 3, tipLength=0.3)
                elif direction == "center":
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
                
                # Draw tracking info
                tracking = detection_result.get("tracking")
                if tracking:
                    velocity = tracking.get("velocity", [0, 0])
                    info_text = f"ID:{tracking.get('object_id', 0)} Vel:({velocity[0]:.1f},{velocity[1]:.1f})"
                    cv2.putText(frame, info_text, (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw status overlay
        status_y = 30
        cv2.putText(frame, f"Target: {self.config.target_class}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        direction = detection_result.get("direction", "none")
        if detection_result.get("target_found"):
            status_text = f"Found: {direction.upper()}"
            if detection_result.get("reached"):
                status_text += " - REACHED!"
            cv2.putText(frame, status_text, (10, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Searching...", (10, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw timestamp and FPS
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def _capture_loop(self):
        """Continuous capture and detection loop."""
        print("ðŸ“¹ Capture loop started - capturing frames for live feed...")
        while self.running:
            try:
                # Capture frame - always do this for live feed
                frame = self.camera.capture()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # Always store the frame for live feed (even without detection)
                with self.frame_lock:
                    self.frame = frame.copy()
                    self.frame_count += 1
                
                # Try to get detection result from server (optional - live feed works without it)
                try:
                    detection_result = self.vision_client.detect_realtime_stream()
                    if detection_result and detection_result.get("success"):
                        # Update with detection results
                        with self.frame_lock:
                            self.detection_result = detection_result
                    else:
                        # No detection available - that's fine for live feed
                        with self.frame_lock:
                            self.detection_result = {
                                "success": False,
                                "target_found": False,
                                "message": "Live feed active"
                            }
                except Exception:
                    # Stream processor not running - that's fine, just show raw feed
                    with self.frame_lock:
                        self.detection_result = {
                            "success": False,
                            "target_found": False,
                            "message": "Live feed active"
                        }
                
                # Calculate FPS
                self._fps_frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 1.0:
                    self.fps_actual = self._fps_frame_count / (now - self._last_fps_time)
                    self._fps_frame_count = 0
                    self._last_fps_time = now
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"âš  Capture loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
    
    def _start_capture_thread(self):
        """Start background capture thread."""
        if self.running:
            print("âš  Capture thread already running")
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        print("âœ“ Capture thread started - frames will be available for live feed")
    
    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Get frame with detection annotations."""
        with self.frame_lock:
            if self.frame is not None:
                if self.detection_result is not None:
                    annotated = self._draw_detections(self.frame.copy(), self.detection_result)
                    return annotated
                else:
                    # No detection result yet, just return raw frame with basic overlay
                    frame = self.frame.copy()
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, (10, frame.shape[0] - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {self.fps_actual:.1f}", (10, frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, "LIVE FEED", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return frame
        return None
    
    def stop(self):
        """Stop stream server."""
        self.running = False
        time.sleep(0.2)
        print("Stream server stopped")
    
    def run_server(self, port: int):
        """Run FastAPI server."""
        pi_ip = self._get_local_ip()
        print(f"\nðŸŒ Stream server running at:")
        print(f"   http://{pi_ip}:{port}/")
        print(f"   http://{pi_ip}:{port}/video_feed")
        print(f"   http://{pi_ip}:{port}/test (test endpoint)")
        print(f"\nðŸ“‹ Available routes:")
        print(f"   GET  /              - Web page with video player")
        print(f"   GET  /video_feed    - MJPEG video stream")
        print(f"   GET  /api/frame     - Single frame snapshot")
        print(f"   GET  /api/status    - Stream status")
        print(f"   GET  /test          - Test endpoint")
        print()
        uvicorn.run(self.app, host="0.0.0.0", port=port, log_level="info")


# === Main Robot Controller ===

class RobotController:
    """Main robot control loop."""
    
    def __init__(self, config: Config):
        self.config = config
        print("\n" + "=" * 50)
        print("  INITIALIZING ROBOT CONTROLLER")
        print("=" * 50)
        
        # Initialize camera first
        print("\n[1/3] Initializing camera...")
        try:
            self.camera = Camera(config)
            # Test camera capture
            test_frame = self.camera.capture()
            if test_frame is not None:
                print(f"âœ“ Camera test successful - Frame shape: {test_frame.shape}")
            else:
                raise RuntimeError("Camera test capture failed")
        except Exception as e:
            print(f"âœ— Camera initialization failed: {e}")
            raise
        
        # Initialize motors
        print("\n[2/3] Initializing motor controller...")
        self.motors = MotorController(config.serial_port, config.baudrate)
        
        # Initialize API client
        print("\n[3/3] Connecting to vision server...")
        self.client = VisionClient(config.server_url)
        
        self.running = False
        self.stream_server = None
        self.stream_server_thread = None
        self.stream_url = None
        
        print("\nâœ“ Robot controller initialized successfully!")
        print("=" * 50 + "\n")
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
    
    def _start_stream_server(self):
        """Start integrated stream server in background thread."""
        if self.stream_server:
            return True
        
        try:
            print("Starting integrated stream server with detection visualization...")
            self.stream_server = IntegratedStreamServer(
                self.camera,
                self.client,
                self.config
            )
            
            # Start server in background thread
            pi_ip = self._get_local_ip()
            self.stream_url = f"http://{pi_ip}:{self.config.stream_port}"
            
            def run_server():
                try:
                    self.stream_server.run_server(self.config.stream_port)
                except Exception as e:
                    print(f"Stream server error: {e}")
            
            self.stream_server_thread = threading.Thread(target=run_server, daemon=True)
            self.stream_server_thread.start()
            
            # Wait for server to start
            print(f"Waiting for stream server at {self.stream_url}...")
            max_attempts = 10
            for attempt in range(max_attempts):
                time.sleep(1)
                try:
                    resp = requests.get(f"{self.stream_url}/api/status", timeout=2)
                    if resp.status_code == 200:
                        status = resp.json()
                        print(f"âœ“ Stream server is running!")
                        print(f"  Status: {status.get('status')}")
                        print(f"  Resolution: {status.get('resolution')}")
                        return True
                except Exception:
                    if attempt < max_attempts - 1:
                        print(f"  Attempt {attempt + 1}/{max_attempts}...")
                    continue
            
            print(f"âš  Stream server started but not responding")
            return False
            
        except Exception as e:
            print(f"âš  Failed to start stream server: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _stop_stream_server(self):
        """Stop stream server."""
        if self.stream_server:
            self.stream_server.stop()
            self.stream_server = None
            print("âœ“ Stream server stopped")
    
    def run_realtime(self):
        """Run real-time tracking mode using video stream."""
        print("\n" + "=" * 50)
        print("  REAL-TIME TRACKING MODE (Video Stream)")
        print(f"  Target: {self.config.target_class}")
        print("=" * 50 + "\n")
        
        # Start stream server if using stream mode
        if self.config.use_stream:
            print("Starting local stream server...")
            if not self._start_stream_server():
                print("âš  Stream server failed, falling back to image-based mode")
                self.config.use_stream = False
            else:
                print(f"âœ“ Stream server ready at {self.stream_url}")
        
        # Don't start server-side stream processing automatically
        # Wait for user to click "Start" in the UI
        if self.config.use_stream and self.stream_url:
            print(f"\nâ¸ Waiting for user to start operation from UI...")
            print(f"  Stream URL: {self.stream_url}")
            print(f"  Go to http://SERVER_IP:8000/ui and click 'Start' to begin tracking")
            print(f"  Target will be set from UI")
        
        if not self.config.use_stream:
            print("\nUsing image-based mode (sending individual frames)")
            print("â¸ Waiting for user to start operation from UI...")
        
        self.running = True
        last_command_time = 0
        command_cooldown = 0.5
        auto_tracking_enabled = False  # Start disabled - wait for UI to start
        stream_processor_started = False
        
        # Search behavior when target not found
        search_state = "servo_scan"  # "servo_scan", "rotate", "move_forward"
        last_search_action_time = 0
        search_cooldown = 1.0  # Wait 1 second between search actions
        servo_positions = ["left", "center", "right"]
        current_servo_index = 1  # Start at center
        no_target_count = 0  # Count consecutive frames without target
        
        try:
            while self.running:
                # Check for manual control commands first (priority over auto-tracking)
                manual_command = self.client.poll_control_command()
                if manual_command:
                    print(f"ðŸ“± Manual command received: {manual_command}")
                    auto_tracking_enabled = False  # Disable auto-tracking when manual command received
                    
                    # Execute manual command - handle ALL UI commands
                    try:
                        if manual_command == "stop":
                            self.motors.stop()
                            print("  âœ“ Stopped")
                            # Don't auto-enable tracking - wait for UI to start again
                            
                        elif manual_command == "forward":
                            self.motors.move("center", 0.5)
                            print("  âœ“ Moving forward")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "backward":
                            # Note: backward movement - rotate 180 then move forward
                            print("  âš  Backward: rotating 180Â° then moving")
                            self.motors.rotate_180()
                            time.sleep(0.3)
                            self.motors.move("center", 0.5)
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "left":
                            self.motors.move("left", 0.5)
                            print("  âœ“ Moving left")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "right":
                            self.motors.move("right", 0.5)
                            print("  âœ“ Moving right")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "far_left":
                            self.motors.move("far_left", 0.5)
                            print("  âœ“ Moving far left")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "far_right":
                            self.motors.move("far_right", 0.5)
                            print("  âœ“ Moving far right")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "rotate_180":
                            self.motors.rotate_180()
                            print("  âœ“ Rotated 180Â°")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "servo_left":
                            self.motors.rotate_servo("left")
                            print("  âœ“ Servo moved left")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "servo_center":
                            self.motors.rotate_servo("center")
                            print("  âœ“ Servo centered")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        elif manual_command == "servo_right":
                            self.motors.rotate_servo("right")
                            print("  âœ“ Servo moved right")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                        else:
                            print(f"  âš  Unknown manual command: {manual_command}")
                            # Don't auto-enable - tracking controlled by UI start/stop
                            
                    except Exception as e:
                        print(f"  âœ— Error executing command '{manual_command}': {e}")
                        # Don't auto-enable - tracking controlled by UI start/stop
                    
                    # Skip auto-tracking this cycle
                    time.sleep(0.1)
                    continue
                
                # Check if stream processor is running on server (started from UI)
                if self.config.use_stream:
                    # Check stream status to see if it's running
                    try:
                        resp = self.client.session.get(f"{self.client.server_url}/stream/status", timeout=2)
                        if resp.status_code == 200:
                            status_data = resp.json()
                            if status_data.get("active") and not stream_processor_started:
                                stream_processor_started = True
                                auto_tracking_enabled = True
                                print("âœ“ Stream processor started from UI - beginning auto-tracking")
                                # Start capture thread for visualization
                                if self.stream_server:
                                    self.stream_server._start_capture_thread()
                            elif not status_data.get("active") and stream_processor_started:
                                stream_processor_started = False
                                auto_tracking_enabled = False
                                print("â¸ Stream processor stopped from UI - pausing auto-tracking")
                    except Exception:
                        pass  # Server might not be accessible yet
                
                # Auto-tracking mode (only if no manual command and enabled)
                if not auto_tracking_enabled:
                    time.sleep(0.5)  # Check less frequently when waiting
                    continue
                
                if self.config.use_stream:
                    # Get result from stream
                    result = self.client.detect_realtime_stream()
                    if not result.get("success"):
                        error_msg = result.get('error', 'Unknown error')
                        if "Stream processor not running" in str(error_msg):
                            # Stream processor not started yet - wait
                            if not stream_processor_started:
                                time.sleep(1)
                            else:
                                print("âš  Stream processor stopped")
                                stream_processor_started = False
                                auto_tracking_enabled = False
                            continue
                        else:
                            print(f"âš  Stream detection error: {error_msg}")
                        time.sleep(0.5)
                        continue
                else:
                    # Fallback: capture and send image
                    image_bytes = self.camera.capture_jpeg()
                    if image_bytes is None:
                        print("âš  Failed to capture image from camera")
                        time.sleep(0.5)
                        continue
                    
                    result = self.client.detect_realtime(
                        image_bytes,
                        target_class=self.config.target_class,
                        confidence=self.config.confidence,
                        track=True
                    )
                
                if not result.get("success"):
                    error_msg = result.get('error', 'Unknown error')
                    print(f"âš  Detection failed: {error_msg}")
                    time.sleep(0.5)
                    continue
                
                # Process result
                direction = result.get("direction", "none")
                target_found = result.get("target_found", False)
                reached = result.get("reached", False)
                
                if target_found:
                    # Reset search state when target is found
                    no_target_count = 0
                    current_servo_index = 1  # Reset to center
                    search_state = "servo_scan"
                    
                    tracking = result.get("tracking", {})
                    velocity = tracking.get("velocity", [0, 0]) if tracking else [0, 0]
                    area = result.get("distance_ratio", 0)
                    
                    print(f"ðŸŽ¯ Target: dir={direction}, area={area:.1%}, vel=({velocity[0]:.1f}, {velocity[1]:.1f})")
                    
                    if reached:
                        print("ðŸŽ‰ TARGET REACHED!")
                        self.motors.stop()
                    else:
                        # Smart movement with velocity prediction - continuously track and move
                        current_time = time.time()
                        if current_time - last_command_time > command_cooldown:
                            if direction == "left":
                                if velocity[0] > 50:  # Moving right
                                    print("  â†’ Object moving to center, waiting...")
                                else:
                                    print("  â†’ Moving left to track target")
                                    self.motors.move("left", self.config.movement_duration)
                            elif direction == "right":
                                if velocity[0] < -50:  # Moving left
                                    print("  â†’ Object moving to center, waiting...")
                                else:
                                    print("  â†’ Moving right to track target")
                                    self.motors.move("right", self.config.movement_duration)
                            else:  # center
                                print("  â†’ Moving forward to track target")
                                self.motors.move("center", self.config.movement_duration)
                            
                            last_command_time = current_time
                else:
                    # Target not found - implement active search behavior
                    no_target_count += 1
                    current_time = time.time()
                    
                    # Active search pattern: servo scan â†’ rotate robot â†’ move forward
                    if current_time - last_search_action_time > search_cooldown:
                        if search_state == "servo_scan":
                            # Rotate servo to scan different angles
                            servo_pos = servo_positions[current_servo_index]
                            print(f"ðŸ” Searching: Rotating servo to {servo_pos}")
                            self.motors.rotate_servo(servo_pos)
                            
                            # Move to next servo position
                            current_servo_index = (current_servo_index + 1) % len(servo_positions)
                            
                            # After scanning all positions, try rotating robot
                            if current_servo_index == 1:  # Back to center, completed one full scan
                                if no_target_count > 15:  # After ~15 seconds of no target
                                    search_state = "rotate"
                                    print("  â†’ Servo scan complete, rotating robot to search new area")
                                else:
                                    print("  â†’ Continuing servo scan...")
                            
                            last_search_action_time = current_time
                            
                        elif search_state == "rotate":
                            # Rotate robot 90 degrees to search new area
                            print("ðŸ” Searching: Rotating robot 90Â°")
                            self.motors.move("left", 0.8)  # Turn left for ~0.8 seconds = ~90 degrees
                            search_state = "move_forward"
                            last_search_action_time = current_time
                            
                        elif search_state == "move_forward":
                            # Move forward a bit to check new area
                            print("ðŸ” Searching: Moving forward to explore")
                            self.motors.move("center", 0.5)
                            search_state = "servo_scan"  # Reset to servo scan
                            current_servo_index = 1  # Reset servo to center
                            last_search_action_time = current_time
                    else:
                        # Still in cooldown, just print status
                        if no_target_count % 10 == 0:  # Print every 10 frames (~1 second)
                            print(f"ðŸ” Target not found - searching... (count: {no_target_count})")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            if self.config.use_stream:
                self.client.stop_stream()
                self._stop_stream_server()
            self.motors.stop()
    
    def run_multiview(self):
        """Run multi-view object finding mode."""
        print("\n" + "=" * 50)
        print("  MULTI-VIEW OBJECT FINDING MODE")
        print(f"  Target: {self.config.target_class}")
        print("  5-Direction Navigation")
        print("=" * 50 + "\n")
        
        # Start stream server for live feed (always available)
        if self.config.use_stream:
            print("Starting live feed stream server...")
            if not self._start_stream_server():
                print("âš  Stream server failed, live feed will not be available")
                self.config.use_stream = False
            else:
                print(f"âœ“ Live feed available at {self.stream_url}")
                # Start capture thread immediately for live feed
                if self.stream_server:
                    self.stream_server._start_capture_thread()
        
        self.running = True
        
        try:
            while self.running:
                # === FRONT SCAN ===
                print("\n[FRONT SCAN]")
                images = self._capture_3_views()
                
                if not images:
                    print("Failed to capture images")
                    continue
                
                # Send to server
                result = self.client.detect_multiview(
                    images,
                    target_class=self.config.target_class,
                    confidence=self.config.confidence
                )
                
                if not result.get("success"):
                    print(f"Detection failed: {result.get('error')}")
                    continue
                
                # Process result
                direction = result.get("direction", "none")
                target_found = result.get("target_found", False)
                reached = result.get("reached", False)
                prominence = result.get("prominence", {})
                
                print(f"  Prominence: L={prominence.get('left', 0):.2f}, "
                      f"C={prominence.get('center', 0):.2f}, R={prominence.get('right', 0):.2f}")
                print(f"  Result: {result.get('message')}")
                
                if target_found:
                    if reached:
                        print("\nðŸŽ‰ TARGET REACHED!")
                        self.motors.stop()
                        time.sleep(2)
                    else:
                        # Navigate
                        print(f"  Moving: {direction}")
                        self.motors.move(direction, self.config.movement_duration)
                else:
                    # === BACK SCAN ===
                    print("\n[ROTATING 180Â° - BACK SCAN]")
                    self.motors.rotate_180()
                    
                    images = self._capture_3_views()
                    if not images:
                        continue
                    
                    result = self.client.detect_multiview(
                        images,
                        target_class=self.config.target_class,
                        confidence=self.config.confidence
                    )
                    
                    if result.get("target_found"):
                        direction = result.get("direction", "center")
                        print(f"  Target found in back! Direction: {direction}")
                        self.motors.move(direction, self.config.movement_duration)
                    else:
                        print("  Target not found - searching...")
                        self.motors.move("center", 1.0)
                    
                    # Rotate back
                    self.motors.rotate_180()
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.motors.stop()
    
    def _capture_3_views(self) -> Optional[Dict[str, bytes]]:
        """Capture left, center, right views."""
        images = {}
        
        for view, servo_pos in [("center", "center"), ("left", "left"), ("right", "right")]:
            print(f"  Scanning {view}...")
            self.motors.rotate_servo(servo_pos)
            time.sleep(self.config.servo_settle_time)
            
            img_bytes = self.camera.capture_jpeg()
            if img_bytes is None:
                print(f"  Failed to capture {view}")
                return None
            
            images[view] = img_bytes
        
        # Return to center
        self.motors.rotate_servo("center")
        return images
    
    def run_search_and_track(self):
        """
        Hybrid mode: Multi-view search with automatic transition to tracking.
        
        Phase 1 (SEARCH): Captures 3 images (left, center, right), determines
                         5-direction navigation with obstacle avoidance.
        Phase 2 (TRACK): When target is found and centered, switches to 
                         real-time continuous tracking.
        
        Option B: Immediately starts tracking when target found in any view.
        
        IMPORTANT: Search does NOT start automatically. User must click 'Start' in UI.
        """
        print("\n" + "=" * 50)
        print("  SEARCH AND TRACK MODE")
        print(f"  Target: {self.config.target_class}")
        print("  5-Direction Navigation + Obstacle Avoidance")
        print("  Auto-transition to Real-time Tracking")
        print("=" * 50 + "\n")
        
        # Start stream server for live feed
        if self.config.use_stream:
            print("Starting live feed stream server...")
            if not self._start_stream_server():
                print("âš  Stream server failed, live feed will not be available")
                self.config.use_stream = False
            else:
                print(f"âœ“ Live feed available at {self.stream_url}")
                if self.stream_server:
                    self.stream_server._start_capture_thread()
        
        self.running = True
        mode = "idle"  # "idle" (waiting for start), "search", "track", or "manual"
        last_command_time = 0
        command_cooldown = 0.5
        no_target_count = 0
        auto_operation_enabled = False  # Controlled by UI start/stop
        
        print("\nâ¸ Waiting for user to start operation from UI...")
        print(f"  Go to http://SERVER_IP:8000/ui and click 'Start' to begin")
        print(f"  Or use manual controls while waiting")
        
        try:
            while self.running:
                # Check for manual control commands first (ALWAYS priority over auto)
                manual_command = self.client.poll_control_command()
                if manual_command:
                    print(f"ðŸ“± Manual command received: {manual_command}")
                    
                    # Special handling for mode control commands
                    if manual_command == "start_operation":
                        print("\nâœ“ Starting auto operation from UI")
                        auto_operation_enabled = True
                        mode = "search"
                        no_target_count = 0
                        time.sleep(0.1)
                        continue
                    elif manual_command == "stop_operation":
                        print("\nâ¸ Stopping auto operation from UI")
                        auto_operation_enabled = False
                        mode = "idle"
                        self.motors.stop()
                        time.sleep(0.1)
                        continue
                    elif manual_command == "enter_manual_mode":
                        print("\nðŸŽ® Entering MANUAL MODE - only manual commands will be processed")
                        auto_operation_enabled = False
                        mode = "manual"
                        self.motors.stop()
                        time.sleep(0.1)
                        continue
                    
                    # For movement/servo commands - execute them
                    # If auto operation is running, stop it first
                    if auto_operation_enabled:
                        print("  â†’ Stopping auto operation for manual control")
                        auto_operation_enabled = False
                        mode = "manual"
                        self.motors.stop()  # Stop any ongoing movement
                    
                    # Execute the manual command
                    self._execute_manual_command(manual_command)
                    time.sleep(0.1)
                    continue
                
                # Check if stream processor is running on server (started from UI)
                if not auto_operation_enabled:
                    try:
                        resp = self.client.session.get(f"{self.client.server_url}/stream/status", timeout=2)
                        if resp.status_code == 200:
                            status_data = resp.json()
                            if status_data.get("active"):
                                print("\nâœ“ Stream processor started from UI - beginning auto-tracking")
                                auto_operation_enabled = True
                                mode = "search"
                    except Exception:
                        pass  # Server might not be accessible yet
                
                # If not in auto operation mode, just wait for commands
                if not auto_operation_enabled or mode == "idle" or mode == "manual":
                    time.sleep(0.5)  # Check less frequently when waiting
                    continue
                
                if mode == "search":
                    # === SEARCH MODE ===
                    print(f"\n[SEARCH MODE] Scanning for {self.config.target_class}...")
                    
                    # Capture 3 views
                    images = self._capture_3_views()
                    if not images:
                        print("Failed to capture images")
                        time.sleep(1)
                        continue
                    
                    # Send to server for multi-view detection
                    result = self.client.detect_multiview(
                        images,
                        target_class=self.config.target_class,
                        confidence=self.config.confidence
                    )
                    
                    if not result.get("success"):
                        print(f"Detection failed: {result.get('error')}")
                        time.sleep(1)
                        continue
                    
                    # Process result
                    direction = result.get("direction", "none")
                    target_found = result.get("target_found", False)
                    reached = result.get("reached", False)
                    prominence = result.get("prominence", {})
                    obstacle_warning = result.get("obstacle_warning", False)
                    obstacle_area = result.get("obstacle_area_ratio", 0)
                    should_track = result.get("should_track", False)
                    
                    print(f"  Prominence: L={prominence.get('left', 0):.2f}, "
                          f"C={prominence.get('center', 0):.2f}, R={prominence.get('right', 0):.2f}")
                    print(f"  Direction: {direction} | Obstacle: {obstacle_warning} ({obstacle_area:.1%})")
                    print(f"  Message: {result.get('message')}")
                    
                    if target_found:
                        no_target_count = 0
                        
                        if reached:
                            print("\nðŸŽ‰ TARGET REACHED!")
                            self.motors.stop()
                            time.sleep(2)
                        elif should_track:
                            # Option B: Immediately switch to tracking when ready
                            print("\nðŸ”„ Switching to TRACKING MODE...")
                            mode = "track"
                            # Reset tracker on server
                            self.client.reset_tracker()
                            continue
                        else:
                            # Navigate based on 5-direction with obstacle avoidance
                            if obstacle_warning and obstacle_area > 0.25:
                                print(f"  âš  LARGE OBSTACLE - adjusted direction to: {direction}")
                            
                            print(f"  Moving: {direction}")
                            self.motors.move(direction, self.config.movement_duration)
                    else:
                        no_target_count += 1
                        print(f"  Target not found (count: {no_target_count})")
                        
                        if no_target_count > 3:
                            # Rotate 180 to search behind
                            print("\n[ROTATING 180Â° - BACK SCAN]")
                            self.motors.rotate_180()
                            no_target_count = 0
                        else:
                            # Move forward a bit to explore
                            self.motors.move("center", 0.5)
                    
                    time.sleep(0.3)
                
                else:
                    # === TRACK MODE ===
                    # Use existing realtime tracking logic
                    if self.config.use_stream:
                        result = self.client.detect_realtime_stream()
                    else:
                        image_bytes = self.camera.capture_jpeg()
                        if image_bytes is None:
                            print("âš  Failed to capture image")
                            time.sleep(0.5)
                            continue
                        result = self.client.detect_realtime(
                            image_bytes,
                            target_class=self.config.target_class,
                            confidence=self.config.confidence,
                            track=True
                        )
                    
                    if not result.get("success"):
                        error_msg = result.get('error', 'Unknown error')
                        if "Stream processor not running" in str(error_msg):
                            # Start stream processor
                            if self.stream_url:
                                from functools import partial
                                stream_config = {
                                    "stream_url": self.stream_url,
                                    "target_class": self.config.target_class,
                                    "confidence": self.config.confidence,
                                    "interval_ms": 100
                                }
                                try:
                                    self.client.session.post(
                                        f"{self.client.server_url}/stream/start",
                                        json=stream_config,
                                        timeout=5
                                    )
                                    print("âœ“ Stream processor started for tracking")
                                except Exception as e:
                                    print(f"âš  Failed to start stream: {e}")
                            time.sleep(0.5)
                            continue
                        print(f"âš  Detection error: {error_msg}")
                        time.sleep(0.5)
                        continue
                    
                    # Process tracking result
                    direction = result.get("direction", "none")
                    target_found = result.get("target_found", False)
                    reached = result.get("reached", False)
                    
                    if target_found:
                        no_target_count = 0
                        tracking = result.get("tracking", {})
                        velocity = tracking.get("velocity", [0, 0]) if tracking else [0, 0]
                        area = result.get("distance_ratio", 0)
                        
                        print(f"ðŸŽ¯ TRACKING: dir={direction}, area={area:.1%}, vel=({velocity[0]:.1f}, {velocity[1]:.1f})")
                        
                        if reached:
                            print("ðŸŽ‰ TARGET REACHED!")
                            self.motors.stop()
                        else:
                            # Smart movement with velocity prediction
                            current_time = time.time()
                            if current_time - last_command_time > command_cooldown:
                                if direction == "left":
                                    if velocity[0] > 50:  # Moving right
                                        print("  â†’ Object moving to center, waiting...")
                                    else:
                                        print("  â†’ Moving left to track")
                                        self.motors.move("left", self.config.movement_duration)
                                elif direction == "right":
                                    if velocity[0] < -50:  # Moving left
                                        print("  â†’ Object moving to center, waiting...")
                                    else:
                                        print("  â†’ Moving right to track")
                                        self.motors.move("right", self.config.movement_duration)
                                else:  # center
                                    print("  â†’ Moving forward to track")
                                    self.motors.move("center", self.config.movement_duration)
                                
                                last_command_time = current_time
                    else:
                        no_target_count += 1
                        if no_target_count > 10:
                            # Lost target - switch back to search mode
                            print("\nâš  Target lost! Switching back to SEARCH MODE...")
                            mode = "search"
                            no_target_count = 0
                            self.client.stop_stream()
                        else:
                            print(f"ðŸ” Target not found (count: {no_target_count})")
                    
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping search and track mode...")
        finally:
            if self.config.use_stream:
                self.client.stop_stream()
                self._stop_stream_server()
            self.motors.stop()
    
    def _execute_manual_command(self, command: str):
        """Execute a manual control command."""
        try:
            # Note: start_operation and stop_operation are handled in run_search_and_track loop
            if command in ["start_operation", "stop_operation"]:
                # These are handled in the main loop, not here
                return
            elif command == "stop":
                self.motors.stop()
                print("  âœ“ Stopped")
            elif command == "forward":
                self.motors.move("center", 0.5)
                print("  âœ“ Moving forward")
            elif command == "backward":
                print("  âš  Backward: rotating 180Â° then moving")
                self.motors.rotate_180()
                time.sleep(0.3)
                self.motors.move("center", 0.5)
            elif command == "left":
                self.motors.move("left", 0.5)
                print("  âœ“ Moving left")
            elif command == "right":
                self.motors.move("right", 0.5)
                print("  âœ“ Moving right")
            elif command == "far_left":
                self.motors.move("far_left", 0.5)
                print("  âœ“ Moving far left")
            elif command == "far_right":
                self.motors.move("far_right", 0.5)
                print("  âœ“ Moving far right")
            elif command == "rotate_180":
                self.motors.rotate_180()
                print("  âœ“ Rotated 180Â°")
            elif command == "servo_left":
                self.motors.rotate_servo("left")
                print("  âœ“ Servo moved left")
            elif command == "servo_center":
                self.motors.rotate_servo("center")
                print("  âœ“ Servo centered")
            elif command == "servo_right":
                self.motors.rotate_servo("right")
                print("  âœ“ Servo moved right")
            else:
                print(f"  âš  Unknown command: {command}")
        except Exception as e:
            print(f"  âœ— Error executing command '{command}': {e}")

    def run_manual_control(self):
        """Run manual control mode - poll for commands from web UI."""
        print("\n" + "=" * 50)
        print("  MANUAL CONTROL MODE")
        print("  Waiting for commands from web UI...")
        print("=" * 50 + "\n")
        
        # Start stream server for live feed (always available)
        if self.config.use_stream:
            print("Starting live feed stream server...")
            if not self._start_stream_server():
                print("âš  Stream server failed, live feed will not be available")
                self.config.use_stream = False
            else:
                print(f"âœ“ Live feed available at {self.stream_url}")
                # Start capture thread immediately for live feed (no detection needed)
                if self.stream_server:
                    self.stream_server._start_capture_thread()
        
        self.running = True
        
        try:
            while self.running:
                # Poll for manual control commands
                command = self.client.poll_control_command()
                
                if command:
                    print(f"ðŸ“± Received command: {command}")
                    
                    # Handle ALL UI commands
                    try:
                        if command == "stop":
                            self.motors.stop()
                            print("  âœ“ Stopped")
                            
                        elif command == "forward":
                            self.motors.move("center", 0.5)
                            print("  âœ“ Moving forward")
                            
                        elif command == "backward":
                            # Backward: rotate 180Â° then move forward
                            print("  âš  Backward: rotating 180Â° then moving")
                            self.motors.rotate_180()
                            time.sleep(0.3)
                            self.motors.move("center", 0.5)
                            
                        elif command == "left":
                            self.motors.move("left", 0.5)
                            print("  âœ“ Moving left")
                            
                        elif command == "right":
                            self.motors.move("right", 0.5)
                            print("  âœ“ Moving right")
                            
                        elif command == "far_left":
                            self.motors.move("far_left", 0.5)
                            print("  âœ“ Moving far left")
                            
                        elif command == "far_right":
                            self.motors.move("far_right", 0.5)
                            print("  âœ“ Moving far right")
                            
                        elif command == "rotate_180":
                            self.motors.rotate_180()
                            print("  âœ“ Rotated 180Â°")
                            
                        elif command == "servo_left":
                            self.motors.rotate_servo("left")
                            print("  âœ“ Servo moved left")
                            
                        elif command == "servo_center":
                            self.motors.rotate_servo("center")
                            print("  âœ“ Servo centered")
                            
                        elif command == "servo_right":
                            self.motors.rotate_servo("right")
                            print("  âœ“ Servo moved right")
                            
                        else:
                            print(f"  âš  Unknown command: {command}")
                            
                    except Exception as e:
                        print(f"  âœ— Error executing command '{command}': {e}")
                
                time.sleep(0.1)  # Poll every 100ms
                
        except KeyboardInterrupt:
            print("\nStopping manual control...")
        finally:
            self.motors.stop()
    
    def run(self):
        """Run selected mode."""
        try:
            if self.config.mode == "realtime":
                self.run_realtime()
            elif self.config.mode == "manual":
                self.run_manual_control()
            elif self.config.mode == "search_track":
                self.run_search_and_track()
            else:
                self.run_multiview()
        finally:
            self.close()
    
    def close(self):
        """Cleanup."""
        if self.config.use_stream:
            self.client.stop_stream()
            self._stop_stream_server()
        self.motors.stop()
        self.camera.close()
        self.motors.close()
        print("Cleanup complete")


# === Entry Point ===

def main():
    parser = argparse.ArgumentParser(description="Robot Vision Client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Arduino serial port")
    parser.add_argument("--target", default="person", help="Target object class")
    parser.add_argument("--mode", choices=["realtime", "multiview", "manual", "search_track"], default="search_track",
                        help="Detection mode: realtime (continuous tracking), multiview (3-view scan), manual (remote control), search_track (hybrid search + auto-tracking)")
    parser.add_argument("--confidence", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--camera", type=int, default=0, help="USB camera index (default: 0)")
    parser.add_argument("--stream-port", type=int, default=8080, help="Stream server port (default: 8080)")
    parser.add_argument("--no-stream", action="store_true", help="Disable video stream, use image-based mode")
    
    args = parser.parse_args()
    
    config = Config(
        server_url=args.server,
        serial_port=args.port,
        target_class=args.target,
        mode=args.mode,
        confidence=args.confidence,
        camera_index=args.camera,
        stream_port=args.stream_port,
        use_stream=not args.no_stream
    )
    
    robot = RobotController(config)
    robot.run()


if __name__ == "__main__":
    main()
