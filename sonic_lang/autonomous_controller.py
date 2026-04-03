"""
Autonomous Controller - YOLO-based object detection and tracking for robot navigation.

Runs as a SEPARATE process from robot_listener.py. Talks to the robot via HTTP.

Workflow:
1. Scans 3 positions (left, center, right) with camera servo
2. Uses YOLO to detect target object in each position
3. Rotates robot to face detected object
4. Approaches object until it occupies 50% of frame

Usage (standalone - robot_listener must be running on the Pi):
    python run_autonomous.py --target bottle
    python run_autonomous.py --target cup --pi-url http://192.168.1.100:8080
"""

import os
import time
import threading
import base64
from typing import Optional, List, Dict, Any, Tuple, Callable, Protocol
from enum import Enum
from dataclasses import dataclass

import cv2
import numpy as np

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not installed. Install with: pip install ultralytics")


# =========================
# CONFIGURATION
# =========================
_SONIC_LANG_DIR = os.path.dirname(os.path.abspath(__file__))


def _default_yolo_model_path() -> str:
    env = os.getenv("YOLO_MODEL_PATH")
    if env and env.strip():
        return env.strip()
    for p in (
        os.path.join(_SONIC_LANG_DIR, "yolo26n.pt"),
        "/home/pi/Desktop/sonic/sonic_lang/yolo26n.pt",
    ):
        if os.path.isfile(p):
            return p
    return os.path.join(_SONIC_LANG_DIR, "yolo26n.pt")


class AutonomousConfig:
    """Configuration for autonomous mode"""
    
    # YOLO settings (YOLO_MODEL_PATH env, else sonic_lang/yolo26n.pt or legacy Pi path)
    YOLO_MODEL_PATH = _default_yolo_model_path()
    YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.4"))
    
    # Servo positions for scanning (0=left, 90=center, 180=right)
    SERVO_LEFT_ANGLE = int(os.getenv("SERVO_LEFT_ANGLE", "0"))
    SERVO_CENTER_ANGLE = int(os.getenv("SERVO_CENTER_ANGLE", "90"))
    SERVO_RIGHT_ANGLE = int(os.getenv("SERVO_RIGHT_ANGLE", "180"))
    SERVO_SETTLE_TIME = float(os.getenv("SERVO_SETTLE_TIME", "0.8"))
    
    # Target approach settings
    TARGET_AREA_THRESHOLD = float(os.getenv("TARGET_AREA_THRESHOLD", "0.5"))  # 50% of frame
    APPROACH_SPEED = int(os.getenv("APPROACH_SPEED", "100"))
    TURN_SPEED = int(os.getenv("TURN_SPEED", "120"))
    TURN_DURATION = float(os.getenv("TURN_DURATION", "0.3"))  # Seconds per turn step
    
    # Movement tuning
    APPROACH_STEP_DURATION = float(os.getenv("APPROACH_STEP_DURATION", "0.3"))
    CENTER_TOLERANCE = float(os.getenv("CENTER_TOLERANCE", "0.15"))  # 15% from center
    APPROACH_LOST_RESCAN_FRAMES = int(os.getenv("APPROACH_LOST_RESCAN_FRAMES", "5"))
    
    @classmethod
    def get_scan_positions(cls) -> List[Tuple[int, str]]:
        """Get servo positions with names for scanning"""
        return [
            (cls.SERVO_LEFT_ANGLE, "left"),
            (cls.SERVO_CENTER_ANGLE, "center"),
            (cls.SERVO_RIGHT_ANGLE, "right")
        ]


# =========================
# ROBOT CLIENT (for separate process - talks to robot_listener via HTTP)
# =========================
class RobotClient(Protocol):
    """Interface for robot control. Use HttpRobotClient when running autonomous separately."""

    def set_servo(self, angle: int) -> bool: ...
    def capture_frame(self) -> Optional[np.ndarray]: ...
    def move(self, direction: str, speed: int) -> bool: ...
    def stop(self) -> bool: ...


class HttpRobotClient:
    """
    Robot client that controls the robot via HTTP (robot_listener API).
    Use this when running autonomous mode as a separate process.
    """

    def __init__(self, base_url: str, timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None

    def _get_session(self):
        if self._session is None:
            try:
                import requests
                self._session = requests.Session()
            except ImportError:
                raise ImportError("requests is required for HttpRobotClient. pip install requests")
        return self._session

    def set_servo(self, angle: int) -> bool:
        try:
            r = self._get_session().post(
                f"{self.base_url}/robot/servo",
                json={"angle": angle},
                timeout=self.timeout
            )
            return r.status_code == 200 and r.json().get("success", False)
        except Exception as e:
            print(f"set_servo error: {e}")
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        try:
            r = self._get_session().get(f"{self.base_url}/camera/capture", timeout=self.timeout)
            if r.status_code != 200:
                return None
            data = r.json()
            if not data.get("success") or not data.get("image_base64"):
                return None
            buf = base64.b64decode(data["image_base64"])
            arr = np.frombuffer(buf, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            print(f"capture_frame error: {e}")
            return None

    def move(self, direction: str, speed: int) -> bool:
        try:
            r = self._get_session().post(
                f"{self.base_url}/robot/move",
                json={"direction": direction, "speed": speed},
                timeout=self.timeout
            )
            return r.status_code == 200 and r.json().get("success", False)
        except Exception as e:
            print(f"move error: {e}")
            return False

    def stop(self) -> bool:
        try:
            r = self._get_session().post(f"{self.base_url}/robot/stop", timeout=self.timeout)
            return r.status_code == 200 and r.json().get("success", False)
        except Exception as e:
            print(f"stop error: {e}")
            return False


# =========================
# ENUMS AND DATA CLASSES
# =========================
class AutonomousState(Enum):
    """State machine for autonomous mode"""
    IDLE = "idle"
    SCANNING = "scanning"
    OBJECT_FOUND = "object_found"
    TURNING = "turning"
    APPROACHING = "approaching"
    TARGET_REACHED = "target_reached"
    OBJECT_LOST = "object_lost"
    ERROR = "error"


@dataclass
class Detection:
    """Represents a single YOLO detection"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center x, y
    area: float  # Normalized area (0-1 relative to frame)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox,
            "center": self.center,
            "area": round(self.area, 4)
        }


@dataclass
class ScanResult:
    """Result of scanning at a single position"""
    position_angle: int
    position_name: str
    detections: List[Detection]
    target_found: bool
    target_detection: Optional[Detection]
    image_base64: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position_angle": self.position_angle,
            "position_name": self.position_name,
            "detections": [d.to_dict() for d in self.detections],
            "target_found": self.target_found,
            "target_detection": self.target_detection.to_dict() if self.target_detection else None,
            "has_image": self.image_base64 is not None
        }


# =========================
# YOLO DETECTOR
# =========================
class YOLODetector:
    """Handles YOLO model loading and inference"""
    
    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_path: str = None, confidence: float = None):
        self.model_path = model_path or AutonomousConfig.YOLO_MODEL_PATH
        self.confidence = confidence or AutonomousConfig.YOLO_CONFIDENCE
        self.model: Optional[YOLO] = None
        self._loaded = False
        
    def load_model(self) -> bool:
        """Load the YOLO model"""
        if not YOLO_AVAILABLE:
            print("YOLO not available - ultralytics not installed")
            return False
            
        try:
            print(f"Loading YOLO model: {self.model_path}")
            self.model = YOLO(self.model_path)
            self._loaded = True
            print("YOLO model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self._loaded = False
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._loaded and self.model is not None
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO detection on a frame
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            List of Detection objects
        """
        if not self.is_loaded():
            if not self.load_model():
                return []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence, verbose=False)
            
            detections = []
            frame_height, frame_width = frame.shape[:2]
            frame_area = frame_width * frame_height
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}"
                    
                    # Calculate center and area
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    box_area = (x2 - x1) * (y2 - y1)
                    normalized_area = box_area / frame_area
                    
                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=(center_x, center_y),
                        area=normalized_area
                    )
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def find_target(self, frame: np.ndarray, target_class: str) -> Optional[Detection]:
        """
        Find a specific target class in the frame
        
        Args:
            frame: BGR image from OpenCV
            target_class: Class name to find (e.g., "bottle", "person")
            
        Returns:
            Detection of target if found, None otherwise
        """
        detections = self.detect(frame)
        
        # Find the largest detection of the target class
        target_class_lower = target_class.lower()
        target_detections = [
            d for d in detections 
            if d.class_name.lower() == target_class_lower
        ]
        
        if not target_detections:
            return None
            
        # Return the one with largest area (closest/most prominent)
        return max(target_detections, key=lambda d: d.area)
    
    @classmethod
    def get_available_classes(cls) -> List[str]:
        """Get list of available COCO classes"""
        return cls.COCO_CLASSES.copy()


# =========================
# AUTONOMOUS CONTROLLER
# =========================
class AutonomousController:
    """
    Main controller for autonomous object search and approach.
    
    Workflow:
    1. User sets target object class
    2. Robot scans left, center, right positions
    3. YOLO detects objects in each frame
    4. Robot turns to face the direction where target was found
    5. Robot approaches until target is 50% of frame
    
    Uses a RobotClient (e.g. HttpRobotClient) so it can run as a separate process.
    """
    
    def __init__(self, robot_client: "RobotClient"):
        """
        Initialize autonomous controller
        
        Args:
            robot_client: RobotClient implementation (e.g. HttpRobotClient for HTTP API)
        """
        self.robot = robot_client
        self.detector = YOLODetector()
        
        # State
        self.state = AutonomousState.IDLE
        self.target_class: Optional[str] = None
        self.current_scan_results: List[ScanResult] = []
        self.last_detection: Optional[Detection] = None
        self.found_position: Optional[str] = None
        
        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Callbacks
        self._state_callback: Optional[Callable[[AutonomousState, Dict], None]] = None
        
        # Status tracking
        self.status_log: List[str] = []
        self.error_message: Optional[str] = None
    
    @staticmethod
    def _best_scan_result(scan_results: List["ScanResult"]) -> Optional["ScanResult"]:
        """Prefer the sector where the target has the largest normalized bbox area."""
        candidates = [r for r in scan_results if r.target_found and r.target_detection is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.target_detection.area)
    
    def _recover_target_after_loss(self) -> Optional[Detection]:
        """After repeated lost frames: try center view, then full servo scan + coarse turn."""
        self.robot.stop()
        self._log("Recovery: checking center view...")
        self._move_servo(AutonomousConfig.SERVO_CENTER_ANGLE)
        frame = self._capture_frame()
        if frame is not None:
            det = self.detector.find_target(frame, self.target_class)
            if det is not None:
                self._log("Recovery: target visible at center.")
                return det
        self._log("Recovery: full servo scan...")
        scan_results = self.scan_all_positions(include_images=False)
        best = AutonomousController._best_scan_result(scan_results)
        if best is None or best.target_detection is None:
            self._log("Recovery: target still not found.")
            return None
        self._move_servo(AutonomousConfig.SERVO_CENTER_ANGLE)
        if best.position_name == "left":
            self._log("Recovery: re-aligning after left sector hit.")
            self._turn_robot("left", 0.5)
        elif best.position_name == "right":
            self._log("Recovery: re-aligning after right sector hit.")
            self._turn_robot("right", 0.5)
        frame2 = self._capture_frame()
        if frame2 is None:
            return best.target_detection
        det = self.detector.find_target(frame2, self.target_class)
        return det if det is not None else best.target_detection
    
    def _log(self, message: str):
        """Add to status log"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.status_log.append(log_entry)
        print(f"[Autonomous] {message}")
        # Keep only last 50 entries
        if len(self.status_log) > 50:
            self.status_log = self.status_log[-50:]
    
    def _set_state(self, new_state: AutonomousState, data: Dict = None):
        """Update state and trigger callback"""
        old_state = self.state
        self.state = new_state
        self._log(f"State: {old_state.value} -> {new_state.value}")
        
        if self._state_callback:
            try:
                self._state_callback(new_state, data or {})
            except Exception as e:
                print(f"State callback error: {e}")
    
    def set_state_callback(self, callback: Callable[[AutonomousState, Dict], None]):
        """Set callback for state changes"""
        self._state_callback = callback
    
    def set_target(self, target_class: str) -> Dict[str, Any]:
        """
        Set the target object class to search for
        
        Args:
            target_class: COCO class name (e.g., "bottle", "person", "cup")
            
        Returns:
            Result dict with success status
        """
        target_lower = target_class.lower()
        available = YOLODetector.get_available_classes()
        
        if target_lower not in [c.lower() for c in available]:
            return {
                "success": False,
                "error": f"Unknown class: {target_class}",
                "available_classes": available
            }
        
        self.target_class = target_lower
        self._log(f"Target set to: {target_lower}")
        
        return {
            "success": True,
            "target_class": target_lower,
            "message": f"Target set to '{target_lower}'. Ready to search."
        }
    
    def _move_servo(self, angle: int) -> bool:
        """Move servo and wait for it to settle"""
        success = self.robot.set_servo(angle)
        if success:
            time.sleep(AutonomousConfig.SERVO_SETTLE_TIME)
        return success
    
    def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from camera as numpy array"""
        try:
            return self.robot.capture_frame()
        except Exception as e:
            print(f"Frame capture error: {e}")
            return None
    
    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer).decode('utf-8')
    
    def scan_all_positions(self, include_images: bool = True) -> List[ScanResult]:
        """
        Scan all 3 positions and detect objects
        
        Args:
            include_images: Whether to include base64 images in results
            
        Returns:
            List of ScanResult for each position
        """
        if not self.target_class:
            self._log("No target set for scanning")
            return []
        
        self._set_state(AutonomousState.SCANNING)
        results = []
        
        for angle, name in AutonomousConfig.get_scan_positions():
            if not self._running:
                break
                
            self._log(f"Scanning {name} position ({angle} deg)")
            
            # Move servo
            if not self._move_servo(angle):
                self._log(f"Failed to move servo to {angle} deg")
                continue
            
            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                self._log(f"Failed to capture frame at {name}")
                continue
            
            # Run detection — if multiple instances, use largest area (same as find_target)
            detections = self.detector.detect(frame)
            target_matches = [
                d for d in detections
                if d.class_name.lower() == self.target_class
            ]
            target_detection = None
            target_found = False
            if target_matches:
                target_detection = max(target_matches, key=lambda d: d.area)
                target_found = True
                self._log(
                    f"TARGET FOUND at {name}! Area: {target_detection.area:.2%}, "
                    f"Confidence: {target_detection.confidence:.2f}"
                )
            
            if not target_found:
                self._log(f"No target at {name}. Found: {[d.class_name for d in detections]}")
            
            result = ScanResult(
                position_angle=angle,
                position_name=name,
                detections=detections,
                target_found=target_found,
                target_detection=target_detection,
                image_base64=self._frame_to_base64(frame) if include_images else None
            )
            results.append(result)
        
        self.current_scan_results = results
        return results
    
    def _turn_robot(self, direction: str, duration: float = None):
        """Turn robot left or right"""
        duration = duration or AutonomousConfig.TURN_DURATION
        speed = AutonomousConfig.TURN_SPEED
        
        if direction == "left":
            self.robot.move("left", speed)
        elif direction == "right":
            self.robot.move("right", speed)
        
        time.sleep(duration)
        self.robot.stop()
        time.sleep(0.1)  # Brief pause
    
    def _move_forward(self, duration: float = None):
        """Move robot forward for a duration"""
        duration = duration or AutonomousConfig.APPROACH_STEP_DURATION
        speed = AutonomousConfig.APPROACH_SPEED
        
        self.robot.move("forward", speed)
        time.sleep(duration)
        self.robot.stop()
        time.sleep(0.1)
    
    def _center_on_target(self, detection: Detection, frame_width: int) -> bool:
        """
        Adjust robot rotation to center on target
        
        Returns True if target is centered, False if adjustment made
        """
        center_x = detection.center[0]
        frame_center = frame_width / 2
        offset = (center_x - frame_center) / frame_width  # Normalized offset
        
        tolerance = AutonomousConfig.CENTER_TOLERANCE
        
        if abs(offset) < tolerance:
            return True  # Already centered
        
        # Turn towards target
        if offset > 0:
            self._log(f"Target is right of center (offset: {offset:.2f}), turning right")
            self._turn_robot("right", 0.15)
        else:
            self._log(f"Target is left of center (offset: {offset:.2f}), turning left")
            self._turn_robot("left", 0.15)
        
        return False
    
    def _autonomous_loop(self):
        """Main autonomous control loop"""
        try:
            if not self.target_class:
                self._set_state(AutonomousState.ERROR)
                self.error_message = "No target class set"
                return
            
            # Load YOLO model
            if not self.detector.is_loaded():
                self._log("Loading YOLO model...")
                if not self.detector.load_model():
                    self._set_state(AutonomousState.ERROR)
                    self.error_message = "Failed to load YOLO model"
                    return
            
            # Phase 1: Scan all positions
            self._log("Starting scan phase...")
            scan_results = self.scan_all_positions(include_images=True)
            
            if not self._running:
                return
            
            found_result = AutonomousController._best_scan_result(scan_results)
            
            if not found_result:
                self._set_state(AutonomousState.OBJECT_LOST)
                self._log("Target not found in any position")
                # Return servo to center
                self._move_servo(AutonomousConfig.SERVO_CENTER_ANGLE)
                return
            
            self.found_position = found_result.position_name
            self.last_detection = found_result.target_detection
            self._set_state(AutonomousState.OBJECT_FOUND, {
                "position": found_result.position_name,
                "detection": found_result.target_detection.to_dict()
            })
            
            # Phase 2: Turn robot to face the detected direction
            self._set_state(AutonomousState.TURNING)
            
            # First, center the servo
            self._move_servo(AutonomousConfig.SERVO_CENTER_ANGLE)
            
            # Turn robot body based on where object was found
            if found_result.position_name == "left":
                self._log("Object on LEFT - turning robot left")
                self._turn_robot("left", 0.6)  # Larger turn
            elif found_result.position_name == "right":
                self._log("Object on RIGHT - turning robot right")
                self._turn_robot("right", 0.6)
            else:
                self._log("Object in CENTER - no turn needed")
            
            # Phase 3: Approach until target is 50% of frame
            self._set_state(AutonomousState.APPROACHING)
            
            approach_attempts = 0
            max_attempts = 50  # Safety limit
            lost_streak = 0
            
            while self._running and approach_attempts < max_attempts:
                approach_attempts += 1
                
                # Capture current frame
                frame = self._capture_frame()
                if frame is None:
                    self._log("Failed to capture frame during approach")
                    time.sleep(0.2)
                    continue
                
                frame_height, frame_width = frame.shape[:2]
                
                # Detect target
                detection = self.detector.find_target(frame, self.target_class)
                
                if detection is None:
                    lost_streak += 1
                    self._log(f"Lost sight of target (attempt {approach_attempts}, streak {lost_streak})")
                    if lost_streak >= AutonomousConfig.APPROACH_LOST_RESCAN_FRAMES:
                        recovered = self._recover_target_after_loss()
                        lost_streak = 0
                        if recovered is not None:
                            self.last_detection = recovered
                            self._log("Recovery: target re-acquired; continuing approach.")
                        time.sleep(0.2)
                        continue
                    time.sleep(0.3)
                    continue
                
                lost_streak = 0
                self.last_detection = detection
                self._log(f"Target area: {detection.area:.2%} (goal: {AutonomousConfig.TARGET_AREA_THRESHOLD:.0%})")
                
                # Check if target is large enough
                if detection.area >= AutonomousConfig.TARGET_AREA_THRESHOLD:
                    self._set_state(AutonomousState.TARGET_REACHED, {
                        "final_area": detection.area,
                        "detection": detection.to_dict()
                    })
                    self._log(f"TARGET REACHED! Final area: {detection.area:.2%}")
                    self.robot.stop()
                    return
                
                # Center on target first
                if not self._center_on_target(detection, frame_width):
                    continue  # Made an adjustment, re-check
                
                # Move forward
                self._log(f"Approaching... (area: {detection.area:.2%})")
                self._move_forward()
            
            if approach_attempts >= max_attempts:
                self._set_state(AutonomousState.ERROR)
                self.error_message = "Max approach attempts reached"
                self._log("Max approach attempts reached - stopping")
            
        except Exception as e:
            self._set_state(AutonomousState.ERROR)
            self.error_message = str(e)
            self._log(f"Error in autonomous loop: {e}")
        finally:
            self.robot.stop()  # Ensure stopped
            self._running = False
    
    def start(self) -> Dict[str, Any]:
        """Start autonomous search and approach"""
        with self._lock:
            if self._running:
                return {"success": False, "error": "Already running"}
            
            if not self.target_class:
                return {"success": False, "error": "No target set. Call set_target first."}
            
            self._running = True
            self.error_message = None
            self.status_log = []
            self.current_scan_results = []
            
            self._thread = threading.Thread(target=self._autonomous_loop, daemon=True)
            self._thread.start()
            
            return {
                "success": True,
                "message": f"Autonomous search started for '{self.target_class}'",
                "target": self.target_class
            }
    
    def stop(self) -> Dict[str, Any]:
        """Stop autonomous mode"""
        with self._lock:
            was_running = self._running
            self._running = False
            
            # Stop motors immediately
            self.robot.stop()
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)
            
            self._set_state(AutonomousState.IDLE)
            
            return {
                "success": True,
                "was_running": was_running,
                "message": "Autonomous mode stopped"
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current autonomous mode status"""
        return {
            "running": self._running,
            "state": self.state.value,
            "target_class": self.target_class,
            "found_position": self.found_position,
            "last_detection": self.last_detection.to_dict() if self.last_detection else None,
            "scan_results": [r.to_dict() for r in self.current_scan_results],
            "error": self.error_message,
            "log": self.status_log[-20:]  # Last 20 log entries
        }
    
    def scan_once(self, include_images: bool = True) -> Dict[str, Any]:
        """
        Perform a single scan without approaching.
        Useful for testing/calibration.
        """
        if self._running:
            return {"success": False, "error": "Autonomous mode is running"}
        
        if not self.target_class:
            return {"success": False, "error": "No target set"}
        
        # Load model if needed
        if not self.detector.is_loaded():
            if not self.detector.load_model():
                return {"success": False, "error": "Failed to load YOLO model"}
        
        results = self.scan_all_positions(include_images=include_images)
        
        # Return servo to center
        self._move_servo(AutonomousConfig.SERVO_CENTER_ANGLE)
        
        best = AutonomousController._best_scan_result(results)
        target_position = best.position_name if best else None
        target_detection = best.target_detection if best else None
        
        return {
            "success": True,
            "target_class": self.target_class,
            "target_found": target_position is not None,
            "target_position": target_position,
            "target_detection": target_detection.to_dict() if target_detection else None,
            "scan_results": [r.to_dict() for r in results]
        }
    
    def detect_single_frame(self) -> Dict[str, Any]:
        """
        Detect objects in current camera view without moving.
        Useful for calibration and testing.
        """
        if not self.detector.is_loaded():
            if not self.detector.load_model():
                return {"success": False, "error": "Failed to load YOLO model"}
        
        frame = self._capture_frame()
        if frame is None:
            return {"success": False, "error": "Failed to capture frame"}
        
        detections = self.detector.detect(frame)
        
        return {
            "success": True,
            "detections": [d.to_dict() for d in detections],
            "target_class": self.target_class,
            "target_found": any(d.class_name.lower() == self.target_class for d in detections) if self.target_class else None,
            "image_base64": self._frame_to_base64(frame)
        }


def is_yolo_available() -> bool:
    """Check if YOLO is available"""
    return YOLO_AVAILABLE
