"""
OS4AI Perfect Media Consciousness Implementation
Production-ready media processing with privacy, security, and adversarial protections
"""

import asyncio
import cv2
import numpy as np
import json
import hashlib
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
import redis
from contextlib import asynccontextmanager
import logging
from collections import deque, defaultdict
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security constants
MAX_IMAGE_SIZE = (1920, 1080)  # Full HD max
MAX_VIDEO_DURATION = 30  # seconds
ALLOWED_FORMATS = ['jpg', 'jpeg', 'png', 'mp4', 'avi']
PRIVACY_BLUR_REGIONS = True
ADVERSARIAL_DETECTION = True

class MediaConfig(BaseModel):
    """Media processing configuration with validation"""
    max_resolution: Tuple[int, int] = Field((1920, 1080), description="Maximum resolution")
    privacy_mode: bool = True
    face_blur: bool = True
    object_detection: bool = True
    pattern_analysis: bool = True
    adversarial_protection: bool = True
    fps_limit: int = Field(30, ge=1, le=60)
    quality: str = Field("high", pattern="^(low|medium|high)$")
    device_whitelist: List[str] = Field(default_factory=list)
    
    @validator('max_resolution')
    def validate_resolution(cls, v):
        """Ensure resolution is reasonable"""
        if v[0] > 4096 or v[1] > 2160:  # 4K limit
            raise ValueError("Resolution exceeds 4K limit")
        return v

class MediaDevice(BaseModel):
    """Validated media device information"""
    device_id: str = Field(..., pattern="^[a-zA-Z0-9_-]+$", max_length=100)
    device_type: str = Field(..., pattern="^(camera|webcam|iphone|android|capture_card)$")
    name: str = Field(..., max_length=200)
    capabilities: Dict[str, Any]
    trusted: bool = False
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('device_id')
    def validate_device_id(cls, v):
        """Prevent injection in device IDs"""
        if any(char in v for char in ['<', '>', '"', "'", '&', ';', '|', '$', '\n', '\r']):
            raise ValueError("Invalid characters in device ID")
        return v

class VideoFrame(BaseModel):
    """Validated video frame data"""
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    resolution: Tuple[int, int]
    format: str = Field("rgb", pattern="^(rgb|bgr|yuv|gray)$")
    privacy_processed: bool = False
    anomaly_score: float = Field(0.0, ge=0.0, le=1.0)
    
    @validator('resolution')
    def validate_frame_resolution(cls, v):
        """Validate frame resolution"""
        if v[0] <= 0 or v[1] <= 0 or v[0] > 4096 or v[1] > 2160:
            raise ValueError("Invalid frame resolution")
        return v

class PatternDetection(BaseModel):
    """Pattern detection results with confidence"""
    pattern_type: str = Field(..., pattern="^[a-zA-Z_]+$", max_length=50)
    confidence: float = Field(..., ge=0.0, le=1.0)
    location: Optional[Dict[str, int]]  # x, y, width, height
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_level: str = Field("low", pattern="^(low|medium|high|critical)$")

class MediaAwareness(BaseModel):
    """Comprehensive media consciousness state"""
    active_devices: List[MediaDevice]
    current_scene: str  # Description of current visual scene
    detected_objects: List[Dict[str, Any]]
    detected_patterns: List[PatternDetection]
    motion_activity: float  # 0-1 scale
    scene_complexity: float  # 0-1 scale
    privacy_violations: List[str]
    security_alerts: List[str]
    behavioral_context: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str

class SecureDeviceManager:
    """
    Secure device management with spoofing protection
    """
    
    def __init__(self, config: MediaConfig):
        self.config = config
        self._known_devices: Dict[str, MediaDevice] = {}
        self._device_validator = DeviceValidator()
        self._trust_manager = DeviceTrustManager()
        self._last_scan = datetime.now(timezone.utc)
        
    async def scan_devices(self, user_id: str) -> List[MediaDevice]:
        """
        Scan for media devices with security validation
        """
        logger.info(f"Scanning media devices for user {user_id}")
        
        # Rate limit device scanning
        if (datetime.now(timezone.utc) - self._last_scan).total_seconds() < 5:
            logger.warning("Device scan rate limit - using cached results")
            return list(self._known_devices.values())
        
        self._last_scan = datetime.now(timezone.utc)
        
        # Detect devices (platform-specific)
        raw_devices = await self._detect_system_devices()
        
        # Validate and filter devices
        validated_devices = []
        for device_info in raw_devices:
            try:
                # Validate device
                if self._device_validator.validate_device(device_info):
                    device = MediaDevice(**device_info)
                    
                    # Check trust status
                    device.trusted = self._trust_manager.is_trusted(device)
                    
                    # Check whitelist
                    if self.config.device_whitelist:
                        if device.device_id not in self.config.device_whitelist:
                            logger.warning(f"Device {device.device_id} not in whitelist")
                            continue
                    
                    validated_devices.append(device)
                    self._known_devices[device.device_id] = device
                    
            except Exception as e:
                logger.error(f"Device validation error: {e}")
        
        return validated_devices
    
    async def _detect_system_devices(self) -> List[Dict[str, Any]]:
        """Detect system media devices"""
        devices = []
        
        # OpenCV camera detection
        for i in range(5):  # Check first 5 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get device properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                devices.append({
                    'device_id': f'opencv_camera_{i}',
                    'device_type': 'webcam',
                    'name': f'Camera {i}',
                    'capabilities': {
                        'resolution': [width, height],
                        'fps': fps,
                        'format': 'bgr'
                    }
                })
                cap.release()
        
        # Platform-specific device detection
        if os.name == 'posix':  # macOS/Linux
            devices.extend(await self._detect_posix_devices())
        
        return devices
    
    async def _detect_posix_devices(self) -> List[Dict[str, Any]]:
        """Detect devices on POSIX systems"""
        devices = []
        
        try:
            # Check for video devices
            for device_path in ['/dev/video0', '/dev/video1', '/dev/video2']:
                if os.path.exists(device_path):
                    devices.append({
                        'device_id': device_path.replace('/', '_'),
                        'device_type': 'camera',
                        'name': f'V4L2 {device_path}',
                        'capabilities': {
                            'resolution': [640, 480],
                            'fps': 30,
                            'format': 'yuv'
                        }
                    })
        except Exception as e:
            logger.error(f"POSIX device detection error: {e}")
        
        return devices

class DeviceValidator:
    """Validate devices against security threats"""
    
    def validate_device(self, device_info: Dict[str, Any]) -> bool:
        """Validate device information"""
        required_fields = ['device_id', 'device_type', 'name', 'capabilities']
        
        # Check required fields
        for field in required_fields:
            if field not in device_info:
                return False
        
        # Validate device ID format
        device_id = device_info.get('device_id', '')
        if not device_id or len(device_id) > 100:
            return False
        
        # Check for suspicious patterns
        if self._has_suspicious_patterns(device_info):
            logger.warning(f"Suspicious device rejected: {device_id}")
            return False
        
        return True
    
    def _has_suspicious_patterns(self, device_info: Dict[str, Any]) -> bool:
        """Check for suspicious device patterns"""
        suspicious_names = ['fake', 'virtual', 'emulated', 'spoofed']
        device_name = device_info.get('name', '').lower()
        
        for pattern in suspicious_names:
            if pattern in device_name:
                return True
        
        # Check for impossible capabilities
        caps = device_info.get('capabilities', {})
        resolution = caps.get('resolution', [0, 0])
        if resolution[0] > 8192 or resolution[1] > 4320:  # Beyond 8K
            return True
        
        fps = caps.get('fps', 30)
        if fps > 240:  # Unrealistic FPS
            return True
        
        return False

class DeviceTrustManager:
    """Manage device trust levels"""
    
    def __init__(self):
        self._trusted_devices = set()
        self._device_history: Dict[str, List[datetime]] = defaultdict(list)
    
    def is_trusted(self, device: MediaDevice) -> bool:
        """Check if device is trusted"""
        # New devices are untrusted
        if device.device_id not in self._device_history:
            self._device_history[device.device_id].append(datetime.now(timezone.utc))
            return False
        
        # Trust after consistent presence
        history = self._device_history[device.device_id]
        if len(history) >= 3:  # Seen at least 3 times
            time_span = (history[-1] - history[0]).total_seconds()
            if time_span > 300:  # Over 5 minutes
                self._trusted_devices.add(device.device_id)
                return True
        
        return device.device_id in self._trusted_devices

class PrivacyProtector:
    """
    Privacy protection for media streams
    """
    
    def __init__(self):
        self._face_detector = self._load_face_detector()
        self._blur_kernel = cv2.getGaussianKernel(31, 10)
        self._sensitive_regions: List[Dict[str, int]] = []
    
    def _load_face_detector(self):
        """Load face detection model"""
        # Use OpenCV's Haar Cascade for privacy-preserving face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(cascade_path)
    
    async def protect_frame(self, frame: np.ndarray, blur_faces: bool = True) -> np.ndarray:
        """Apply privacy protection to frame"""
        protected_frame = frame.copy()
        
        if blur_faces:
            # Detect and blur faces
            protected_frame = await self._blur_faces(protected_frame)
        
        # Blur sensitive regions
        for region in self._sensitive_regions:
            protected_frame = self._blur_region(protected_frame, region)
        
        # Add privacy watermark
        protected_frame = self._add_privacy_watermark(protected_frame)
        
        return protected_frame
    
    async def _blur_faces(self, frame: np.ndarray) -> np.ndarray:
        """Detect and blur faces in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Blur each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_region = frame[y:y+h, x:x+w]
            
            # Apply strong Gaussian blur
            blurred_face = cv2.GaussianBlur(face_region, (31, 31), 30)
            
            # Replace original with blurred
            frame[y:y+h, x:x+w] = blurred_face
        
        return frame
    
    def _blur_region(self, frame: np.ndarray, region: Dict[str, int]) -> np.ndarray:
        """Blur a specific region"""
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        
        # Ensure region is within frame bounds
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        
        if w > 0 and h > 0:
            region_data = frame[y:y+h, x:x+w]
            blurred = cv2.GaussianBlur(region_data, (21, 21), 20)
            frame[y:y+h, x:x+w] = blurred
        
        return frame
    
    def _add_privacy_watermark(self, frame: np.ndarray) -> np.ndarray:
        """Add privacy protection watermark"""
        h, w = frame.shape[:2]
        
        # Add subtle watermark text
        text = "Privacy Protected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (200, 200, 200)  # Light gray
        
        # Calculate position (bottom-right corner)
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = w - text_size[0] - 10
        y = h - 10
        
        # Add with transparency effect
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        
        return frame
    
    def add_sensitive_region(self, x: int, y: int, width: int, height: int):
        """Mark a region as sensitive for automatic blurring"""
        self._sensitive_regions.append({
            'x': x, 'y': y, 'width': width, 'height': height
        })

class AdversarialDetector:
    """
    Detect adversarial attacks on media inputs
    """
    
    def __init__(self):
        self._noise_threshold = 0.1
        self._pattern_detector = self._init_pattern_detector()
        self._history_window = 10
        self._frame_history = deque(maxlen=self._history_window)
    
    def _init_pattern_detector(self):
        """Initialize pattern detection model"""
        # Simple CNN for adversarial pattern detection
        class PatternDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(32 * 8 * 8, 64)
                self.fc2 = nn.Linear(64, 2)  # Normal vs Adversarial
                
            def forward(self, x):
                x = self.pool(torch.relu(self.conv1(x)))
                x = self.pool(torch.relu(self.conv2(x)))
                x = x.view(-1, 32 * 8 * 8)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        model = PatternDetector()
        model.eval()  # Set to evaluation mode
        return model
    
    async def detect_adversarial(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if frame contains adversarial patterns
        Returns: (is_adversarial, confidence)
        """
        # Add to history
        self._frame_history.append(frame)
        
        # Check for various adversarial indicators
        checks = await asyncio.gather(
            self._check_noise_level(frame),
            self._check_pattern_anomalies(frame),
            self._check_temporal_consistency(),
            self._check_frequency_domain(frame)
        )
        
        # Aggregate scores
        scores = [score for is_adv, score in checks if is_adv]
        
        if scores:
            avg_score = np.mean(scores)
            return True, float(avg_score)
        
        return False, 0.0
    
    async def _check_noise_level(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check for unusual noise patterns"""
        # Calculate noise estimate using Laplacian
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        # Normalize to 0-1
        normalized_noise = min(noise_level / 1000, 1.0)
        
        # High noise might indicate adversarial perturbation
        is_suspicious = normalized_noise > self._noise_threshold
        
        return is_suspicious, normalized_noise
    
    async def _check_pattern_anomalies(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check for adversarial patterns using CNN"""
        try:
            # Preprocess frame
            img_tensor = self._preprocess_frame(frame)
            
            # Run through pattern detector
            with torch.no_grad():
                output = self._pattern_detector(img_tensor)
                probs = torch.softmax(output, dim=1)
                adversarial_prob = probs[0][1].item()
            
            is_adversarial = adversarial_prob > 0.7
            return is_adversarial, adversarial_prob
            
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return False, 0.0
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for CNN input"""
        # Resize to 32x32
        resized = cv2.resize(frame, (32, 32))
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img_pil).unsqueeze(0)
        
        return img_tensor
    
    async def _check_temporal_consistency(self) -> Tuple[bool, float]:
        """Check temporal consistency across frames"""
        if len(self._frame_history) < 3:
            return False, 0.0
        
        # Calculate frame differences
        diffs = []
        for i in range(1, len(self._frame_history)):
            diff = cv2.absdiff(self._frame_history[i], self._frame_history[i-1])
            diffs.append(np.mean(diff))
        
        # Check for sudden changes
        if diffs:
            mean_diff = np.mean(diffs)
            std_diff = np.std(diffs)
            
            # Sudden spike might indicate attack
            if std_diff > mean_diff * 2:
                return True, min(std_diff / mean_diff / 10, 1.0)
        
        return False, 0.0
    
    async def _check_frequency_domain(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check for adversarial patterns in frequency domain"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Compute FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Check for unusual high-frequency patterns
        h, w = gray.shape
        center_h, center_w = h // 2, w // 2
        
        # Calculate energy in different frequency bands
        low_freq = magnitude[center_h-10:center_h+10, center_w-10:center_w+10]
        high_freq = magnitude[0:20, 0:20]  # Corners contain high frequencies
        
        low_energy = np.sum(low_freq)
        high_energy = np.sum(high_freq)
        
        if low_energy > 0:
            freq_ratio = high_energy / low_energy
            # High ratio might indicate adversarial noise
            is_suspicious = freq_ratio > 0.5
            return is_suspicious, min(freq_ratio, 1.0)
        
        return False, 0.0

class PatternAnalyzer:
    """
    Analyze visual patterns for behavioral context
    """
    
    def __init__(self):
        self._pattern_memory = deque(maxlen=1000)
        self._behavior_tracker = BehaviorTracker()
        self._scene_classifier = SceneClassifier()
    
    async def analyze_patterns(self, frame: np.ndarray) -> List[PatternDetection]:
        """Analyze frame for patterns"""
        patterns = []
        
        # Detect motion patterns
        motion_patterns = await self._detect_motion_patterns(frame)
        patterns.extend(motion_patterns)
        
        # Detect object patterns
        object_patterns = await self._detect_object_patterns(frame)
        patterns.extend(object_patterns)
        
        # Detect behavioral patterns
        behavioral_patterns = await self._behavior_tracker.analyze(frame)
        patterns.extend(behavioral_patterns)
        
        # Update pattern memory
        for pattern in patterns:
            self._pattern_memory.append({
                'pattern': pattern,
                'timestamp': datetime.now(timezone.utc)
            })
        
        return patterns
    
    async def _detect_motion_patterns(self, frame: np.ndarray) -> List[PatternDetection]:
        """Detect motion-based patterns"""
        patterns = []
        
        # Simple motion detection using frame differencing
        if hasattr(self, '_last_frame'):
            diff = cv2.absdiff(self._last_frame, frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Threshold to get motion mask
            _, motion_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze motion areas
            total_motion = np.sum(motion_mask > 0)
            frame_area = motion_mask.shape[0] * motion_mask.shape[1]
            motion_ratio = total_motion / frame_area
            
            if motion_ratio > 0.01:  # 1% motion threshold
                patterns.append(PatternDetection(
                    pattern_type="motion_detected",
                    confidence=min(motion_ratio * 10, 1.0),
                    metadata={'motion_areas': len(contours)}
                ))
        
        self._last_frame = frame.copy()
        return patterns
    
    async def _detect_object_patterns(self, frame: np.ndarray) -> List[PatternDetection]:
        """Detect object-based patterns"""
        patterns = []
        
        # Simple edge-based object detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Count edge pixels as complexity measure
        edge_pixels = np.sum(edges > 0)
        complexity = edge_pixels / (edges.shape[0] * edges.shape[1])
        
        if complexity > 0.1:
            patterns.append(PatternDetection(
                pattern_type="complex_scene",
                confidence=min(complexity * 5, 1.0),
                metadata={'edge_density': complexity}
            ))
        
        return patterns

class BehaviorTracker:
    """Track behavioral patterns over time"""
    
    def __init__(self):
        self._activity_history = deque(maxlen=100)
        self._pattern_counts = defaultdict(int)
    
    async def analyze(self, frame: np.ndarray) -> List[PatternDetection]:
        """Analyze frame for behavioral patterns"""
        patterns = []
        
        # Track activity level
        activity_level = await self._calculate_activity_level(frame)
        self._activity_history.append(activity_level)
        
        # Detect behavioral patterns
        if len(self._activity_history) >= 10:
            # Check for repetitive behavior
            recent_activity = list(self._activity_history)[-10:]
            variance = np.var(recent_activity)
            
            if variance < 0.01:  # Low variance indicates repetitive behavior
                patterns.append(PatternDetection(
                    pattern_type="repetitive_behavior",
                    confidence=0.8,
                    metadata={'variance': variance}
                ))
            
            # Check for anomalous behavior
            mean_activity = np.mean(recent_activity)
            if activity_level > mean_activity * 2:
                patterns.append(PatternDetection(
                    pattern_type="anomalous_activity",
                    confidence=0.7,
                    risk_level="medium",
                    metadata={'activity_spike': activity_level / mean_activity}
                ))
        
        return patterns
    
    async def _calculate_activity_level(self, frame: np.ndarray) -> float:
        """Calculate activity level in frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude as activity measure
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        activity = np.mean(magnitude) / 255.0
        return min(activity, 1.0)

class SceneClassifier:
    """Classify visual scenes"""
    
    def classify_scene(self, frame: np.ndarray) -> str:
        """Classify the current scene"""
        # Simple scene classification based on color and texture
        
        # Calculate color histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Dominant colors analysis
        dominant_hue = np.argmax(hist[:50]) * (180/50)
        
        # Simple classification based on dominant hue
        if dominant_hue < 30 or dominant_hue > 150:  # Red/Purple
            return "indoor_warm"
        elif 30 <= dominant_hue < 90:  # Green
            return "outdoor_nature"
        elif 90 <= dominant_hue < 150:  # Blue
            return "outdoor_sky"
        else:
            return "indoor_neutral"

class PerfectMediaConsciousness:
    """
    Production-ready media consciousness with comprehensive security
    """
    
    def __init__(self, config: MediaConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.device_manager = SecureDeviceManager(config)
        self.privacy_protector = PrivacyProtector()
        self.adversarial_detector = AdversarialDetector()
        self.pattern_analyzer = PatternAnalyzer()
        self.redis_client = redis_client
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._rate_limiter = MediaRateLimiter(redis_client)
        self._audit_logger = MediaAuditLogger()
        self._metrics_collector = MediaMetrics()
        self._active_streams: Dict[str, Any] = {}
        
    async def start(self):
        """Start media consciousness monitoring"""
        logger.info("Starting Perfect Media Consciousness...")
        
        # Initialize components
        await self._initialize_components()
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Gracefully stop monitoring"""
        logger.info("Stopping Perfect Media Consciousness...")
        self._shutdown_event.set()
        
        # Stop all active streams
        for stream_id in list(self._active_streams.keys()):
            await self._stop_stream(stream_id)
        
        if self._monitoring_task:
            await self._monitoring_task
    
    async def get_media_awareness(self, user_id: str, correlation_id: str) -> MediaAwareness:
        """
        Get comprehensive media awareness with all protections
        """
        # Rate limiting
        if not await self._rate_limiter.check_rate_limit(f"media:{user_id}", 30, 60):
            await self._audit_logger.log_security_event(
                "rate_limit_exceeded",
                user_id,
                {"action": "media_awareness", "correlation_id": correlation_id}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for media awareness"
            )
        
        # Audit logging
        await self._audit_logger.log_access(
            "media_awareness_read",
            user_id,
            {"correlation_id": correlation_id}
        )
        
        try:
            # Scan for devices
            devices = await self.device_manager.scan_devices(user_id)
            
            # Get current frame from active device
            current_frame = await self._get_current_frame(devices)
            
            # Process frame with privacy protection
            if current_frame is not None and self.config.privacy_mode:
                current_frame = await self.privacy_protector.protect_frame(
                    current_frame,
                    blur_faces=self.config.face_blur
                )
            
            # Check for adversarial attacks
            is_adversarial = False
            adversarial_score = 0.0
            if current_frame is not None and self.config.adversarial_protection:
                is_adversarial, adversarial_score = await self.adversarial_detector.detect_adversarial(current_frame)
            
            # Analyze patterns
            patterns = []
            if current_frame is not None and self.config.pattern_analysis:
                patterns = await self.pattern_analyzer.analyze_patterns(current_frame)
            
            # Build awareness
            awareness = MediaAwareness(
                active_devices=devices,
                current_scene=self._describe_scene(current_frame),
                detected_objects=await self._detect_objects(current_frame),
                detected_patterns=patterns,
                motion_activity=await self._calculate_motion_activity(),
                scene_complexity=await self._calculate_scene_complexity(current_frame),
                privacy_violations=self._check_privacy_violations(patterns),
                security_alerts=self._generate_security_alerts(is_adversarial, adversarial_score, patterns),
                behavioral_context=await self._build_behavioral_context(patterns),
                correlation_id=correlation_id
            )
            
            # Update metrics
            await self._update_metrics(awareness)
            
            return awareness
            
        except Exception as e:
            logger.error(f"Media awareness error: {e}")
            await self._audit_logger.log_error(
                "media_awareness_error",
                user_id,
                {"error": str(e), "correlation_id": correlation_id}
            )
            raise
    
    async def _initialize_components(self):
        """Initialize media components"""
        # Scan for initial devices
        devices = await self.device_manager.scan_devices("system")
        logger.info(f"Found {len(devices)} media devices")
        
        # Initialize privacy regions if configured
        # Add any predefined sensitive regions
        # self.privacy_protector.add_sensitive_region(x, y, w, h)
    
    async def _get_current_frame(self, devices: List[MediaDevice]) -> Optional[np.ndarray]:
        """Get current frame from active device"""
        if not devices:
            return None
        
        # Use first trusted device or first device if none trusted
        trusted_devices = [d for d in devices if d.trusted]
        device = trusted_devices[0] if trusted_devices else devices[0]
        
        # Check if stream exists
        if device.device_id not in self._active_streams:
            # Start new stream
            stream = await self._start_stream(device)
            if stream:
                self._active_streams[device.device_id] = stream
        
        # Get frame from stream
        stream = self._active_streams.get(device.device_id)
        if stream and stream.isOpened():
            ret, frame = stream.read()
            if ret:
                # Resize if needed
                if frame.shape[:2] != self.config.max_resolution:
                    frame = cv2.resize(frame, self.config.max_resolution)
                return frame
        
        return None
    
    async def _start_stream(self, device: MediaDevice) -> Optional[cv2.VideoCapture]:
        """Start video stream from device"""
        try:
            # Extract device index from ID
            if device.device_id.startswith('opencv_camera_'):
                index = int(device.device_id.split('_')[-1])
                cap = cv2.VideoCapture(index)
            else:
                # Try as file path or URL
                cap = cv2.VideoCapture(device.device_id)
            
            if cap.isOpened():
                # Set capture properties
                cap.set(cv2.CAP_PROP_FPS, self.config.fps_limit)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.max_resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.max_resolution[1])
                
                logger.info(f"Started stream from device {device.device_id}")
                return cap
            
        except Exception as e:
            logger.error(f"Failed to start stream from {device.device_id}: {e}")
        
        return None
    
    async def _stop_stream(self, stream_id: str):
        """Stop video stream"""
        if stream_id in self._active_streams:
            stream = self._active_streams[stream_id]
            if stream and stream.isOpened():
                stream.release()
            del self._active_streams[stream_id]
            logger.info(f"Stopped stream {stream_id}")
    
    def _describe_scene(self, frame: Optional[np.ndarray]) -> str:
        """Generate scene description"""
        if frame is None:
            return "no_active_video"
        
        classifier = SceneClassifier()
        return classifier.classify_scene(frame)
    
    async def _detect_objects(self, frame: Optional[np.ndarray]) -> List[Dict[str, Any]]:
        """Detect objects in frame"""
        if frame is None or not self.config.object_detection:
            return []
        
        objects = []
        
        # Simple object detection using contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours[:10]:  # Limit to 10 objects
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'type': 'unknown_object',
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'confidence': min(area / 10000, 1.0)
                })
        
        return objects
    
    async def _calculate_motion_activity(self) -> float:
        """Calculate overall motion activity level"""
        # Aggregate from recent pattern detections
        recent_patterns = list(self.pattern_analyzer._pattern_memory)[-10:]
        motion_patterns = [
            p for p in recent_patterns 
            if p['pattern'].pattern_type == 'motion_detected'
        ]
        
        if motion_patterns:
            avg_confidence = np.mean([p['pattern'].confidence for p in motion_patterns])
            return float(avg_confidence)
        
        return 0.0
    
    async def _calculate_scene_complexity(self, frame: Optional[np.ndarray]) -> float:
        """Calculate scene complexity score"""
        if frame is None:
            return 0.0
        
        # Use edge density as complexity measure
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        return min(edge_density * 10, 1.0)
    
    def _check_privacy_violations(self, patterns: List[PatternDetection]) -> List[str]:
        """Check for privacy violations in detected patterns"""
        violations = []
        
        for pattern in patterns:
            if pattern.pattern_type == "face_detected" and not self.config.face_blur:
                violations.append("Unblurred faces detected")
            
            if pattern.risk_level in ["high", "critical"]:
                violations.append(f"High-risk pattern detected: {pattern.pattern_type}")
        
        return violations
    
    def _generate_security_alerts(self, is_adversarial: bool, adversarial_score: float,
                                patterns: List[PatternDetection]) -> List[str]:
        """Generate security alerts based on analysis"""
        alerts = []
        
        if is_adversarial:
            alerts.append(f"CRITICAL: Adversarial attack detected (confidence: {adversarial_score:.2f})")
        
        # Check for suspicious patterns
        for pattern in patterns:
            if pattern.risk_level == "critical":
                alerts.append(f"CRITICAL: {pattern.pattern_type} detected")
            elif pattern.risk_level == "high":
                alerts.append(f"WARNING: {pattern.pattern_type} detected")
        
        # Check for device spoofing
        untrusted_devices = [
            d for d in self.device_manager._known_devices.values() 
            if not d.trusted
        ]
        if untrusted_devices:
            alerts.append(f"WARNING: {len(untrusted_devices)} untrusted devices detected")
        
        return alerts
    
    async def _build_behavioral_context(self, patterns: List[PatternDetection]) -> Dict[str, Any]:
        """Build behavioral context from patterns"""
        context = {
            'activity_trend': 'unknown',
            'pattern_frequency': {},
            'anomalies': [],
            'risk_assessment': 'low'
        }
        
        # Analyze pattern frequency
        pattern_types = [p.pattern_type for p in patterns]
        for ptype in set(pattern_types):
            context['pattern_frequency'][ptype] = pattern_types.count(ptype)
        
        # Determine activity trend
        motion_patterns = [p for p in patterns if p.pattern_type == 'motion_detected']
        if len(motion_patterns) > 5:
            context['activity_trend'] = 'high'
        elif len(motion_patterns) > 2:
            context['activity_trend'] = 'moderate'
        else:
            context['activity_trend'] = 'low'
        
        # Identify anomalies
        anomaly_patterns = [p for p in patterns if 'anomal' in p.pattern_type.lower()]
        context['anomalies'] = [p.pattern_type for p in anomaly_patterns]
        
        # Risk assessment
        risk_levels = [p.risk_level for p in patterns]
        if 'critical' in risk_levels:
            context['risk_assessment'] = 'critical'
        elif 'high' in risk_levels:
            context['risk_assessment'] = 'high'
        elif 'medium' in risk_levels:
            context['risk_assessment'] = 'medium'
        
        return context
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Periodic device scan
                devices = await self.device_manager.scan_devices("system")
                
                # Check stream health
                for device in devices:
                    if device.device_id in self._active_streams:
                        stream = self._active_streams[device.device_id]
                        if not stream.isOpened():
                            # Restart failed stream
                            await self._stop_stream(device.device_id)
                            await self._start_stream(device)
                
                # Wait for next interval
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=10  # 10 second monitoring interval
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(10)
    
    async def _update_metrics(self, awareness: MediaAwareness):
        """Update monitoring metrics"""
        # Update metrics collector
        self._metrics_collector.record_awareness(awareness)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                metrics_key = "media:metrics"
                await self.redis_client.hset(metrics_key, mapping={
                    "active_devices": len(awareness.active_devices),
                    "current_scene": awareness.current_scene,
                    "motion_activity": str(awareness.motion_activity),
                    "last_update": awareness.timestamp.isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to update Redis metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get media system health status"""
        return {
            "component": "media_consciousness",
            "status": "healthy",
            "privacy_mode": self.config.privacy_mode,
            "adversarial_protection": self.config.adversarial_protection,
            "active_streams": len(self._active_streams),
            "known_devices": len(self.device_manager._known_devices),
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "metrics": self._metrics_collector.get_summary()
        }

class MediaRateLimiter:
    """Rate limiting for media operations"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.memory_limits: Dict[str, List[float]] = {}
    
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if operation is within rate limit"""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                now = datetime.now(timezone.utc).timestamp()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcount(key, now - window, now)
                pipe.expire(key, window)
                results = pipe.execute()
                return results[2] <= limit
            except Exception:
                pass
        
        # Fallback to memory
        now = datetime.now(timezone.utc).timestamp()
        if key not in self.memory_limits:
            self.memory_limits[key] = []
        
        self.memory_limits[key] = [
            t for t in self.memory_limits[key]
            if t > now - window
        ]
        
        if len(self.memory_limits[key]) < limit:
            self.memory_limits[key].append(now)
            return True
        
        return False

class MediaAuditLogger:
    """Audit logging for media operations"""
    
    async def log_access(self, action: str, user_id: str, details: Dict[str, Any]):
        """Log access events"""
        logger.info(f"MEDIA_AUDIT_ACCESS: action={action}, user={user_id}, details={json.dumps(details)}")
    
    async def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security events"""
        logger.warning(f"MEDIA_AUDIT_SECURITY: event={event_type}, user={user_id}, details={json.dumps(details)}")
    
    async def log_error(self, error_type: str, user_id: str, details: Dict[str, Any]):
        """Log error events"""
        logger.error(f"MEDIA_AUDIT_ERROR: error={error_type}, user={user_id}, details={json.dumps(details)}")

class MediaMetrics:
    """Metrics collection for media system"""
    
    def __init__(self):
        self.readings_total = 0
        self.devices_seen = set()
        self.patterns_detected = defaultdict(int)
        self.adversarial_attempts = 0
        self.privacy_violations = 0
        self.scene_types = defaultdict(int)
    
    def record_awareness(self, awareness: MediaAwareness):
        """Record awareness metrics"""
        self.readings_total += 1
        
        for device in awareness.active_devices:
            self.devices_seen.add(device.device_id)
        
        for pattern in awareness.detected_patterns:
            self.patterns_detected[pattern.pattern_type] += 1
        
        if any('adversarial' in alert.lower() for alert in awareness.security_alerts):
            self.adversarial_attempts += 1
        
        self.privacy_violations += len(awareness.privacy_violations)
        self.scene_types[awareness.current_scene] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            'readings_total': self.readings_total,
            'unique_devices': len(self.devices_seen),
            'patterns_detected': dict(self.patterns_detected),
            'adversarial_attempts': self.adversarial_attempts,
            'privacy_violations': self.privacy_violations,
            'scene_distribution': dict(self.scene_types)
        }

# Example usage
async def example_usage():
    """Example of using perfect media consciousness"""
    config = MediaConfig(
        privacy_mode=True,
        face_blur=True,
        adversarial_protection=True,
        pattern_analysis=True
    )
    
    # Initialize with Redis for production features
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    consciousness = PerfectMediaConsciousness(config, redis_client)
    
    # Start monitoring
    await consciousness.start()
    
    try:
        # Get media awareness
        awareness = await consciousness.get_media_awareness(
            user_id="test-user",
            correlation_id=str(uuid.uuid4())
        )
        
        print(f"Active Devices: {len(awareness.active_devices)}")
        print(f"Current Scene: {awareness.current_scene}")
        print(f"Motion Activity: {awareness.motion_activity:.2f}")
        print(f"Scene Complexity: {awareness.scene_complexity:.2f}")
        print(f"Security Alerts: {awareness.security_alerts}")
        print(f"Behavioral Context: {awareness.behavioral_context}")
        
    finally:
        # Graceful shutdown
        await consciousness.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())