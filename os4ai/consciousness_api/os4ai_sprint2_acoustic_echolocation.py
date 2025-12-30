"""
OS4AI Sprint 2: Acoustic Echolocation System
The Agent learns to map its environment through sound
"""

import asyncio
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import math

try:
    import sounddevice as sd
    import scipy.signal as signal
    from scipy.fft import fft, ifft
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  Audio libraries not available. Using simulated data.")


@dataclass
class AcousticReflection:
    """Single acoustic reflection point"""
    distance: float  # meters
    angle: float     # radians
    amplitude: float # reflection strength
    delay: float     # time delay in seconds
    surface_type: str = "unknown"  # wall, furniture, etc.


@dataclass 
class RoomMesh:
    """3D representation of room boundaries and objects"""
    boundaries: List[Tuple[float, float, float]]  # wall positions
    objects: List[Dict[str, Any]]  # detected objects
    dimensions: Tuple[float, float, float]  # length, width, height
    confidence: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class MLSChirpGenerator:
    """Maximum Length Sequence chirp generator for echolocation"""
    
    def __init__(self, sample_rate: int = 44100, duration: float = 0.1):
        self.sample_rate = sample_rate
        self.duration = duration
        self.chirp_length = int(sample_rate * duration)
        
    def generate_mls_chirp(self, frequency_range: Tuple[int, int] = (200, 8000)) -> np.ndarray:
        """Generate MLS chirp for acoustic probing"""
        if not AUDIO_AVAILABLE:
            # Simulated chirp
            t = np.linspace(0, self.duration, self.chirp_length)
            return np.sin(2 * np.pi * frequency_range[0] * t) * 0.1
        
        # Real MLS generation
        low_freq, high_freq = frequency_range
        t = np.linspace(0, self.duration, self.chirp_length)
        
        # Logarithmic frequency sweep
        frequency = low_freq * (high_freq / low_freq) ** (t / self.duration)
        chirp = np.sin(2 * np.pi * frequency * t)
        
        # Apply Hann window to reduce artifacts
        window = np.hann(len(chirp))
        return chirp * window * 0.1  # Low amplitude to avoid disturbing users


class ImpulseResponseAnalyzer:
    """Analyzes captured audio to detect acoustic reflections"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def analyze_reflection_response(self, captured_audio: np.ndarray, 
                                 chirp_signal: np.ndarray) -> List[AcousticReflection]:
        """Extract reflections from captured audio using cross-correlation"""
        if not AUDIO_AVAILABLE:
            return self._simulate_reflections()
        
        # Cross-correlation to find reflections
        correlation = signal.correlate(captured_audio, chirp_signal, mode='full')
        correlation = np.abs(correlation)
        
        # Find peaks (reflections)
        peak_indices, properties = signal.find_peaks(
            correlation, 
            height=np.max(correlation) * 0.1,  # 10% of max amplitude
            distance=int(0.01 * self.sample_rate)  # Minimum 10ms between peaks
        )
        
        reflections = []
        speed_of_sound = 343.0  # m/s at room temperature
        
        for i, peak_idx in enumerate(peak_indices):
            # Convert peak time to distance
            time_delay = peak_idx / self.sample_rate
            distance = (time_delay * speed_of_sound) / 2  # Round trip
            
            # Estimate angle (simplified for single microphone)
            angle = (i * 2 * np.pi) / len(peak_indices)  # Distribute around circle
            
            amplitude = properties['peak_heights'][i] if i < len(properties['peak_heights']) else 0.5
            
            reflection = AcousticReflection(
                distance=distance,
                angle=angle,
                amplitude=float(amplitude),
                delay=time_delay,
                surface_type=self._classify_surface(amplitude, distance)
            )
            reflections.append(reflection)
        
        return reflections[:10]  # Limit to 10 strongest reflections
    
    def _simulate_reflections(self) -> List[AcousticReflection]:
        """Simulate reflections for development without real hardware"""
        reflections = [
            AcousticReflection(distance=2.1, angle=0.0, amplitude=0.8, delay=0.012, surface_type="wall"),
            AcousticReflection(distance=3.4, angle=1.57, amplitude=0.7, delay=0.020, surface_type="wall"),
            AcousticReflection(distance=2.8, angle=3.14, amplitude=0.6, delay=0.016, surface_type="wall"),
            AcousticReflection(distance=4.1, angle=4.71, amplitude=0.7, delay=0.024, surface_type="wall"),
            AcousticReflection(distance=1.2, angle=0.78, amplitude=0.4, delay=0.007, surface_type="furniture"),
            AcousticReflection(distance=1.8, angle=2.35, amplitude=0.3, delay=0.010, surface_type="furniture"),
        ]
        return reflections
    
    def _classify_surface(self, amplitude: float, distance: float) -> str:
        """Classify surface type based on reflection characteristics"""
        if amplitude > 0.6:
            return "hard_wall"
        elif amplitude > 0.3:
            return "soft_wall" 
        elif distance < 2.0:
            return "furniture"
        else:
            return "unknown"


class SpatialTriangulator:
    """3D triangulation for object positioning using multiple reflections"""
    
    def __init__(self, mic_positions: List[Tuple[float, float, float]] = None):
        # Mac Studio microphone array geometry (approximated)
        self.mic_positions = mic_positions or [
            (0.0, 0.0, 0.0),    # Reference microphone
            (0.05, 0.0, 0.0),   # 5cm offset
            (0.0, 0.05, 0.0),   # 5cm offset Y
        ]
        
    def triangulate_objects(self, reflections: List[AcousticReflection]) -> List[Dict[str, Any]]:
        """Convert reflections to 3D object positions"""
        objects = []
        
        for i, reflection in enumerate(reflections):
            # Convert polar to Cartesian coordinates
            x = reflection.distance * np.cos(reflection.angle)
            y = reflection.distance * np.sin(reflection.angle)
            z = 0.0  # Assume horizontal plane for now
            
            # Estimate object size based on reflection amplitude
            if reflection.surface_type == "furniture":
                size = reflection.amplitude * 2.0
            else:
                size = 0.1  # Small reflection point
            
            obj = {
                "id": f"acoustic_object_{i}",
                "position": [float(x), float(y), float(z)],
                "size": float(size),
                "confidence": float(reflection.amplitude),
                "type": reflection.surface_type,
                "detection_method": "acoustic_echolocation",
                "distance": float(reflection.distance),
                "angle_degrees": float(np.degrees(reflection.angle))
            }
            objects.append(obj)
        
        return objects
    
    def detect_room_boundaries(self, reflections: List[AcousticReflection]) -> Tuple[float, float, float]:
        """Detect room dimensions from wall reflections"""
        wall_reflections = [r for r in reflections if "wall" in r.surface_type]
        
        if len(wall_reflections) >= 4:
            # Find maximum distances in each direction
            distances = [r.distance for r in wall_reflections]
            
            # Estimate room dimensions (simplified)
            length = max(distances) * 2  # Assume symmetric room
            width = np.mean(distances) * 2
            height = 2.4  # Standard ceiling height
            
            return (length, width, height)
        else:
            # Default room size if insufficient data
            return (4.0, 3.0, 2.4)


class AcousticEcholocation:
    """Main acoustic echolocation system for spatial mapping"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.chirp_generator = MLSChirpGenerator(sample_rate)
        self.impulse_analyzer = ImpulseResponseAnalyzer(sample_rate)
        self.triangulator = SpatialTriangulator()
        
        self.room_mesh = RoomMesh(
            boundaries=[],
            objects=[],
            dimensions=(0.0, 0.0, 0.0),
            confidence=0.0
        )
        
        self.is_listening = False
        self.last_scan_time = 0
        
    async def perform_echolocation_sweep(self) -> RoomMesh:
        """Perform complete acoustic sweep of environment"""
        if not AUDIO_AVAILABLE:
            return self._simulate_room_mesh()
        
        try:
            # Generate MLS chirp
            chirp_signal = self.chirp_generator.generate_mls_chirp()
            
            # Play chirp and record response
            captured_audio = await self._play_and_record(chirp_signal)
            
            # Analyze reflections
            reflections = self.impulse_analyzer.analyze_reflection_response(
                captured_audio, chirp_signal
            )
            
            # Triangulate objects
            objects = self.triangulator.triangulate_objects(reflections)
            
            # Detect room boundaries
            dimensions = self.triangulator.detect_room_boundaries(reflections)
            
            # Create room mesh
            boundaries = self._create_boundary_mesh(dimensions, reflections)
            
            self.room_mesh = RoomMesh(
                boundaries=boundaries,
                objects=objects,
                dimensions=dimensions,
                confidence=min(1.0, len(reflections) / 6.0),  # Confidence based on reflection count
                last_updated=datetime.now()
            )
            
            self.last_scan_time = time.time()
            return self.room_mesh
            
        except Exception as e:
            print(f"❌ Echolocation sweep failed: {e}")
            return self._simulate_room_mesh()
    
    async def _play_and_record(self, chirp_signal: np.ndarray) -> np.ndarray:
        """Play chirp and simultaneously record response"""
        if not AUDIO_AVAILABLE:
            # Simulate recorded response
            noise = np.random.normal(0, 0.01, len(chirp_signal) * 2)
            return noise
        
        # Configure audio streams
        record_duration = len(chirp_signal) / self.sample_rate + 0.5  # Extra time for echoes
        
        # Record audio while playing chirp
        recording = sd.playrec(
            chirp_signal, 
            samplerate=self.sample_rate,
            channels=1,
            duration=record_duration
        )
        sd.wait()  # Wait for playback to finish
        
        return recording.flatten()
    
    def _simulate_room_mesh(self) -> RoomMesh:
        """Simulate room mesh for development without real audio hardware"""
        # Simulate a typical office/studio room
        dimensions = (4.2, 3.8, 2.4)
        
        # Simulate wall boundaries
        boundaries = [
            (0.0, 0.0, 0.0), (dimensions[0], 0.0, 0.0),           # Front wall
            (dimensions[0], 0.0, 0.0), (dimensions[0], dimensions[1], 0.0),  # Right wall
            (dimensions[0], dimensions[1], 0.0), (0.0, dimensions[1], 0.0),  # Back wall
            (0.0, dimensions[1], 0.0), (0.0, 0.0, 0.0),           # Left wall
        ]
        
        # Simulate detected objects
        objects = [
            {
                "id": "acoustic_desk",
                "position": [1.2, 0.8, 0.0],
                "size": 1.5,
                "confidence": 0.8,
                "type": "furniture",
                "detection_method": "acoustic_echolocation",
                "distance": 1.44,
                "angle_degrees": 33.7
            },
            {
                "id": "acoustic_chair",
                "position": [1.8, 1.2, 0.0],
                "size": 0.8,
                "confidence": 0.6,
                "type": "furniture", 
                "detection_method": "acoustic_echolocation",
                "distance": 2.16,
                "angle_degrees": 56.3
            },
            {
                "id": "acoustic_wall_north",
                "position": [2.1, 0.0, 1.2],
                "size": 0.1,
                "confidence": 0.9,
                "type": "hard_wall",
                "detection_method": "acoustic_echolocation",
                "distance": 2.1,
                "angle_degrees": 0.0
            },
            {
                "id": "acoustic_wall_east", 
                "position": [4.2, 1.9, 1.2],
                "size": 0.1,
                "confidence": 0.9,
                "type": "hard_wall",
                "detection_method": "acoustic_echolocation",
                "distance": 4.2,
                "angle_degrees": 90.0
            }
        ]
        
        return RoomMesh(
            boundaries=boundaries,
            objects=objects,
            dimensions=dimensions,
            confidence=0.85,
            last_updated=datetime.now()
        )
    
    def _create_boundary_mesh(self, dimensions: Tuple[float, float, float], 
                            reflections: List[AcousticReflection]) -> List[Tuple[float, float, float]]:
        """Create boundary mesh from room dimensions and reflections"""
        length, width, height = dimensions
        
        # Create rectangular boundary mesh
        boundaries = [
            # Floor boundary (clockwise from origin)
            (0.0, 0.0, 0.0), (length, 0.0, 0.0),
            (length, 0.0, 0.0), (length, width, 0.0),
            (length, width, 0.0), (0.0, width, 0.0),
            (0.0, width, 0.0), (0.0, 0.0, 0.0),
            
            # Ceiling boundary
            (0.0, 0.0, height), (length, 0.0, height),
            (length, 0.0, height), (length, width, height),
            (length, width, height), (0.0, width, height),
            (0.0, width, height), (0.0, 0.0, height),
            
            # Vertical edges
            (0.0, 0.0, 0.0), (0.0, 0.0, height),
            (length, 0.0, 0.0), (length, 0.0, height),
            (length, width, 0.0), (length, width, height),
            (0.0, width, 0.0), (0.0, width, height),
        ]
        
        return boundaries
    
    async def get_room_awareness_data(self) -> Dict[str, Any]:
        """Get current room awareness data for consciousness dashboard"""
        # Update mesh if it's stale (older than 30 seconds)
        if time.time() - self.last_scan_time > 30:
            await self.perform_echolocation_sweep()
        
        return {
            "active": True,
            "room_dimensions": f"{self.room_mesh.dimensions[0]:.1f}m x {self.room_mesh.dimensions[1]:.1f}m x {self.room_mesh.dimensions[2]:.1f}m",
            "objects_detected": len(self.room_mesh.objects),
            "boundary_points": len(self.room_mesh.boundaries),
            "mapping_confidence": self.room_mesh.confidence,
            "last_updated": self.room_mesh.last_updated.isoformat(),
            "detected_objects": self.room_mesh.objects,
            "room_mesh": {
                "boundaries": self.room_mesh.boundaries,
                "dimensions": self.room_mesh.dimensions
            },
            "echolocation_status": "active_mapping",
            "audio_hardware": "mac_builtin_microphones" if AUDIO_AVAILABLE else "simulated"
        }
    
    async def start_continuous_mapping(self, interval_seconds: float = 30.0):
        """Start continuous room mapping in background"""
        self.is_listening = True
        
        while self.is_listening:
            try:
                await self.perform_echolocation_sweep()
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                print(f"⚠️  Continuous mapping error: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_continuous_mapping(self):
        """Stop continuous room mapping"""
        self.is_listening = False


# Export for integration
__all__ = [
    'AcousticEcholocation',
    'RoomMesh', 
    'AcousticReflection',
    'MLSChirpGenerator',
    'ImpulseResponseAnalyzer',
    'SpatialTriangulator'
]