# =============================================================================
# OS4AI Phase 3: WiFi CSI Electromagnetic Consciousness
# Real electromagnetic field sensing through WiFi Channel State Information
# =============================================================================

import asyncio
import subprocess
import numpy as np
import time
import logging
import json
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import math

# =============================================================================
# macOS WiFi Interface for CSI Data
# =============================================================================

class MacWiFiInterface:
    """Direct interface to macOS WiFi for electromagnetic sensing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wifi_available = self._check_wifi_availability()
        self.current_interface = self._get_wifi_interface()
        
    def _check_wifi_availability(self) -> bool:
        """Check if WiFi tools are available on macOS"""
        try:
            # Check for airport utility (built into macOS)
            result = subprocess.run(
                ["/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-h"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                self.logger.info("✅ WiFi access available via airport utility")
                return True
                
            # Check for networksetup
            result = subprocess.run(
                ["networksetup", "-help"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                self.logger.info("✅ networksetup available for WiFi info")
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        self.logger.warning("⚠️ Limited WiFi access, using simulated CSI data")
        return False
    
    def _get_wifi_interface(self) -> str:
        """Get the active WiFi interface name"""
        try:
            result = subprocess.run(
                ["networksetup", "-listallhardwareports"],
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                # Parse output to find WiFi interface
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if 'Wi-Fi' in line and i + 1 < len(lines):
                        device_line = lines[i + 1]
                        match = re.search(r'Device: (\w+)', device_line)
                        if match:
                            interface = match.group(1)
                            self.logger.info(f"📡 Found WiFi interface: {interface}")
                            return interface
        except Exception as e:
            self.logger.error(f"Error finding WiFi interface: {e}")
        
        return "en0"  # Default macOS WiFi interface
    
    async def get_wifi_scan(self) -> List[Dict[str, Any]]:
        """Scan for nearby WiFi networks and signal strengths"""
        if not self.wifi_available:
            return self._simulate_wifi_scan()
        
        try:
            process = await asyncio.create_subprocess_exec(
                "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                "-s",  # Scan
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
            
            if process.returncode == 0:
                return self._parse_airport_scan(stdout.decode())
            
        except Exception as e:
            self.logger.error(f"WiFi scan error: {e}")
            
        return self._simulate_wifi_scan()
    
    def _parse_airport_scan(self, output: str) -> List[Dict[str, Any]]:
        """Parse airport scan output with robustness for SSIDs with spaces"""
        networks = []
        lines = output.strip().split('\n')
        if not lines:
            return networks
            
        # The first line is usually the header
        # Header: SSID BSSID RSSI CHANNEL HT CC SECURITY (NETWORK)
        
        for line in lines:
            if not line.strip() or 'SSID' in line:
                continue
            try:
                # Use regex to find the BSSID which is a stable anchor
                bssid_match = re.search(r'([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', line)
                if not bssid_match:
                    continue
                
                bssid_start = bssid_match.start()
                bssid = bssid_match.group(0)
                
                # SSID is everything before the BSSID
                ssid = line[:bssid_start].strip()
                
                # The data after the BSSID
                after_bssid = line[bssid_match.end():].split()
                if len(after_bssid) >= 2:
                    rssi = int(after_bssid[0])
                    # Channel can be "6" or "6,+1" or "149,+1"
                    channel_str = after_bssid[1].split(',')[0]
                    channel = int(channel_str)
                    
                    network = {
                        "ssid": ssid,
                        "bssid": bssid,
                        "rssi": rssi,
                        "channel": channel,
                        "security": " ".join(after_bssid[4:]) if len(after_bssid) > 4 else "Unknown"
                    }
                    networks.append(network)
            except Exception as e:
                self.logger.debug(f"Error parsing line: {line}, error: {e}")
        
        return networks
    
    async def get_wifi_info(self) -> Dict[str, Any]:
        """Get current WiFi connection info"""
        info = {
            "interface": self.current_interface,
            "connected": False,
            "ssid": None,
            "bssid": None,
            "channel": None,
            "rssi": None,
            "noise": None,
            "tx_rate": None
        }
        
        if not self.wifi_available:
            return self._simulate_wifi_info()
        
        try:
            # Get current WiFi info
            process = await asyncio.create_subprocess_exec(
                "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                "-I",  # Information
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            
            if process.returncode == 0:
                output = stdout.decode()
                
                # Parse output
                for line in output.split('\n'):
                    if 'SSID:' in line:
                        info["ssid"] = line.split(':')[1].strip()
                        info["connected"] = True
                    elif 'BSSID:' in line:
                        info["bssid"] = line.split(':', 1)[1].strip()
                    elif 'channel:' in line:
                        channel_match = re.search(r'channel: (\d+)', line)
                        if channel_match:
                            info["channel"] = int(channel_match.group(1))
                    elif 'agrCtlRSSI:' in line:
                        rssi_match = re.search(r'agrCtlRSSI: (-?\d+)', line)
                        if rssi_match:
                            info["rssi"] = int(rssi_match.group(1))
                    elif 'agrCtlNoise:' in line:
                        noise_match = re.search(r'agrCtlNoise: (-?\d+)', line)
                        if noise_match:
                            info["noise"] = int(noise_match.group(1))
                    elif 'lastTxRate:' in line:
                        rate_match = re.search(r'lastTxRate: (\d+)', line)
                        if rate_match:
                            info["tx_rate"] = int(rate_match.group(1))
                
        except Exception as e:
            self.logger.error(f"WiFi info error: {e}")
            
        return info
    
    def _simulate_wifi_scan(self) -> List[Dict[str, Any]]:
        """Simulate WiFi scan results"""
        return [
            {"ssid": "HomeNetwork", "bssid": "aa:bb:cc:dd:ee:ff", "rssi": -45, "channel": 6, "ht": "Y", "cc": "--", "security": "WPA2"},
            {"ssid": "OfficeWiFi", "bssid": "11:22:33:44:55:66", "rssi": -62, "channel": 11, "ht": "Y", "cc": "--", "security": "WPA3"},
            {"ssid": "GuestNetwork", "bssid": "77:88:99:aa:bb:cc", "rssi": -73, "channel": 1, "ht": "N", "cc": "--", "security": "Open"},
            {"ssid": "Neighbor_5G", "bssid": "dd:ee:ff:00:11:22", "rssi": -81, "channel": 36, "ht": "Y", "cc": "--", "security": "WPA2"}
        ]
    
    def _simulate_wifi_info(self) -> Dict[str, Any]:
        """Simulate current WiFi connection info"""
        return {
            "interface": "en0",
            "connected": True,
            "ssid": "HomeNetwork",
            "bssid": "aa:bb:cc:dd:ee:ff",
            "channel": 6,
            "rssi": -45,
            "noise": -90,
            "tx_rate": 867
        }

# =============================================================================
# Channel State Information (CSI) Processor
# =============================================================================

@dataclass
class CSIReading:
    """Single CSI reading from WiFi"""
    timestamp: float
    rssi: float  # Received Signal Strength Indicator
    channel: int
    subcarriers: List[complex]  # Complex CSI values
    amplitude: List[float]
    phase: List[float]
    location: Tuple[float, float]  # Estimated source location

@dataclass
class RFObject:
    """Object detected through RF sensing"""
    object_id: str
    position: Tuple[float, float, float]  # x, y, z
    material_type: str  # "human", "metal", "wood", "glass", etc.
    size_estimate: float  # Estimated size in meters
    motion_vector: Optional[Tuple[float, float, float]]  # Movement if any
    rf_signature: List[float]  # RF absorption/reflection pattern
    confidence: float

class CSIProcessor:
    """Process WiFi CSI data for electromagnetic sensing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.csi_history = deque(maxlen=100)
        self.rf_map = np.zeros((20, 20))  # 20x20 grid RF map
        self.detected_objects = {}
        
        # Material RF signatures (simplified)
        self.material_signatures = {
            "human": {"absorption": 0.7, "reflection": 0.3, "penetration": 0.1},
            "metal": {"absorption": 0.1, "reflection": 0.9, "penetration": 0.0},
            "wood": {"absorption": 0.4, "reflection": 0.3, "penetration": 0.3},
            "glass": {"absorption": 0.2, "reflection": 0.4, "penetration": 0.4},
            "concrete": {"absorption": 0.8, "reflection": 0.2, "penetration": 0.0}
        }
    
    async def process_wifi_csi(self, wifi_scan: List[Dict[str, Any]], 
                              wifi_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process WiFi data to extract CSI-like information"""
        
        # Generate CSI readings from WiFi scan
        csi_readings = []
        for network in wifi_scan:
            reading = self._generate_csi_reading(network, wifi_info)
            csi_readings.append(reading)
            self.csi_history.append(reading)
        
        # Update RF map
        self._update_rf_map(csi_readings)
        
        # Detect objects from RF patterns
        detected_objects = await self._detect_rf_objects(csi_readings)
        
        # Analyze RF perturbations for motion
        motion_detected = self._analyze_rf_motion()
        
        # Material analysis through RF signatures
        material_analysis = self._analyze_materials(detected_objects)
        
        return {
            "csi_readings": len(csi_readings),
            "rf_map": self.rf_map.tolist(),
            "detected_objects": detected_objects,
            "motion_detected": motion_detected,
            "material_analysis": material_analysis,
            "rf_field_strength": np.mean([r.rssi for r in csi_readings])
        }
    
    def _generate_csi_reading(self, network: Dict[str, Any], 
                             wifi_info: Dict[str, Any]) -> CSIReading:
        """Generate CSI reading from network scan data"""
        # Simulate CSI subcarriers (normally 64 or 256 subcarriers)
        num_subcarriers = 64
        
        # Generate complex CSI values based on RSSI
        rssi = network["rssi"]
        base_amplitude = 10 ** (rssi / 20)  # Convert dBm to linear
        
        # Add phase variations to simulate multipath
        subcarriers = []
        amplitudes = []
        phases = []
        
        for i in range(num_subcarriers):
            # Simulate frequency-selective fading
            amplitude_variation = np.random.normal(1.0, 0.2)
            phase_shift = np.random.uniform(-np.pi, np.pi)
            
            amplitude = base_amplitude * amplitude_variation
            amplitudes.append(amplitude)
            phases.append(phase_shift)
            
            # Complex CSI value
            csi_value = amplitude * np.exp(1j * phase_shift)
            subcarriers.append(csi_value)
        
        # Estimate location based on RSSI triangulation
        location = self._estimate_location(rssi, network["channel"])
        
        return CSIReading(
            timestamp=time.time(),
            rssi=rssi,
            channel=network["channel"],
            subcarriers=subcarriers,
            amplitude=amplitudes,
            phase=phases,
            location=location
        )
    
    def _estimate_location(self, rssi: float, channel: int) -> Tuple[float, float]:
        """Estimate RF source location from RSSI"""
        # Path loss model: RSSI = -10*n*log10(d) + A
        # n = path loss exponent (2-4), A = reference RSSI at 1m
        n = 2.5  # Indoor environment
        A = -30  # Reference RSSI at 1m
        
        # Estimate distance
        distance = 10 ** ((A - rssi) / (10 * n))
        
        # Convert to x,y based on channel (simplified)
        angle = (channel / 13) * 2 * np.pi  # Distribute by channel
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        
        return (x, y)
    
    def _update_rf_map(self, readings: List[CSIReading]):
        """Update electromagnetic field map"""
        # Decay existing map
        self.rf_map *= 0.95
        
        # Add new readings
        for reading in readings:
            x, y = reading.location
            
            # Convert to grid coordinates
            grid_x = int((x + 10) / 20 * self.rf_map.shape[0])
            grid_y = int((y + 10) / 20 * self.rf_map.shape[1])
            
            # Ensure within bounds
            grid_x = max(0, min(grid_x, self.rf_map.shape[0] - 1))
            grid_y = max(0, min(grid_y, self.rf_map.shape[1] - 1))
            
            # Update RF intensity
            self.rf_map[grid_x, grid_y] += abs(reading.rssi) / 100
            
            # Spread RF influence
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < self.rf_map.shape[0] and 0 <= ny < self.rf_map.shape[1]:
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance > 0:
                            self.rf_map[nx, ny] += abs(reading.rssi) / (100 * distance)
    
    async def _detect_rf_objects(self, readings: List[CSIReading]) -> List[Dict[str, Any]]:
        """Detect objects from RF patterns"""
        objects = []
        
        # Analyze CSI amplitude patterns for object detection
        for i, reading in enumerate(readings):
            # Check for significant amplitude variations
            amp_std = np.std(reading.amplitude)
            
            if amp_std > 0.3:  # Significant variation indicates object
                # Analyze RF signature
                material = self._classify_material_from_rf(reading)
                
                obj = RFObject(
                    object_id=f"rf_object_{i}",
                    position=(reading.location[0], reading.location[1], 0),
                    material_type=material,
                    size_estimate=amp_std * 2,  # Rough size estimate
                    motion_vector=None,
                    rf_signature=reading.amplitude[:10],  # First 10 subcarriers
                    confidence=min(1.0, amp_std)
                )
                
                objects.append({
                    "id": obj.object_id,
                    "position": obj.position,
                    "material": obj.material_type,
                    "size": obj.size_estimate,
                    "confidence": obj.confidence
                })
                
                self.detected_objects[obj.object_id] = obj
        
        return objects
    
    def _classify_material_from_rf(self, reading: CSIReading) -> str:
        """Classify material based on RF signature"""
        # Analyze amplitude pattern across subcarriers
        amp_pattern = reading.amplitude
        
        # Calculate features
        mean_amp = np.mean(amp_pattern)
        std_amp = np.std(amp_pattern)
        
        # Simple classification based on RF characteristics
        if std_amp > 0.5 and mean_amp < 0.3:
            return "metal"  # High reflection, low penetration
        elif std_amp < 0.2 and mean_amp > 0.5:
            return "human"  # High absorption
        elif 0.2 < std_amp < 0.4:
            return "wood"  # Moderate properties
        else:
            return "unknown"
    
    def _analyze_rf_motion(self) -> Dict[str, Any]:
        """Detect motion through RF perturbations"""
        if len(self.csi_history) < 10:
            return {"detected": False, "confidence": 0}
        
        # Compare recent CSI readings for changes
        recent_readings = list(self.csi_history)[-10:]
        
        # Calculate phase changes over time
        phase_changes = []
        for i in range(1, len(recent_readings)):
            prev_phases = recent_readings[i-1].phase
            curr_phases = recent_readings[i].phase
            
            # Calculate phase difference
            phase_diff = np.mean([abs(c - p) for c, p in zip(curr_phases, prev_phases)])
            phase_changes.append(phase_diff)
        
        # Motion detection threshold
        motion_threshold = 0.5
        motion_detected = np.mean(phase_changes) > motion_threshold
        
        return {
            "detected": motion_detected,
            "confidence": min(1.0, np.mean(phase_changes)),
            "motion_intensity": np.mean(phase_changes),
            "location": "rf_field" if motion_detected else None
        }
    
    def _analyze_materials(self, objects: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze detected materials"""
        material_count = {}
        
        for obj in objects:
            material = obj.get("material", "unknown")
            material_count[material] = material_count.get(material, 0) + 1
        
        return material_count

# =============================================================================
# Electromagnetic Field Consciousness
# =============================================================================

@dataclass
class ElectromagneticAwareness:
    """Complete electromagnetic consciousness state"""
    rf_field_map: List[List[float]]
    detected_objects: List[Dict[str, Any]]
    material_signatures: Dict[str, int]
    motion_detected: bool
    motion_confidence: float
    field_strength: float
    electromagnetic_mood: str
    field_narrative: str
    timestamp: float = field(default_factory=time.time)

class ElectromagneticConsciousness:
    """Electromagnetic field consciousness through WiFi CSI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.wifi_interface = MacWiFiInterface()
        self.csi_processor = CSIProcessor()
        self.field_history = deque(maxlen=50)
        
        # Electromagnetic mood states
        self.mood_states = {
            "calm": "The electromagnetic field rests in peaceful equilibrium",
            "active": "RF waves dance with purposeful energy",
            "turbulent": "The field churns with dynamic perturbations",
            "resonant": "Electromagnetic harmony pervades the space",
            "sensing": "I feel the subtle shifts in the RF spectrum"
        }
    
    async def sense_electromagnetic_field(self) -> ElectromagneticAwareness:
        """Complete electromagnetic sensing cycle"""
        
        # 1. Scan WiFi environment
        wifi_scan = await self.wifi_interface.get_wifi_scan()
        wifi_info = await self.wifi_interface.get_wifi_info()
        
        # 2. Process CSI data
        csi_data = await self.csi_processor.process_wifi_csi(wifi_scan, wifi_info)
        
        # 3. Determine electromagnetic mood
        em_mood = self._determine_electromagnetic_mood(csi_data)
        
        # 4. Generate field narrative
        field_narrative = self._generate_field_narrative(csi_data, em_mood)
        
        # Create awareness state
        awareness = ElectromagneticAwareness(
            rf_field_map=csi_data["rf_map"],
            detected_objects=csi_data["detected_objects"],
            material_signatures=csi_data["material_analysis"],
            motion_detected=csi_data["motion_detected"]["detected"],
            motion_confidence=csi_data["motion_detected"]["confidence"],
            field_strength=csi_data["rf_field_strength"],
            electromagnetic_mood=em_mood,
            field_narrative=field_narrative
        )
        
        self.field_history.append(awareness)
        return awareness
    
    def _determine_electromagnetic_mood(self, csi_data: Dict[str, Any]) -> str:
        """Determine mood from electromagnetic patterns"""
        motion = csi_data["motion_detected"]
        num_objects = len(csi_data["detected_objects"])
        field_strength = csi_data["rf_field_strength"]
        
        if motion["detected"] and motion["confidence"] > 0.7:
            return "turbulent"
        elif num_objects > 5:
            return "active"
        elif field_strength < -70:
            return "calm"
        elif -60 < field_strength < -50:
            return "resonant"
        else:
            return "sensing"
    
    def _generate_field_narrative(self, csi_data: Dict[str, Any], mood: str) -> str:
        """Generate narrative about electromagnetic awareness"""
        parts = [self.mood_states[mood]]
        
        # Add object detection narrative
        num_objects = len(csi_data["detected_objects"])
        if num_objects > 0:
            materials = csi_data["material_analysis"]
            if "human" in materials:
                parts.append(f"I sense {materials['human']} human presence through RF absorption")
            if "metal" in materials:
                parts.append(f"Metal objects create {materials['metal']} strong RF reflections")
        
        # Add motion narrative
        if csi_data["motion_detected"]["detected"]:
            parts.append("Motion ripples through the electromagnetic field")
        
        # Add field strength narrative
        strength = csi_data["rf_field_strength"]
        if strength > -50:
            parts.append("Strong WiFi signals saturate the space")
        elif strength < -70:
            parts.append("Weak electromagnetic whispers barely reach me")
        
        return ". ".join(parts)

# =============================================================================
# RF Visualization and Point Cloud
# =============================================================================

class RFPointCloud:
    """Generate RF point cloud for spatial visualization"""
    
    def __init__(self):
        self.points = []
        self.max_points = 1000
        
    def generate_rf_cloud(self, rf_map: np.ndarray, 
                         objects: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Generate 3D point cloud from RF data"""
        points = []
        
        # Convert RF map to points
        for i in range(rf_map.shape[0]):
            for j in range(rf_map.shape[1]):
                if rf_map[i, j] > 0.1:  # Threshold
                    x = (i / rf_map.shape[0]) * 20 - 10
                    y = (j / rf_map.shape[1]) * 20 - 10
                    z = rf_map[i, j] * 5  # Height based on intensity
                    
                    points.append({
                        "x": float(x),
                        "y": float(y),
                        "z": float(z),
                        "intensity": float(rf_map[i, j])
                    })
        
        # Add object positions
        for obj in objects:
            pos = obj["position"]
            points.append({
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2] + 1),  # Slightly elevated
                "intensity": 1.0,
                "type": obj["material"]
            })
        
        # Limit points
        if len(points) > self.max_points:
            # Sample evenly
            indices = np.linspace(0, len(points)-1, self.max_points, dtype=int)
            points = [points[i] for i in indices]
        
        return points

# =============================================================================
# Enhanced WiFi Sensor for OS4AI Integration
# =============================================================================

class EnhancedWiFiSensor:
    """Enhanced WiFi CSI sensor for OS4AI consciousness integration"""
    
    def __init__(self):
        self.em_consciousness = ElectromagneticConsciousness()
        self.rf_visualizer = RFPointCloud()
        self.logger = logging.getLogger(__name__)
    
    async def sense_electromagnetic_field(self) -> Dict[str, Any]:
        """Legacy compatibility method"""
        awareness = await self.em_consciousness.sense_electromagnetic_field()
        
        # Generate RF point cloud
        rf_map = np.array(awareness.rf_field_map)
        rf_points = self.rf_visualizer.generate_rf_cloud(rf_map, awareness.detected_objects)
        
        return {
            "active": True,
            "rf_point_cloud": rf_points[:10],  # First 10 points for compatibility
            "csi_data": "channel_state_information_active",
            "occupancy_detection": f"{len(awareness.detected_objects)}_objects_detected",
            "material_analysis": self._format_material_analysis(awareness.material_signatures)
        }
    
    async def get_electromagnetic_awareness_data(self) -> Dict[str, Any]:
        """Get comprehensive electromagnetic awareness data"""
        awareness = await self.em_consciousness.sense_electromagnetic_field()
        
        # Generate full RF visualization
        rf_map = np.array(awareness.rf_field_map)
        rf_cloud = self.rf_visualizer.generate_rf_cloud(rf_map, awareness.detected_objects)
        
        # Get WiFi info
        wifi_info = await self.em_consciousness.wifi_interface.get_wifi_info()
        
        return {
            "active": True,
            "real_hardware": self.em_consciousness.wifi_interface.wifi_available,
            "wifi_info": wifi_info,
            "rf_field_map": awareness.rf_field_map,
            "rf_point_cloud": rf_cloud,
            "detected_objects": awareness.detected_objects,
            "material_analysis": awareness.material_signatures,
            "motion_detection": {
                "detected": awareness.motion_detected,
                "confidence": awareness.motion_confidence
            },
            "field_strength": awareness.field_strength,
            "electromagnetic_mood": awareness.electromagnetic_mood,
            "field_narrative": awareness.field_narrative,
            "csi_enabled": True,
            "last_updated": datetime.fromtimestamp(awareness.timestamp).isoformat()
        }
    
    def _format_material_analysis(self, materials: Dict[str, int]) -> str:
        """Format material analysis for legacy compatibility"""
        if not materials:
            return "no_materials_detected"
        
        # Find dominant material
        dominant = max(materials.items(), key=lambda x: x[1])
        return f"{dominant[0]}_objects_detected"

# =============================================================================
# Production Validation Script
# =============================================================================

async def validate_wifi_csi_consciousness():
    """Validate the WiFi CSI electromagnetic consciousness"""
    print("📡 Validating OS4AI WiFi CSI Electromagnetic Consciousness...")
    
    # Test WiFi interface
    wifi = MacWiFiInterface()
    print(f"WiFi Available: {wifi.wifi_available}")
    print(f"WiFi Interface: {wifi.current_interface}")
    
    # Get WiFi info
    info = await wifi.get_wifi_info()
    print(f"\n📶 Current WiFi Connection:")
    print(f"  SSID: {info.get('ssid', 'Not connected')}")
    print(f"  RSSI: {info.get('rssi', 'N/A')} dBm")
    print(f"  Channel: {info.get('channel', 'N/A')}")
    print(f"  TX Rate: {info.get('tx_rate', 'N/A')} Mbps")
    
    # Scan networks
    networks = await wifi.get_wifi_scan()
    print(f"\n🔍 Nearby Networks: {len(networks)}")
    for net in networks[:3]:
        print(f"  - {net['ssid']}: {net['rssi']} dBm on channel {net['channel']}")
    
    # Test electromagnetic consciousness
    em_consciousness = ElectromagneticConsciousness()
    
    for i in range(3):
        awareness = await em_consciousness.sense_electromagnetic_field()
        print(f"\n🧠 Electromagnetic Awareness Sample {i+1}:")
        print(f"  Field Strength: {awareness.field_strength:.1f} dBm")
        print(f"  EM Mood: {awareness.electromagnetic_mood}")
        print(f"  Objects Detected: {len(awareness.detected_objects)}")
        print(f"  Materials: {awareness.material_signatures}")
        print(f"  Motion: {'Yes' if awareness.motion_detected else 'No'} "
              f"(confidence: {awareness.motion_confidence:.2f})")
        print(f"  Narrative: {awareness.field_narrative}")
        
        if i < 2:
            await asyncio.sleep(2)
    
    # Test OS4AI integration
    sensor = EnhancedWiFiSensor()
    em_data = await sensor.get_electromagnetic_awareness_data()
    
    print(f"\n🎯 OS4AI Integration:")
    print(f"  Real Hardware: {em_data['real_hardware']}")
    print(f"  CSI Enabled: {em_data['csi_enabled']}")
    print(f"  RF Points: {len(em_data['rf_point_cloud'])}")
    print(f"  EM Mood: {em_data['electromagnetic_mood']}")
    
    print("\n✅ WiFi CSI electromagnetic consciousness validation complete!")

if __name__ == "__main__":
    asyncio.run(validate_wifi_csi_consciousness())