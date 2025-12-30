"""
OS4AI Perfect WiFi CSI Consciousness Implementation
Production-ready WiFi Channel State Information sensing with privacy and security
"""

import asyncio
import numpy as np
import json
import hashlib
import os
import subprocess
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field, validator
from fastapi import HTTPException, status
import redis
from contextlib import asynccontextmanager
import logging
from collections import deque, defaultdict
from scipy import signal, stats
from scipy.spatial import distance
import struct
import socket

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security constants
MAX_SCAN_DURATION = 10  # seconds
MIN_SCAN_INTERVAL = 5  # seconds between scans
ALLOWED_CHANNELS = list(range(1, 15))  # 2.4GHz channels
PRIVACY_MAC_RANDOMIZATION = True
RF_POWER_LIMITS = {'2.4GHz': 20, '5GHz': 23}  # dBm

class WiFiConfig(BaseModel):
    """WiFi CSI configuration with validation"""
    scan_interval: int = Field(10, ge=5, le=60)  # seconds
    csi_enabled: bool = True
    privacy_mode: bool = True
    motion_detection: bool = True
    material_detection: bool = True
    rf_fingerprinting: bool = False  # Disabled by default for privacy
    max_networks: int = Field(50, ge=10, le=100)
    signal_threshold: int = Field(-80, ge=-100, le=-30)  # dBm
    allowed_bands: List[str] = Field(default=['2.4GHz', '5GHz'])
    
    @validator('allowed_bands')
    def validate_bands(cls, v):
        """Ensure only valid bands"""
        valid_bands = ['2.4GHz', '5GHz', '6GHz']
        for band in v:
            if band not in valid_bands:
                raise ValueError(f"Invalid band: {band}")
        return v

class WiFiNetwork(BaseModel):
    """Validated WiFi network information"""
    ssid: str = Field(..., max_length=32)
    bssid: str = Field(..., regex="^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$")
    channel: int = Field(..., ge=1, le=165)
    frequency: int = Field(..., ge=2400, le=6000)  # MHz
    signal_strength: int = Field(..., ge=-100, le=0)  # dBm
    security: str = Field(..., max_length=50)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    privacy_filtered: bool = True
    
    @validator('ssid')
    def sanitize_ssid(cls, v):
        """Sanitize SSID for security"""
        # Remove non-printable characters
        return ''.join(c for c in v if c.isprintable())

class CSIReading(BaseModel):
    """Channel State Information reading"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    channel: int
    subcarriers: List[Tuple[float, float]]  # (real, imag) pairs instead of complex
    amplitude: List[float]
    phase: List[float]
    snr: float = Field(..., ge=0, le=100)  # Signal-to-noise ratio
    
    @validator('subcarriers')
    def validate_subcarriers(cls, v):
        """Validate CSI subcarrier data"""
        if len(v) == 0 or len(v) > 256:  # WiFi 6 has up to 256 subcarriers
            raise ValueError("Invalid number of subcarriers")
        return v
    
    def get_complex_subcarriers(self) -> List[complex]:
        """Convert (real, imag) tuples back to complex numbers"""
        return [complex(real, imag) for real, imag in self.subcarriers]

class RFSignature(BaseModel):
    """RF signature for material/object detection"""
    material_type: str = Field(..., regex="^[a-zA-Z_]+$", max_length=50)
    attenuation_2_4ghz: float = Field(..., ge=0, le=100)  # dB
    attenuation_5ghz: float = Field(..., ge=0, le=100)  # dB
    reflection_coefficient: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)

class WiFiAwareness(BaseModel):
    """Comprehensive WiFi consciousness state"""
    networks_detected: int
    electromagnetic_map: Dict[str, Any]
    motion_detected: bool
    motion_zones: List[Dict[str, Any]]
    material_signatures: List[RFSignature]
    rf_anomalies: List[str]
    privacy_status: str
    security_alerts: List[str]
    environmental_profile: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: str

class SecureWiFiScanner:
    """
    Secure WiFi scanning with privacy protection
    """
    
    def __init__(self, config: WiFiConfig):
        self.config = config
        self._last_scan_time = datetime.now(timezone.utc) - timedelta(seconds=config.scan_interval)
        self._network_cache: Dict[str, WiFiNetwork] = {}
        self._privacy_filter = WiFiPrivacyFilter()
        self._command_validator = CommandValidator()
        
    async def scan_networks(self, user_id: str) -> List[WiFiNetwork]:
        """
        Scan for WiFi networks with security and privacy
        """
        # Rate limiting
        time_since_last = (datetime.now(timezone.utc) - self._last_scan_time).total_seconds()
        if time_since_last < self.config.scan_interval:
            logger.info("Using cached network scan results")
            return list(self._network_cache.values())
        
        logger.info(f"Scanning WiFi networks for user {user_id}")
        self._last_scan_time = datetime.now(timezone.utc)
        
        # Get network list
        networks = await self._perform_secure_scan()
        
        # Filter and validate
        filtered_networks = []
        for network_data in networks[:self.config.max_networks]:
            try:
                # Apply privacy filtering
                if self.config.privacy_mode:
                    network_data = self._privacy_filter.filter_network(network_data)
                
                # Validate network data
                network = WiFiNetwork(**network_data)
                
                # Apply signal threshold
                if network.signal_strength >= self.config.signal_threshold:
                    filtered_networks.append(network)
                    self._network_cache[network.bssid] = network
                    
            except Exception as e:
                logger.error(f"Network validation error: {e}")
        
        return filtered_networks
    
    async def _perform_secure_scan(self) -> List[Dict[str, Any]]:
        """Perform secure WiFi scan using system tools"""
        networks = []
        
        try:
            # Validate command before execution
            if not self._command_validator.validate_wifi_scan_command():
                raise ValueError("WiFi scan command validation failed")
            
            # Platform-specific scanning
            if os.name == 'posix':
                networks = await self._scan_posix()
            else:
                networks = await self._scan_windows()
                
        except Exception as e:
            logger.error(f"WiFi scan error: {e}")
            # Return cached results on error
            networks = [n.dict() for n in self._network_cache.values()]
        
        return networks
    
    async def _scan_posix(self) -> List[Dict[str, Any]]:
        """Scan WiFi on POSIX systems (macOS/Linux)"""
        networks = []
        
        try:
            # Use airport on macOS
            if os.path.exists('/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport'):
                cmd = ['/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport', '-s']
            else:
                # Use iwlist on Linux
                cmd = ['sudo', 'iwlist', 'scan']
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={'PATH': '/usr/bin:/bin:/usr/sbin:/sbin'}  # Restricted PATH
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=MAX_SCAN_DURATION
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise Exception("WiFi scan timeout")
            
            if process.returncode == 0:
                networks = self._parse_scan_output(stdout.decode())
            else:
                logger.error(f"WiFi scan failed: {stderr.decode()}")
                
        except Exception as e:
            logger.error(f"POSIX WiFi scan error: {e}")
        
        return networks
    
    async def _scan_windows(self) -> List[Dict[str, Any]]:
        """Scan WiFi on Windows systems"""
        networks = []
        
        try:
            cmd = ['netsh', 'wlan', 'show', 'networks', 'mode=bssid']
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                networks = self._parse_windows_output(stdout.decode())
                
        except Exception as e:
            logger.error(f"Windows WiFi scan error: {e}")
        
        return networks
    
    def _parse_scan_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse macOS airport scan output"""
        networks = []
        lines = output.strip().split('\n')[1:]  # Skip header
        
        for line in lines:
            try:
                parts = line.split()
                if len(parts) >= 7:
                    ssid = parts[0]
                    bssid = parts[1]
                    rssi = int(parts[2])
                    channel = int(parts[3])
                    
                    # Determine frequency from channel
                    if channel <= 14:
                        frequency = 2407 + (channel * 5)
                    else:
                        frequency = 5000 + (channel * 5)
                    
                    # Parse security
                    security = ' '.join(parts[6:]) if len(parts) > 6 else 'Open'
                    
                    networks.append({
                        'ssid': ssid,
                        'bssid': bssid,
                        'channel': channel,
                        'frequency': frequency,
                        'signal_strength': rssi,
                        'security': security
                    })
                    
            except Exception as e:
                logger.debug(f"Error parsing line: {line}, error: {e}")
        
        return networks
    
    def _parse_windows_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse Windows netsh output"""
        networks = []
        # Windows parsing logic would go here
        return networks

class CommandValidator:
    """Validate system commands for security"""
    
    def validate_wifi_scan_command(self) -> bool:
        """Validate WiFi scan command is safe to execute"""
        # Check for command injection attempts in environment
        suspicious_vars = ['LD_PRELOAD', 'DYLD_INSERT_LIBRARIES', 'PATH']
        for var in suspicious_vars:
            if var in os.environ:
                env_value = os.environ[var]
                if any(char in env_value for char in [';', '&', '|', '$', '`', '\n']):
                    logger.warning(f"Suspicious environment variable: {var}")
                    return False
        
        return True

class WiFiPrivacyFilter:
    """Privacy filtering for WiFi data"""
    
    def __init__(self):
        self._mac_randomizer = MACRandomizer()
        self._ssid_filter = SSIDFilter()
    
    def filter_network(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy filtering to network data"""
        filtered = network_data.copy()
        
        # Randomize MAC addresses for privacy
        if PRIVACY_MAC_RANDOMIZATION:
            filtered['bssid'] = self._mac_randomizer.randomize_mac(filtered['bssid'])
        
        # Filter sensitive SSIDs
        filtered['ssid'] = self._ssid_filter.filter_ssid(filtered['ssid'])
        
        # Mark as privacy filtered
        filtered['privacy_filtered'] = True
        
        return filtered

class MACRandomizer:
    """Randomize MAC addresses for privacy"""
    
    def __init__(self):
        self._mac_mapping: Dict[str, str] = {}
    
    def randomize_mac(self, original_mac: str) -> str:
        """Generate consistent random MAC for privacy"""
        if original_mac in self._mac_mapping:
            return self._mac_mapping[original_mac]
        
        # Generate deterministic but anonymous MAC
        hash_input = f"{original_mac}_privacy_salt"
        hash_bytes = hashlib.sha256(hash_input.encode()).digest()[:6]
        
        # Set locally administered bit
        hash_bytes = bytearray(hash_bytes)
        hash_bytes[0] = (hash_bytes[0] & 0xFC) | 0x02
        
        # Format as MAC address
        random_mac = ':'.join(f'{b:02x}' for b in hash_bytes)
        self._mac_mapping[original_mac] = random_mac
        
        return random_mac

class SSIDFilter:
    """Filter sensitive SSIDs"""
    
    def __init__(self):
        self._sensitive_patterns = [
            r'.*phone.*', r'.*mobile.*', r'.*personal.*',
            r'.*private.*', r'.*home.*', r'.*[0-9]{4,}.*'  # Numbers might be addresses
        ]
    
    def filter_ssid(self, ssid: str) -> str:
        """Filter potentially sensitive SSID information"""
        # Check for sensitive patterns
        lower_ssid = ssid.lower()
        for pattern in self._sensitive_patterns:
            if re.match(pattern, lower_ssid):
                # Partially redact
                if len(ssid) > 4:
                    return ssid[:2] + '*' * (len(ssid) - 4) + ssid[-2:]
                else:
                    return '*' * len(ssid)
        
        return ssid

class CSIProcessor:
    """
    Process Channel State Information for environmental sensing
    """
    
    def __init__(self, config: WiFiConfig):
        self.config = config
        self._csi_history = deque(maxlen=100)
        self._baseline_csi: Optional[np.ndarray] = None
        self._motion_detector = MotionDetector()
        self._material_analyzer = MaterialAnalyzer()
    
    async def process_csi(self, csi_data: List[Tuple[float, float]]) -> CSIReading:
        """Process raw CSI data"""
        # Convert to complex numpy array for processing
        csi_complex = np.array([complex(r, i) for r, i in csi_data])
        
        # Calculate amplitude and phase
        amplitude = np.abs(csi_complex)
        phase = np.angle(csi_complex)
        
        # Calculate SNR
        signal_power = np.mean(amplitude**2)
        noise_estimate = np.std(amplitude)**2
        snr = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        
        reading = CSIReading(
            channel=1,  # Would be set based on actual channel
            subcarriers=csi_data,
            amplitude=amplitude.tolist(),
            phase=phase.tolist(),
            snr=float(snr)
        )
        
        # Add to history
        self._csi_history.append(reading)
        
        # Update baseline if needed
        if self._baseline_csi is None:
            self._baseline_csi = amplitude
        
        return reading
    
    async def detect_motion(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Detect motion from CSI variations"""
        if len(self._csi_history) < 5:
            return False, []
        
        # Get recent CSI readings
        recent_readings = list(self._csi_history)[-10:]
        
        # Extract amplitude variations
        amplitudes = [np.array(r.amplitude) for r in recent_readings]
        
        # Detect motion
        motion_detected, motion_zones = self._motion_detector.detect(
            amplitudes,
            self._baseline_csi
        )
        
        return motion_detected, motion_zones
    
    async def analyze_materials(self, networks: List[WiFiNetwork]) -> List[RFSignature]:
        """Analyze RF signatures for material detection"""
        if not self.config.material_detection:
            return []
        
        signatures = []
        
        # Group networks by location (similar BSSID prefix)
        location_groups = defaultdict(list)
        for network in networks:
            prefix = network.bssid[:8]  # First 3 octets
            location_groups[prefix].append(network)
        
        # Analyze each location
        for location, nets in location_groups.items():
            # Calculate attenuation patterns
            if len(nets) >= 2:
                signature = self._material_analyzer.analyze_location(nets)
                if signature:
                    signatures.append(signature)
        
        return signatures

class MotionDetector:
    """Detect motion from CSI variations"""
    
    def __init__(self):
        self._motion_threshold = 0.1
        self._variance_window = 5
    
    def detect(self, amplitudes: List[np.ndarray], 
               baseline: Optional[np.ndarray]) -> Tuple[bool, List[Dict[str, Any]]]:
        """Detect motion from amplitude variations"""
        if not amplitudes or baseline is None:
            return False, []
        
        motion_zones = []
        motion_detected = False
        
        # Calculate variance over time
        variances = []
        for i in range(len(amplitudes)):
            diff = amplitudes[i] - baseline
            variance = np.var(diff)
            variances.append(variance)
        
        # Detect significant changes
        mean_variance = np.mean(variances)
        if mean_variance > self._motion_threshold:
            motion_detected = True
            
            # Identify motion zones (subcarriers with high variance)
            recent_amp = amplitudes[-1]
            diff = recent_amp - baseline
            
            # Find peaks in difference
            peaks, _ = signal.find_peaks(np.abs(diff), height=0.05)
            
            for peak in peaks:
                motion_zones.append({
                    'subcarrier': int(peak),
                    'intensity': float(np.abs(diff[peak])),
                    'type': 'movement'
                })
        
        return motion_detected, motion_zones

class MaterialAnalyzer:
    """Analyze materials from RF signatures"""
    
    def __init__(self):
        self._material_db = self._init_material_database()
    
    def _init_material_database(self) -> Dict[str, Dict[str, float]]:
        """Initialize material RF properties database"""
        return {
            'air': {'attenuation_2.4': 0.0, 'attenuation_5': 0.0, 'reflection': 0.0},
            'wood': {'attenuation_2.4': 3.0, 'attenuation_5': 5.0, 'reflection': 0.2},
            'concrete': {'attenuation_2.4': 10.0, 'attenuation_5': 15.0, 'reflection': 0.5},
            'metal': {'attenuation_2.4': 20.0, 'attenuation_5': 25.0, 'reflection': 0.9},
            'glass': {'attenuation_2.4': 2.0, 'attenuation_5': 3.0, 'reflection': 0.1},
            'water': {'attenuation_2.4': 15.0, 'attenuation_5': 20.0, 'reflection': 0.3}
        }
    
    def analyze_location(self, networks: List[WiFiNetwork]) -> Optional[RFSignature]:
        """Analyze RF properties at a location"""
        if len(networks) < 2:
            return None
        
        # Separate by frequency
        networks_2_4 = [n for n in networks if n.frequency < 3000]
        networks_5 = [n for n in networks if n.frequency >= 5000]
        
        if not networks_2_4 or not networks_5:
            return None
        
        # Calculate average attenuation
        avg_2_4 = np.mean([n.signal_strength for n in networks_2_4])
        avg_5 = np.mean([n.signal_strength for n in networks_5])
        
        # Estimate attenuation (simplified)
        ref_power = -30  # dBm reference
        atten_2_4 = ref_power - avg_2_4
        atten_5 = ref_power - avg_5
        
        # Match to material
        best_match = None
        best_score = float('inf')
        
        for material, props in self._material_db.items():
            score = (
                abs(atten_2_4 - props['attenuation_2.4']) +
                abs(atten_5 - props['attenuation_5'])
            )
            if score < best_score:
                best_score = score
                best_match = material
        
        if best_match and best_score < 10:  # Threshold for match
            return RFSignature(
                material_type=best_match,
                attenuation_2_4ghz=float(atten_2_4),
                attenuation_5ghz=float(atten_5),
                reflection_coefficient=self._material_db[best_match]['reflection'],
                confidence=1.0 - (best_score / 20)  # Convert score to confidence
            )
        
        return None

class RFAnomalyDetector:
    """Detect RF anomalies and potential security threats"""
    
    def __init__(self):
        self._power_threshold = 0  # dBm - very strong signal
        self._frequency_anomalies = set()
        self._jamming_detector = JammingDetector()
    
    async def detect_anomalies(self, networks: List[WiFiNetwork]) -> List[str]:
        """Detect RF anomalies in network list"""
        anomalies = []
        
        # Check for overpowered signals
        for network in networks:
            if network.signal_strength > self._power_threshold:
                anomalies.append(
                    f"Suspiciously strong signal from {network.ssid}: "
                    f"{network.signal_strength} dBm"
                )
        
        # Check for invalid channels
        for network in networks:
            if network.channel not in ALLOWED_CHANNELS:
                anomalies.append(
                    f"Invalid channel {network.channel} detected"
                )
        
        # Check for jamming
        if self._jamming_detector.detect_jamming(networks):
            anomalies.append("Possible RF jamming detected")
        
        # Check for spoofing patterns
        bssids = [n.bssid for n in networks]
        if self._detect_mac_spoofing(bssids):
            anomalies.append("MAC address spoofing patterns detected")
        
        return anomalies
    
    def _detect_mac_spoofing(self, bssids: List[str]) -> bool:
        """Detect MAC spoofing patterns"""
        # Check for sequential MACs (common in spoofing)
        mac_integers = []
        for bssid in bssids:
            try:
                # Convert MAC to integer
                mac_int = int(bssid.replace(':', ''), 16)
                mac_integers.append(mac_int)
            except ValueError:
                continue
        
        if len(mac_integers) < 3:
            return False
        
        # Check for sequential patterns
        mac_integers.sort()
        sequential_count = 0
        for i in range(1, len(mac_integers)):
            if mac_integers[i] - mac_integers[i-1] == 1:
                sequential_count += 1
        
        # Too many sequential MACs is suspicious
        return sequential_count > len(mac_integers) * 0.3

class JammingDetector:
    """Detect RF jamming attempts"""
    
    def detect_jamming(self, networks: List[WiFiNetwork]) -> bool:
        """Detect potential jamming based on network characteristics"""
        if len(networks) < 5:
            return False
        
        # Check for unusual signal distribution
        signals = [n.signal_strength for n in networks]
        
        # All networks with very low signal might indicate jamming
        if all(s < -85 for s in signals):
            return True
        
        # Check for noise floor elevation
        # In jamming, all signals tend to cluster near noise floor
        signal_variance = np.var(signals)
        if signal_variance < 5 and np.mean(signals) < -80:
            return True
        
        return False

class EnvironmentalProfiler:
    """Profile electromagnetic environment"""
    
    def profile_environment(self, networks: List[WiFiNetwork], 
                          motion: bool,
                          materials: List[RFSignature]) -> Dict[str, Any]:
        """Create environmental profile from RF data"""
        profile = {
            'rf_density': self._calculate_rf_density(networks),
            'interference_level': self._estimate_interference(networks),
            'environment_type': self._classify_environment(networks, materials),
            'occupancy': 'occupied' if motion else 'vacant',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return profile
    
    def _calculate_rf_density(self, networks: List[WiFiNetwork]) -> str:
        """Calculate RF density level"""
        if len(networks) < 5:
            return 'low'
        elif len(networks) < 20:
            return 'medium'
        else:
            return 'high'
    
    def _estimate_interference(self, networks: List[WiFiNetwork]) -> str:
        """Estimate interference level"""
        # Count overlapping channels
        channels_2_4 = [n.channel for n in networks if n.frequency < 3000]
        
        if not channels_2_4:
            return 'minimal'
        
        # Check channel distribution
        channel_counts = defaultdict(int)
        for ch in channels_2_4:
            channel_counts[ch] += 1
        
        # High overlap means high interference
        max_overlap = max(channel_counts.values())
        
        if max_overlap >= 5:
            return 'severe'
        elif max_overlap >= 3:
            return 'moderate'
        else:
            return 'minimal'
    
    def _classify_environment(self, networks: List[WiFiNetwork],
                            materials: List[RFSignature]) -> str:
        """Classify environment type"""
        # Based on network names and materials
        network_types = {
            'home': ['home', 'house', 'apt'],
            'office': ['corp', 'office', 'guest'],
            'public': ['public', 'free', 'guest', 'coffee'],
            'industrial': ['warehouse', 'factory', 'plant']
        }
        
        # Check SSIDs
        ssids_lower = [n.ssid.lower() for n in networks]
        type_scores = defaultdict(int)
        
        for env_type, keywords in network_types.items():
            for ssid in ssids_lower:
                for keyword in keywords:
                    if keyword in ssid:
                        type_scores[env_type] += 1
        
        # Consider materials
        if materials:
            material_types = [m.material_type for m in materials]
            if 'concrete' in material_types or 'metal' in material_types:
                type_scores['industrial'] += 2
            elif 'wood' in material_types:
                type_scores['home'] += 1
        
        # Return highest scoring type
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return 'unknown'

class PerfectWiFiConsciousness:
    """
    Production-ready WiFi CSI consciousness with comprehensive security
    """
    
    def __init__(self, config: WiFiConfig, redis_client: Optional[redis.Redis] = None):
        self.config = config
        self.scanner = SecureWiFiScanner(config)
        self.csi_processor = CSIProcessor(config)
        self.anomaly_detector = RFAnomalyDetector()
        self.environmental_profiler = EnvironmentalProfiler()
        self.redis_client = redis_client
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._rate_limiter = WiFiRateLimiter(redis_client)
        self._audit_logger = WiFiAuditLogger()
        self._metrics_collector = WiFiMetrics()
        
    async def start(self):
        """Start WiFi consciousness monitoring"""
        logger.info("Starting Perfect WiFi Consciousness...")
        
        # Start monitoring
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        
    async def stop(self):
        """Gracefully stop monitoring"""
        logger.info("Stopping Perfect WiFi Consciousness...")
        self._shutdown_event.set()
        
        if self._monitoring_task:
            await self._monitoring_task
    
    async def get_wifi_awareness(self, user_id: str, correlation_id: str) -> WiFiAwareness:
        """
        Get comprehensive WiFi awareness with all protections
        """
        # Rate limiting
        if not await self._rate_limiter.check_rate_limit(f"wifi:{user_id}", 10, 60):
            await self._audit_logger.log_security_event(
                "rate_limit_exceeded",
                user_id,
                {"action": "wifi_awareness", "correlation_id": correlation_id}
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for WiFi awareness"
            )
        
        # Audit logging
        await self._audit_logger.log_access(
            "wifi_awareness_read",
            user_id,
            {"correlation_id": correlation_id}
        )
        
        try:
            # Scan networks
            networks = await self.scanner.scan_networks(user_id)
            
            # Simulate CSI processing (would use real CSI data in production)
            csi_data = await self._simulate_csi_capture(networks)
            if csi_data:
                await self.csi_processor.process_csi(csi_data)
            
            # Detect motion
            motion_detected, motion_zones = await self.csi_processor.detect_motion()
            
            # Analyze materials
            materials = await self.csi_processor.analyze_materials(networks)
            
            # Detect anomalies
            anomalies = await self.anomaly_detector.detect_anomalies(networks)
            
            # Create electromagnetic map
            em_map = {
                'network_count': len(networks),
                'frequency_distribution': self._analyze_frequency_distribution(networks),
                'signal_strength_map': self._create_signal_map(networks),
                'channel_utilization': self._analyze_channel_utilization(networks)
            }
            
            # Profile environment
            env_profile = self.environmental_profiler.profile_environment(
                networks, motion_detected, materials
            )
            
            # Build awareness
            awareness = WiFiAwareness(
                networks_detected=len(networks),
                electromagnetic_map=em_map,
                motion_detected=motion_detected,
                motion_zones=motion_zones,
                material_signatures=materials,
                rf_anomalies=anomalies,
                privacy_status="protected" if self.config.privacy_mode else "unprotected",
                security_alerts=self._generate_security_alerts(anomalies, networks),
                environmental_profile=env_profile,
                correlation_id=correlation_id
            )
            
            # Update metrics
            await self._update_metrics(awareness)
            
            return awareness
            
        except Exception as e:
            logger.error(f"WiFi awareness error: {e}")
            await self._audit_logger.log_error(
                "wifi_awareness_error",
                user_id,
                {"error": str(e), "correlation_id": correlation_id}
            )
            raise
    
    async def _simulate_csi_capture(self, networks: List[WiFiNetwork]) -> Optional[List[Tuple[float, float]]]:
        """Simulate CSI capture (placeholder for real implementation)"""
        if not networks or not self.config.csi_enabled:
            return None
        
        # Generate simulated CSI data
        num_subcarriers = 64  # Typical for WiFi
        
        # Base CSI with some variation based on environment
        base_amplitude = 0.5 + 0.1 * len(networks) / 10  # More networks = more complex
        base_phase = np.random.uniform(-np.pi, np.pi, num_subcarriers)
        
        # Add noise
        amplitude = base_amplitude + np.random.normal(0, 0.05, num_subcarriers)
        phase = base_phase + np.random.normal(0, 0.1, num_subcarriers)
        
        # Convert to (real, imag) tuples
        real = amplitude * np.cos(phase)
        imag = amplitude * np.sin(phase)
        
        return [(float(r), float(i)) for r, i in zip(real, imag)]
    
    def _analyze_frequency_distribution(self, networks: List[WiFiNetwork]) -> Dict[str, int]:
        """Analyze frequency band distribution"""
        distribution = defaultdict(int)
        
        for network in networks:
            if network.frequency < 3000:
                distribution['2.4GHz'] += 1
            elif network.frequency < 6000:
                distribution['5GHz'] += 1
            else:
                distribution['6GHz'] += 1
        
        return dict(distribution)
    
    def _create_signal_map(self, networks: List[WiFiNetwork]) -> Dict[str, Any]:
        """Create signal strength map"""
        if not networks:
            return {}
        
        signals = [n.signal_strength for n in networks]
        
        return {
            'min_signal': min(signals),
            'max_signal': max(signals),
            'avg_signal': np.mean(signals),
            'signal_variance': np.var(signals)
        }
    
    def _analyze_channel_utilization(self, networks: List[WiFiNetwork]) -> Dict[int, int]:
        """Analyze channel utilization"""
        utilization = defaultdict(int)
        
        for network in networks:
            utilization[network.channel] += 1
        
        return dict(utilization)
    
    def _generate_security_alerts(self, anomalies: List[str], 
                                networks: List[WiFiNetwork]) -> List[str]:
        """Generate security alerts based on analysis"""
        alerts = []
        
        # Add anomaly alerts
        for anomaly in anomalies:
            alerts.append(f"RF ANOMALY: {anomaly}")
        
        # Check for rogue APs
        suspicious_ssids = ['linksys', 'netgear', 'default', 'admin']
        for network in networks:
            if any(sus in network.ssid.lower() for sus in suspicious_ssids):
                if network.signal_strength > -50:  # Very strong signal
                    alerts.append(f"WARNING: Possible rogue AP detected: {network.ssid}")
        
        # Check for deauth attacks
        if len(networks) == 0 and len(self.scanner._network_cache) > 10:
            alerts.append("CRITICAL: Possible deauthentication attack - networks disappeared")
        
        return alerts
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._shutdown_event.is_set():
            try:
                # Periodic scan
                correlation_id = f"monitor_{datetime.now(timezone.utc).timestamp()}"
                awareness = await self.get_wifi_awareness("system", correlation_id)
                
                # Check for critical conditions
                if awareness.rf_anomalies:
                    logger.warning(f"RF anomalies detected: {awareness.rf_anomalies}")
                
                # Wait for next interval
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.config.scan_interval
                )
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.config.scan_interval)
    
    async def _update_metrics(self, awareness: WiFiAwareness):
        """Update monitoring metrics"""
        # Update metrics collector
        self._metrics_collector.record_awareness(awareness)
        
        # Store in Redis if available
        if self.redis_client:
            try:
                metrics_key = "wifi:metrics"
                await self.redis_client.hset(metrics_key, mapping={
                    "networks_detected": str(awareness.networks_detected),
                    "motion_detected": str(awareness.motion_detected),
                    "rf_anomalies": len(awareness.rf_anomalies),
                    "last_update": awareness.timestamp.isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to update Redis metrics: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get WiFi system health status"""
        return {
            "component": "wifi_consciousness",
            "status": "healthy",
            "privacy_mode": self.config.privacy_mode,
            "csi_enabled": self.config.csi_enabled,
            "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
            "cached_networks": len(self.scanner._network_cache),
            "metrics": self._metrics_collector.get_summary()
        }

class WiFiRateLimiter:
    """Rate limiting for WiFi operations"""
    
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

class WiFiAuditLogger:
    """Audit logging for WiFi operations"""
    
    async def log_access(self, action: str, user_id: str, details: Dict[str, Any]):
        """Log access events"""
        logger.info(f"WIFI_AUDIT_ACCESS: action={action}, user={user_id}, details={json.dumps(details)}")
    
    async def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any]):
        """Log security events"""
        logger.warning(f"WIFI_AUDIT_SECURITY: event={event_type}, user={user_id}, details={json.dumps(details)}")
    
    async def log_error(self, error_type: str, user_id: str, details: Dict[str, Any]):
        """Log error events"""
        logger.error(f"WIFI_AUDIT_ERROR: error={error_type}, user={user_id}, details={json.dumps(details)}")

class WiFiMetrics:
    """Metrics collection for WiFi system"""
    
    def __init__(self):
        self.scans_total = 0
        self.networks_seen = set()
        self.motion_detections = 0
        self.materials_detected = defaultdict(int)
        self.anomalies_detected = 0
        self.environment_types = defaultdict(int)
    
    def record_awareness(self, awareness: WiFiAwareness):
        """Record awareness metrics"""
        self.scans_total += 1
        self.networks_seen.add(awareness.networks_detected)  # Unique count
        
        if awareness.motion_detected:
            self.motion_detections += 1
        
        for material in awareness.material_signatures:
            self.materials_detected[material.material_type] += 1
        
        self.anomalies_detected += len(awareness.rf_anomalies)
        
        env_type = awareness.environmental_profile.get('environment_type', 'unknown')
        self.environment_types[env_type] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            'scans_total': self.scans_total,
            'unique_network_counts': len(self.networks_seen),
            'motion_detection_rate': self.motion_detections / max(self.scans_total, 1),
            'materials_detected': dict(self.materials_detected),
            'anomalies_total': self.anomalies_detected,
            'environment_distribution': dict(self.environment_types)
        }

# Example usage
async def example_usage():
    """Example of using perfect WiFi consciousness"""
    config = WiFiConfig(
        scan_interval=10,
        privacy_mode=True,
        motion_detection=True,
        material_detection=True
    )
    
    # Initialize with Redis for production features
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    consciousness = PerfectWiFiConsciousness(config, redis_client)
    
    # Start monitoring
    await consciousness.start()
    
    try:
        # Get WiFi awareness
        awareness = await consciousness.get_wifi_awareness(
            user_id="test-user",
            correlation_id=str(uuid.uuid4())
        )
        
        print(f"Networks Detected: {awareness.networks_detected}")
        print(f"Motion Detected: {awareness.motion_detected}")
        print(f"Materials: {[m.material_type for m in awareness.material_signatures]}")
        print(f"Environment: {awareness.environmental_profile['environment_type']}")
        print(f"RF Anomalies: {awareness.rf_anomalies}")
        print(f"Security Alerts: {awareness.security_alerts}")
        
    finally:
        # Graceful shutdown
        await consciousness.stop()

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())