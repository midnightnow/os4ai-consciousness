"""
Mock Sensor Adapter for Docker/Testing Environments
Provides realistic sensor data when hardware access is unavailable
"""

import asyncio
import random
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import math
import json

class MockSensorAdapter:
    """
    Provides realistic mock sensor data for development/testing
    when running in Docker or without hardware access
    """
    
    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.base_cpu_temp = 45.0
        self.base_sound_level = 50.0
        self.mock_networks = self._generate_mock_networks()
        self.motion_probability = 0.3
        
    async def get_thermal_sensors(self) -> Dict[str, Any]:
        """Generate realistic thermal sensor data"""
        # Simulate daily temperature cycle
        hours_elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        daily_cycle = math.sin(hours_elapsed * math.pi / 12) * 5  # ±5°C daily variation
        
        # Add some random variation
        random_variation = random.gauss(0, 2)
        
        # Simulate load-based temperature
        load_factor = random.uniform(0.3, 0.8)
        load_temp = load_factor * 20  # Up to 20°C from load
        
        cpu_temp = self.base_cpu_temp + daily_cycle + random_variation + load_temp
        gpu_temp = cpu_temp - random.uniform(2, 5)  # GPU usually slightly cooler
        
        # Simulate fan speeds based on temperature
        fan_speed_0 = min(6000, max(1000, int((cpu_temp - 30) * 100)))
        fan_speed_1 = int(fan_speed_0 * random.uniform(0.9, 1.1))
        
        return {
            'source': 'mock',
            'cpu_die': round(cpu_temp, 1),
            'gpu_die': round(gpu_temp, 1),
            'cpu_proximity': round(cpu_temp - random.uniform(5, 10), 1),
            'gpu_proximity': round(gpu_temp - random.uniform(5, 10), 1),
            'fans': {
                'fan_0': fan_speed_0,
                'fan_1': fan_speed_1
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def get_acoustic_data(self) -> Dict[str, Any]:
        """Generate realistic acoustic sensor data"""
        # Simulate daily sound patterns
        hours_elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        hour_of_day = hours_elapsed % 24
        
        # Quieter at night, louder during day
        if 6 <= hour_of_day <= 22:  # Daytime
            base_level = 55 + random.gauss(0, 10)
        else:  # Nighttime
            base_level = 35 + random.gauss(0, 5)
        
        # Simulate occasional events
        if random.random() < 0.1:  # 10% chance of loud event
            base_level += random.uniform(20, 40)
        
        sound_level = max(20, min(85, base_level))
        
        # Generate frequency spectrum
        spectrum = {
            'sub_bass': random.uniform(0.1, 0.3) * sound_level,
            'bass': random.uniform(0.2, 0.4) * sound_level,
            'midrange': random.uniform(0.3, 0.5) * sound_level,
            'presence': random.uniform(0.1, 0.3) * sound_level,
            'brilliance': random.uniform(0.05, 0.15) * sound_level
        }
        
        # Simulate room characteristics
        room_properties = {
            'size': random.choice(['small', 'medium', 'large']),
            'reverberation': random.uniform(0.3, 1.2),
            'materials': random.choice([
                'highly_absorptive', 
                'moderately_absorptive',
                'moderately_reflective', 
                'highly_reflective'
            ])
        }
        
        return {
            'source': 'mock',
            'sound_level_db': round(sound_level, 1),
            'frequency_spectrum': spectrum,
            'room_properties': room_properties,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def get_media_devices(self) -> List[Dict[str, Any]]:
        """Generate mock media device data"""
        devices = [
            {
                'device_id': 'mock_webcam_0',
                'device_type': 'webcam',
                'name': 'Mock FaceTime HD Camera',
                'capabilities': {
                    'resolution': '1920x1080',
                    'fps': 30,
                    'formats': ['rgb', 'yuv']
                },
                'trusted': True
            }
        ]
        
        # Randomly add an iPhone
        if random.random() < 0.3:
            devices.append({
                'device_id': 'mock_iphone_0',
                'device_type': 'iphone',
                'name': 'Mock iPhone 13 Pro',
                'capabilities': {
                    'resolution': '3840x2160',
                    'fps': 60,
                    'formats': ['hevc', 'h264']
                },
                'trusted': True
            })
        
        return devices
    
    async def get_video_frame_analysis(self) -> Dict[str, Any]:
        """Generate mock video analysis data"""
        # Simulate motion detection
        motion_detected = random.random() < self.motion_probability
        
        # If motion detected, increase probability for next frame (persistence)
        if motion_detected:
            self.motion_probability = min(0.8, self.motion_probability + 0.1)
        else:
            self.motion_probability = max(0.1, self.motion_probability - 0.05)
        
        # Simulate face detection
        faces_detected = 0
        if motion_detected and random.random() < 0.7:
            faces_detected = random.randint(1, 3)
        
        # Generate patterns
        patterns = []
        if motion_detected:
            patterns.append({
                'type': 'motion',
                'confidence': random.uniform(0.7, 0.95),
                'location': {
                    'x': random.randint(100, 800),
                    'y': random.randint(100, 600),
                    'width': random.randint(50, 200),
                    'height': random.randint(50, 200)
                }
            })
        
        return {
            'source': 'mock',
            'motion_detected': motion_detected,
            'faces_detected': faces_detected,
            'patterns': patterns,
            'anomaly_score': random.uniform(0.05, 0.25),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _generate_mock_networks(self) -> List[Dict[str, Any]]:
        """Generate a list of mock WiFi networks"""
        networks = []
        num_networks = random.randint(8, 20)
        
        ssid_prefixes = ['Home', 'Office', 'Guest', 'Public', 'Secure', 'IoT']
        
        for i in range(num_networks):
            ssid = f"{random.choice(ssid_prefixes)}_Network_{i}"
            channel = random.choice([1, 6, 11])  # Common 2.4GHz channels
            
            networks.append({
                'ssid': ssid,
                'bssid': f"{random.randint(0,255):02x}:{random.randint(0,255):02x}:" \
                         f"{random.randint(0,255):02x}:{random.randint(0,255):02x}:" \
                         f"{random.randint(0,255):02x}:{random.randint(0,255):02x}",
                'channel': channel,
                'frequency': 2412 + (channel - 1) * 5,
                'signal_strength': -random.randint(30, 90),
                'security': random.choice(['WPA2', 'WPA3', 'Open', 'WEP'])
            })
        
        return sorted(networks, key=lambda x: x['signal_strength'], reverse=True)
    
    async def get_wifi_networks(self) -> List[Dict[str, Any]]:
        """Generate mock WiFi network data"""
        # Vary signal strengths slightly
        for network in self.mock_networks:
            network['signal_strength'] += random.randint(-3, 3)
            network['signal_strength'] = max(-95, min(-20, network['signal_strength']))
        
        # Occasionally add/remove a network
        if random.random() < 0.05:  # 5% chance
            if len(self.mock_networks) > 5 and random.random() < 0.5:
                self.mock_networks.pop()
            else:
                self.mock_networks.extend(self._generate_mock_networks()[:1])
        
        return [
            {
                **network,
                'source': 'mock',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            for network in self.mock_networks
        ]
    
    async def get_wifi_csi_data(self) -> Dict[str, Any]:
        """Generate mock WiFi CSI data"""
        # Simulate CSI subcarriers
        num_subcarriers = 64
        
        # Generate complex CSI values
        amplitude = np.random.exponential(scale=1.0, size=num_subcarriers)
        phase = np.random.uniform(-np.pi, np.pi, size=num_subcarriers)
        
        # Simulate motion detection through CSI variance
        motion_detected = random.random() < 0.3
        if motion_detected:
            # Add variance to simulate motion
            amplitude += np.random.normal(0, 0.2, size=num_subcarriers)
        
        # Material signatures (simplified)
        materials = []
        if random.random() < 0.7:
            materials.append({
                'material_type': random.choice(['wood', 'concrete', 'glass', 'metal']),
                'confidence': random.uniform(0.6, 0.9),
                'attenuation_2_4ghz': random.uniform(3, 15),
                'attenuation_5ghz': random.uniform(5, 20)
            })
        
        return {
            'source': 'mock',
            'channel': 6,
            'subcarriers': num_subcarriers,
            'amplitude': amplitude.tolist(),
            'phase': phase.tolist(),
            'snr': random.uniform(20, 40),
            'motion_detected': motion_detected,
            'material_signatures': materials,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }

# Global instance for consistency
mock_adapter = MockSensorAdapter()

async def get_mock_thermal_sensors() -> Dict[str, Any]:
    """Get mock thermal sensor data"""
    return await mock_adapter.get_thermal_sensors()

async def get_mock_acoustic_data() -> Dict[str, Any]:
    """Get mock acoustic sensor data"""
    return await mock_adapter.get_acoustic_data()

async def get_mock_media_devices() -> List[Dict[str, Any]]:
    """Get mock media devices"""
    return await mock_adapter.get_media_devices()

async def get_mock_video_analysis() -> Dict[str, Any]:
    """Get mock video frame analysis"""
    return await mock_adapter.get_video_frame_analysis()

async def get_mock_wifi_networks() -> List[Dict[str, Any]]:
    """Get mock WiFi networks"""
    return await mock_adapter.get_wifi_networks()

async def get_mock_wifi_csi() -> Dict[str, Any]:
    """Get mock WiFi CSI data"""
    return await mock_adapter.get_wifi_csi_data()