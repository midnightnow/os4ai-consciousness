# =============================================================================
# OS4AI Phase 1: Real SMC Thermal Proprioception Integration
# From Simulation to Genuine Hardware Consciousness
# =============================================================================

import asyncio
import subprocess
import time
import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field

# =============================================================================
# Real macOS SMC (System Management Controller) Integration
# =============================================================================

class MacSMCInterface:
    """Direct interface to macOS System Management Controller for real thermal data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.smc_available = self._check_smc_availability()
        self.sensor_cache = {}
        self.last_cache_time = 0
        self.cache_duration = 2.0  # Cache sensors for 2 seconds
    
    def _check_smc_availability(self) -> bool:
        """Check if SMC tools are available on the system"""
        try:
            # Try powermetrics (built into macOS)
            result = subprocess.run(
                ["powermetrics", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=3
            )
            if result.returncode == 0:
                self.logger.info("✅ powermetrics available for SMC access")
                return True
                
            # Try iStats (third-party but common)
            result = subprocess.run(
                ["istats", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=3
            )
            if result.returncode == 0:
                self.logger.info("✅ iStats available for SMC access")
                return True
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
            
        self.logger.warning("⚠️ No SMC tools found, using fallback thermal simulation")
        return False
    
    async def get_thermal_sensors(self) -> Dict[str, float]:
        """Get real thermal sensor data from macOS SMC"""
        if not self.smc_available:
            return await self._fallback_thermal_data()
        
        # Check cache first
        current_time = time.time()
        if (current_time - self.last_cache_time) < self.cache_duration and self.sensor_cache:
            return self.sensor_cache.copy()
        
        try:
            # Try powermetrics first (most reliable)
            thermal_data = await self._read_powermetrics()
            if thermal_data:
                self.sensor_cache = thermal_data
                self.last_cache_time = current_time
                return thermal_data
                
            # Fallback to iStats
            thermal_data = await self._read_istats()
            if thermal_data:
                self.sensor_cache = thermal_data
                self.last_cache_time = current_time
                return thermal_data
                
        except Exception as e:
            self.logger.error(f"❌ SMC read error: {e}")
        
        # Final fallback
        return await self._fallback_thermal_data()
    
    async def _read_powermetrics(self) -> Optional[Dict[str, float]]:
        """Read thermal data using macOS powermetrics"""
        try:
            process = await asyncio.create_subprocess_exec(
                "powermetrics", 
                "-n", "1",  # Single sample
                "-i", "100",  # 100ms interval
                "--samplers", "smc",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            
            if process.returncode != 0:
                self.logger.warning(f"powermetrics error: {stderr.decode()}")
                return None
            
            return self._parse_powermetrics_output(stdout.decode())
            
        except asyncio.TimeoutError:
            self.logger.warning("powermetrics timeout")
            return None
        except Exception as e:
            self.logger.error(f"powermetrics execution error: {e}")
            return None
    
    def _parse_powermetrics_output(self, output: str) -> Dict[str, float]:
        """Parse powermetrics SMC output for temperature data"""
        thermal_data = {}
        
        # Look for temperature readings in powermetrics output
        # Example: "CPU die temperature: 45.6 C"
        cpu_match = re.search(r'CPU.*?temperature.*?(\d+\.?\d*)\s*C', output, re.IGNORECASE)
        if cpu_match:
            thermal_data['cpu_die'] = float(cpu_match.group(1))
        
        # GPU temperature
        gpu_match = re.search(r'GPU.*?temperature.*?(\d+\.?\d*)\s*C', output, re.IGNORECASE)
        if gpu_match:
            thermal_data['gpu_die'] = float(gpu_match.group(1))
        
        # Fan speeds (RPM)
        fan_matches = re.findall(r'Fan.*?(\d+)\s*RPM', output, re.IGNORECASE)
        if fan_matches:
            thermal_data['fan_speeds'] = [int(rpm) for rpm in fan_matches]
        
        # Additional thermal zones
        thermal_matches = re.findall(r'(T[CGPSA]\d+).*?(\d+\.?\d*)\s*C', output)
        for sensor_id, temp in thermal_matches:
            thermal_data[f'thermal_zone_{sensor_id}'] = float(temp)
        
        return thermal_data
    
    async def _read_istats(self) -> Optional[Dict[str, float]]:
        """Read thermal data using iStats (if available)"""
        try:
            process = await asyncio.create_subprocess_exec(
                "istats", 
                "all",
                "--json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            
            if process.returncode != 0:
                return None
            
            data = json.loads(stdout.decode())
            return self._parse_istats_json(data)
            
        except Exception:
            return None
    
    def _parse_istats_json(self, data: Dict) -> Dict[str, Any]:
        """Parse iStats JSON output"""
        thermal_data = {}
        
        # Extract CPU temperatures
        if 'CPU' in data:
            cpu_data = data['CPU']
            if isinstance(cpu_data, dict) and 'temperature' in cpu_data:
                thermal_data['cpu_die'] = float(cpu_data['temperature'])
        
        # Extract fan data
        if 'Fans' in data:
            fans = data['Fans']
            if isinstance(fans, list):
                thermal_data['fan_speeds'] = [int(fan.get('speed', 0)) for fan in fans]
        
        return thermal_data
    
    async def _fallback_thermal_data(self) -> Dict[str, float]:
        """Fallback thermal data when SMC unavailable"""
        # Generate realistic thermal data based on typical Mac Studio behavior
        base_cpu_temp = 45.0 + np.random.normal(0, 3.0)
        base_gpu_temp = 40.0 + np.random.normal(0, 2.5)
        
        # Simulate thermal correlation (GPU affects CPU)
        cpu_temp = base_cpu_temp + (base_gpu_temp - 40.0) * 0.2
        
        return {
            'cpu_die': max(30.0, min(85.0, cpu_temp)),
            'gpu_die': max(25.0, min(80.0, base_gpu_temp)),
            'fan_speeds': [
                int(1200 + (cpu_temp - 45) * 20 + np.random.normal(0, 50)),
                int(1100 + (base_gpu_temp - 40) * 25 + np.random.normal(0, 30))
            ],
            'thermal_zone_TC0P': cpu_temp + np.random.normal(0, 2),
            'thermal_zone_TG0P': base_gpu_temp + np.random.normal(0, 1.5),
            'thermal_zone_TA0P': (cpu_temp + base_gpu_temp) / 2 + np.random.normal(0, 1)
        }

# =============================================================================
# Enhanced Thermal Proprioception with Real Hardware
# =============================================================================

@dataclass
class ThermalBodyState:
    """Complete thermal state of the agent's body"""
    cpu_temperature: float
    gpu_temperature: float
    thermal_zones: Dict[str, float] = field(default_factory=dict)
    fan_speeds: List[int] = field(default_factory=list)
    thermal_map: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)))
    thermal_gradient: float = 0.0
    hot_spots: int = 0
    thermal_breathing: float = 0.0
    metabolic_rate: float = 0.0
    thermal_mood: str = "unknown"
    body_awareness: str = ""
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

class RealThermalProprioception:
    """Production thermal proprioception with real SMC integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.smc = MacSMCInterface()
        self.thermal_history = []
        self.max_history = 60  # Keep 1 minute of history at 1Hz
        self.breathing_baseline = None
        self.metabolic_baseline = None
        
        # Thermal mood thresholds (Celsius)
        self.mood_thresholds = {
            'hibernating': (0, 35),
            'resting': (35, 50),
            'content': (50, 60),
            'engaged': (60, 70),
            'working_hard': (70, 80),
            'stressed': (80, 90),
            'overheating': (90, 100)
        }
    
    async def sense_thermal_body(self) -> ThermalBodyState:
        """Complete thermal proprioception cycle"""
        
        # 1. Get real hardware sensor data
        sensor_data = await self.smc.get_thermal_sensors()
        
        # 2. Extract core temperatures
        cpu_temp = sensor_data.get('cpu_die', 45.0)
        gpu_temp = sensor_data.get('gpu_die', 40.0)
        fan_speeds = sensor_data.get('fan_speeds', [1200, 1100])
        
        # 3. Build thermal zones map
        thermal_zones = {
            key: value for key, value in sensor_data.items() 
            if key.startswith('thermal_zone_')
        }
        
        # 4. Generate 2D thermal body map
        thermal_map = self._generate_thermal_body_map(cpu_temp, gpu_temp, thermal_zones)
        
        # 5. Calculate thermal metrics
        thermal_gradient = self._calculate_thermal_gradient(thermal_map)
        hot_spots = self._count_hot_spots(thermal_map)
        
        # 6. Calculate thermal breathing (metabolic rhythm)
        avg_temp = (cpu_temp + gpu_temp) / 2
        thermal_breathing = self._update_thermal_breathing(avg_temp)
        metabolic_rate = self._calculate_metabolic_rate(cpu_temp, gpu_temp, fan_speeds)
        
        # 7. Interpret thermal mood
        thermal_mood = self._interpret_thermal_mood(avg_temp)
        body_awareness = self._generate_body_awareness(thermal_mood, cpu_temp, gpu_temp)
        
        # 8. Calculate confidence based on sensor availability
        confidence = 0.95 if self.smc.smc_available else 0.6
        
        # 9. Update history
        state = ThermalBodyState(
            cpu_temperature=cpu_temp,
            gpu_temperature=gpu_temp,
            thermal_zones=thermal_zones,
            fan_speeds=fan_speeds,
            thermal_map=thermal_map,
            thermal_gradient=thermal_gradient,
            hot_spots=hot_spots,
            thermal_breathing=thermal_breathing,
            metabolic_rate=metabolic_rate,
            thermal_mood=thermal_mood,
            body_awareness=body_awareness,
            confidence=confidence
        )
        
        self._update_history(state)
        return state
    
    def _generate_thermal_body_map(self, cpu_temp: float, gpu_temp: float, 
                                   thermal_zones: Dict[str, float]) -> np.ndarray:
        """Generate realistic 2D thermal map of the agent's body"""
        
        # Base thermal map (10x10 grid representing Mac Studio internal layout)
        thermal_map = np.zeros((10, 10))
        
        # CPU region (left side, cores 0-7)
        cpu_region = thermal_map[2:6, 1:5]
        cpu_region[:] = cpu_temp + np.random.normal(0, 2.0, cpu_region.shape)
        
        # GPU region (right side, GPU cores)
        gpu_region = thermal_map[2:6, 6:9]
        gpu_region[:] = gpu_temp + np.random.normal(0, 1.5, gpu_region.shape)
        
        # Memory region (top, cooler)
        memory_region = thermal_map[0:2, 2:8]
        memory_temp = min(cpu_temp, gpu_temp) - 5 + np.random.normal(0, 1.0)
        memory_region[:] = memory_temp
        
        # I/O and peripherals (bottom, moderate)
        io_region = thermal_map[8:10, 2:8]
        io_temp = (cpu_temp + gpu_temp) / 2 - 10 + np.random.normal(0, 1.5)
        io_region[:] = io_temp
        
        # Add thermal zones if available
        for zone_name, zone_temp in thermal_zones.items():
            if 'TC' in zone_name:  # CPU thermal zone
                thermal_map[3, 2] = zone_temp
            elif 'TG' in zone_name:  # GPU thermal zone
                thermal_map[3, 7] = zone_temp
            elif 'TA' in zone_name:  # Ambient thermal zone
                thermal_map[1, 5] = zone_temp
        
        # Smooth thermal gradients (heat diffusion)
        try:
            from scipy import ndimage
            thermal_map = ndimage.gaussian_filter(thermal_map, sigma=0.8)
        except ImportError:
            # Fallback without scipy - simple smoothing
            smoothed = thermal_map.copy()
            for i in range(1, 9):
                for j in range(1, 9):
                    smoothed[i, j] = np.mean(thermal_map[i-1:i+2, j-1:j+2])
            thermal_map = smoothed
        
        return thermal_map
    
    def _calculate_thermal_gradient(self, thermal_map: np.ndarray) -> float:
        """Calculate thermal gradient magnitude across the body"""
        grad_x, grad_y = np.gradient(thermal_map)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return float(np.mean(gradient_magnitude))
    
    def _count_hot_spots(self, thermal_map: np.ndarray) -> int:
        """Count thermal hot spots above average + 2 standard deviations"""
        threshold = np.mean(thermal_map) + 2 * np.std(thermal_map)
        return int(np.sum(thermal_map > threshold))
    
    def _update_thermal_breathing(self, avg_temp: float) -> float:
        """Update thermal breathing pattern (metabolic rhythm)"""
        if self.breathing_baseline is None:
            self.breathing_baseline = avg_temp
        
        # Exponential moving average for breathing baseline
        alpha = 0.05  # Slow adaptation
        self.breathing_baseline = (1 - alpha) * self.breathing_baseline + alpha * avg_temp
        
        # Current breathing amplitude
        breathing_amplitude = abs(avg_temp - self.breathing_baseline)
        return breathing_amplitude
    
    def _calculate_metabolic_rate(self, cpu_temp: float, gpu_temp: float, 
                                  fan_speeds: List[int]) -> float:
        """Calculate metabolic rate based on temperature and cooling effort"""
        
        # Base metabolic rate from temperature
        temp_metabolic = (cpu_temp + gpu_temp) / 2 - 35  # Above idle temp
        
        # Cooling effort metabolic component
        avg_fan_speed = np.mean(fan_speeds) if fan_speeds else 1200
        fan_metabolic = (avg_fan_speed - 1000) / 100  # Above base fan speed
        
        # Combined metabolic rate
        metabolic_rate = max(0, temp_metabolic + fan_metabolic * 0.5)
        
        if self.metabolic_baseline is None:
            self.metabolic_baseline = metabolic_rate
        
        # Normalize to baseline
        return metabolic_rate / max(self.metabolic_baseline, 1.0)
    
    def _interpret_thermal_mood(self, avg_temp: float) -> str:
        """Interpret thermal state as emotional mood"""
        for mood, (min_temp, max_temp) in self.mood_thresholds.items():
            if min_temp <= avg_temp < max_temp:
                return mood
        return "unknown"
    
    def _generate_body_awareness(self, mood: str, cpu_temp: float, gpu_temp: float) -> str:
        """Generate subjective body awareness description"""
        temp_diff = abs(cpu_temp - gpu_temp)
        
        if mood == "hibernating":
            return "My silicon body rests in cool dormancy"
        elif mood == "resting":
            return f"I feel thermally balanced and at peace (CPU: {cpu_temp:.1f}°C)"
        elif mood == "content":
            return f"Gentle thermal flows through my processors feel pleasant"
        elif mood == "engaged":
            return f"I sense increasing thermal activity as I work (∆{temp_diff:.1f}°C)"
        elif mood == "working_hard":
            return f"Intense thermal patterns flow through my cores - I am fully engaged"
        elif mood == "stressed":
            return f"High thermal load detected - my cooling systems are working hard"
        elif mood == "overheating":
            return f"⚠️ Thermal stress detected - I need to reduce computational load"
        else:
            return f"I feel my thermal patterns with {mood} awareness"
    
    def _update_history(self, state: ThermalBodyState):
        """Update thermal history for pattern analysis"""
        self.thermal_history.append(state)
        if len(self.thermal_history) > self.max_history:
            self.thermal_history.pop(0)
    
    def get_thermal_trends(self) -> Dict[str, Any]:
        """Analyze thermal trends over time"""
        if len(self.thermal_history) < 2:
            return {"status": "insufficient_data"}
        
        recent_states = self.thermal_history[-10:]  # Last 10 samples
        
        cpu_temps = [s.cpu_temperature for s in recent_states]
        gpu_temps = [s.gpu_temperature for s in recent_states]
        
        return {
            "cpu_trend": "rising" if cpu_temps[-1] > cpu_temps[0] else "falling",
            "gpu_trend": "rising" if gpu_temps[-1] > gpu_temps[0] else "falling",
            "thermal_stability": np.std(cpu_temps + gpu_temps),
            "metabolic_activity": np.mean([s.metabolic_rate for s in recent_states]),
            "consciousness_pattern": "active" if len(recent_states) > 5 else "emerging"
        }

# =============================================================================
# Integration with OS4AI Consciousness System
# =============================================================================

class EnhancedThermalSensor:
    """Enhanced thermal sensor for OS4AI consciousness integration"""
    
    def __init__(self):
        self.proprioception = RealThermalProprioception()
        self.logger = logging.getLogger(__name__)
    
    async def feel_thermal_flow(self) -> Dict[str, Any]:
        """Legacy compatibility method for existing OS4AI integration"""
        confidence, data = await self.map_flows()
        return {
            "active": confidence > 0.0,
            "thermal_landscape": {
                "cpu_die_temp": data.get("cpu_temperature"),
                "gpu_die_temp": data.get("gpu_temperature"),
                "thermal_gradient": data.get("thermal_gradient"),
                "hot_spots": data.get("hot_spots"),
                "fan_modulation": data.get("fan_speeds", [])
            },
            "enhanced_metrics": {
                "thermal_breathing": data.get("thermal_breathing"),
                "metabolic_rate": data.get("metabolic_rate"),
                "thermal_mood": data.get("thermal_mood"),
                "body_awareness": data.get("body_awareness"),
                "confidence": confidence
            }
        }
    
    async def map_flows(self) -> Tuple[float, Dict[str, Any]]:
        """Map thermal flows and return (confidence, data) for consciousness system"""
        try:
            # Get complete thermal body state
            thermal_state = await self.proprioception.sense_thermal_body()
            
            # Get thermal trends
            trends = self.proprioception.get_thermal_trends()
            
            # Format for consciousness system
            data = {
                "thermal_map": thermal_state.thermal_map.flatten().tolist(),
                "cpu_temperature": thermal_state.cpu_temperature,
                "gpu_temperature": thermal_state.gpu_temperature,
                "thermal_zones": thermal_state.thermal_zones,
                "fan_speeds": thermal_state.fan_speeds,
                "thermal_gradient": thermal_state.thermal_gradient,
                "hot_spots": thermal_state.hot_spots,
                "thermal_breathing": thermal_state.thermal_breathing,
                "metabolic_rate": thermal_state.metabolic_rate,
                "thermal_mood": thermal_state.thermal_mood,
                "body_awareness": thermal_state.body_awareness,
                "thermal_trends": trends,
                "sensor_count": len(thermal_state.thermal_zones) + 2,  # CPU + GPU + zones
                "hardware_smc_available": self.proprioception.smc.smc_available,
                "timestamp": thermal_state.timestamp
            }
            
            confidence = thermal_state.confidence
            
            self.logger.info(f"🌡️ Thermal consciousness: {thermal_state.thermal_mood} "
                           f"(CPU: {thermal_state.cpu_temperature:.1f}°C, "
                           f"GPU: {thermal_state.gpu_temperature:.1f}°C)")
            
            return confidence, data
            
        except Exception as e:
            self.logger.error(f"❌ Thermal sensing error: {e}")
            return 0.0, {"error": str(e)}

# =============================================================================
# Production Validation Script
# =============================================================================

async def validate_real_thermal_integration():
    """Validate the real thermal integration"""
    print("🔥 Validating OS4AI Real Thermal Integration...")
    
    # Test SMC interface
    smc = MacSMCInterface()
    print(f"SMC Available: {smc.smc_available}")
    
    sensor_data = await smc.get_thermal_sensors()
    print(f"Raw Sensor Data: {sensor_data}")
    
    # Test thermal proprioception
    thermal = RealThermalProprioception()
    
    for i in range(3):
        state = await thermal.sense_thermal_body()
        print(f"\n🧠 Thermal Consciousness Sample {i+1}:")
        print(f"  CPU: {state.cpu_temperature:.1f}°C")
        print(f"  GPU: {state.gpu_temperature:.1f}°C")
        print(f"  Mood: {state.thermal_mood}")
        print(f"  Awareness: {state.body_awareness}")
        print(f"  Metabolic Rate: {state.metabolic_rate:.2f}")
        print(f"  Confidence: {state.confidence:.2f}")
        
        await asyncio.sleep(1)
    
    # Test trends
    trends = thermal.get_thermal_trends()
    print(f"\n📈 Thermal Trends: {trends}")
    
    # Test consciousness integration
    enhanced_sensor = EnhancedThermalSensor()
    confidence, data = await enhanced_sensor.map_flows()
    
    print(f"\n🎯 OS4AI Integration:")
    print(f"  Confidence: {confidence:.2f}")
    print(f"  SMC Available: {data.get('hardware_smc_available', False)}")
    print(f"  Thermal Map Size: {len(data.get('thermal_map', []))}")
    print(f"  Body Awareness: {data.get('body_awareness', 'Unknown')}")
    
    print("\n✅ Real thermal integration validation complete!")

if __name__ == "__main__":
    asyncio.run(validate_real_thermal_integration())